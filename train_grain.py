import os
import random

import albumentations as A  # noqa: N812
import cv2
import grain
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
import torch
from flax import nnx
from tqdm import tqdm

from models.resnet import resnet50

# Configuration
params = {
    "num_epochs": 300,
    "batch_size": 64,
    "target_size": 224,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "seed": 42,
    "num_classes": 16,
    "train_data_path": "./MpoxData/train",
    "val_data_path": "./MpoxData/validation",
    "checkpoint_dir": "/Users/billy/Documents/DLStudy/JaxVision/checkpoints",
}


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)


def load_image(img_path):
    """Load image in RGB format"""
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def create_transforms(target_size, is_training=True) -> A.Compose:  # noqa: FBT002
    """
    为皮肤病图像创建数据增强变换。

    考虑了皮肤病图像的特定需求，旨在保留病灶特征并增加数据多样性。
    """  # noqa: RUF002
    transforms_list = [
        # 固定图像大小，确保所有输入图像尺寸一致  # noqa: RUF003
        A.Resize(height=target_size, width=target_size, p=1.0),
    ]

    if is_training:
        transforms_list.extend(
            [
                # 几何变换：对于皮肤病图像通常是安全的，可以增加模型的旋转和翻转不变性。  # noqa: RUF003
                # 角度限制可以根据实际情况调整，避免过度旋转导致病灶识别困难。  # noqa: RUF003
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(
                    limit=30, p=0.5, border_mode=cv2.BORDER_REFLECT_101
                ),  # 使用 cv2.BORDER_REFLECT_101 填充边缘
                # 颜色抖动：适度调整可以模拟不同光照条件和相机设置。  # noqa: RUF003
                # 对于皮肤病图像，hue（色相）的调整需要特别小心，因为颜色变化可能影响病灶的识别。  # noqa: RUF003
                # 可以考虑减小 hue 的范围或对其进行更精细的控制。
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,  # 减小 hue 的变化范围
                    p=0.5,
                ),
                # 额外的几何变换：  # noqa: RUF003
                # RandomResizedCrop: 模拟不同视角下的病灶，但需注意不要裁掉关键病灶区域。  # noqa: RUF003
                # 对于皮肤病图像，可以考虑使用更保守的裁剪参数，或者只在确认病灶占据图像大部分区域时使用。  # noqa: RUF003
                A.RandomResizedCrop(
                    size=(target_size, target_size),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.33),
                    p=0.3,
                ),
                # Affine变换：可以模拟轻微的透视或畸变，有助于模型泛化。  # noqa: RUF003
                A.Affine(
                    shear=(-10, 10),
                    rotate=(-10, 10),
                    scale=(0.9, 1.1),
                    p=0.3,
                    mode=cv2.BORDER_REFLECT_101,
                ),
                # 模糊：可以模拟图像失焦或低质量图像，增加模型的鲁棒性。  # noqa: RUF003
                # 对于皮肤病图像，过度模糊可能使细节消失，所以需要谨慎。  # noqa: RUF003
                A.GaussianBlur(blur_limit=(3, 7), p=0.1),  # 轻微高斯模糊
                A.MotionBlur(blur_limit=(3, 7), p=0.1),  # 轻微运动模糊
                # 噪声：模拟传感器噪声或图像采集过程中的随机干扰。  # noqa: RUF003
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),  # 添加高斯噪声
                # 弹力变形 (ElasticTransform)：可以生成新的病灶形状，但要注意不要过度变形，以免生成非自然的病灶。  # noqa: RUF003
                # 对于皮肤病图像，需要仔细调整参数，以确保变形后的病灶仍然具有医学合理性。  # noqa: RUF003
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.1,
                ),
                # 网格失真 (GridDistortion)：与弹力变形类似，可以引入局部变形。  # noqa: RUF003
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.3,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.1,
                ),
                # 随机擦除 (RandomErasing)：遮挡部分区域，促使模型学习局部特征而不是依赖整体结构。  # noqa: RUF003
                # 对于病灶较小的图像，需要谨慎使用，避免擦除整个病灶。  # noqa: RUF003
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(target_size * 0.1),
                    max_width=int(target_size * 0.1),
                    min_holes=1,
                    min_height=int(target_size * 0.05),
                    min_width=int(target_size * 0.05),
                    p=0.1,
                ),
            ]
        )

    transforms_list.append(
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet 均值
            std=[0.229, 0.224, 0.225],  # ImageNet 标准差
            max_pixel_value=255.0,
        )
    )

    return A.Compose(transforms_list)


class AlbumentationsTransform(grain.transforms.Map):
    def __init__(self, transforms):
        self.transforms = transforms

    def map(self, element: tuple[np.ndarray, int]) -> tuple[np.ndarray, int]:
        return (self.transforms(image=element[0])["image"], element[1])


class LoadImageMap(grain.transforms.Map):
    def map(self, element: tuple[str, int]) -> tuple[np.ndarray, int]:
        """Load image from path and return as numpy array with label"""
        img_path, label = element
        img = cv2.imread(img_path)
        if img is None:
            msg = f"Image at {img_path} could not be loaded."
            raise ValueError(msg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, label


class CustomImageFolderDataSource:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.samples: list[tuple[str, int]] = []  # 存储 (image_path, class_idx)
        self.classes: list[str] = []  # 存储类别名称

        # 这部分逻辑与 ImageFolder 的内部实现类似，用于遍历目录和映射类别  # noqa: RUF003
        class_to_idx = {}
        idx_to_class = []

        if not os.path.isdir(root_dir):  # noqa: PTH112
            msg = f"Root directory not found: {root_dir}"
            raise FileNotFoundError(msg)

        # 遍历根目录下的每个子目录（代表一个类别）  # noqa: RUF003
        for target_class_name in sorted(os.listdir(root_dir)):  # noqa: PTH208
            class_dir_path = os.path.join(root_dir, target_class_name)  # noqa: PTH118
            if not os.path.isdir(class_dir_path):  # noqa: PTH112
                continue  # 跳过非目录文件

            # 为类别分配一个索引
            if target_class_name not in class_to_idx:
                class_to_idx[target_class_name] = len(class_to_idx)
                idx_to_class.append(target_class_name)

            class_idx = class_to_idx[target_class_name]

            # 遍历类别目录下的所有图片文件
            for img_file_name in sorted(os.listdir(class_dir_path)):  # noqa: PTH208
                img_path = os.path.join(class_dir_path, img_file_name)  # noqa: PTH118
                # 检查文件是否是图片（这里只做了简单的扩展名判断，可以更完善）  # noqa: RUF003
                if os.path.isfile(img_path) and img_file_name.lower().endswith(  # noqa: PTH113
                    (".png", ".jpg", ".jpeg", ".gif", ".bmp")
                ):
                    self.samples.append((img_path, class_idx))

        self.classes = idx_to_class
        if not self.samples:
            msg = f"在目录 '{root_dir}' 中没有找到任何图片。请检查目录结构和文件扩展名。"
            raise RuntimeError(
                msg
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.samples[index]


def create_datasets(params):
    """Create training and validation datasets"""
    train_dataset = CustomImageFolderDataSource(params["train_data_path"])

    val_dataset = CustomImageFolderDataSource(params["val_data_path"])

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, params):
    """Create data loaders"""
    train_loader = (
        grain.MapDataset.source(train_dataset)
        .shuffle(seed=params["seed"])
        .map(LoadImageMap())
        .map(
            AlbumentationsTransform(
                create_transforms(params["target_size"], is_training=True)
            )
        )
        .to_iter_dataset()
        .batch(
            batch_size=params["batch_size"],
            drop_remainder=True,
            batch_fn=lambda ts: (
                np.stack([t[0] for t in ts], axis=0),
                np.stack([t[1] for t in ts], axis=0),
            ),
        )
    )
    val_loader = (
        grain.MapDataset.source(val_dataset)
        .shuffle(seed=params["seed"])
        .map(LoadImageMap())
        .map(
            AlbumentationsTransform(
                create_transforms(params["target_size"], is_training=False)
            )
        )
        .to_iter_dataset()
        .batch(
            batch_size=params["batch_size"],
            drop_remainder=False,
            batch_fn=lambda ts: (
                np.stack([t[0] for t in ts], axis=0),
                np.stack([t[1] for t in ts], axis=0),
            ),
        )
    )

    return train_loader, val_loader


def create_model(seed, num_classes):
    """Create a new model"""
    return resnet50(rngs=nnx.Rngs(seed), num_classes=num_classes)


def save_model(model, path):
    """Save model state"""
    os.makedirs(os.path.dirname(path), exist_ok=True)  # noqa: PTH103, PTH120
    checkpointer = ocp.PyTreeCheckpointer()
    state = nnx.state(model)
    checkpointer.save(path, state)


def load_model(path, num_classes):
    """Load model from checkpoint"""
    model = nnx.eval_shape(lambda: create_model(0, num_classes))
    state = nnx.state(model)

    checkpointer = ocp.PyTreeCheckpointer()
    state = checkpointer.restore(path, item=state)

    nnx.update(model, state)
    return model


def loss_fn(model, batch):
    """Calculate loss and logits"""
    images, labels = batch
    logits = model(images)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()
    return loss, logits


@nnx.jit
def train_step(model, optimizer, metrics, batch):
    """Single training step"""
    # Convert numpy arrays to jnp.array on GPU
    x, y_true = jnp.asarray(batch[0]), jnp.asarray(batch[1])
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, (x, y_true))
    metrics.update(loss=loss, logits=logits, labels=y_true)
    optimizer.update(grads)


@nnx.jit
def eval_step(model, metrics, batch):
    """Single evaluation step"""
    # convert numpy arrays to jnp.array on GPU
    x, y_true = jnp.asarray(batch[0]), jnp.asarray(batch[1])
    loss, logits = loss_fn(model, (x, y_true))
    metrics.update(loss=loss, logits=logits, labels=y_true)


def train_epoch(model, optimizer, metrics, train_loader, epoch_num):
    """Train for one epoch"""
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch_num + 1} Training"):
        train_step(model, optimizer, metrics, batch)

    result = metrics.compute()
    metrics.reset()

    print(f"✅ Train Loss: {result['loss']:.4f}, Acc: {result['accuracy'] * 100:.6f}%")
    return result


def validate_epoch(model, metrics, val_loader, epoch_num):
    """Validate for one epoch"""
    model.eval()
    for batch in tqdm(val_loader, desc=f"Epoch {epoch_num + 1} Validation"):
        eval_step(model, metrics, batch)

    result = metrics.compute()
    metrics.reset()

    print(f"📊 Val Loss: {result['loss']:.4f}, Acc: {result['accuracy'] * 100:.6f}%")
    return result


def update_metrics_history(metrics_history, train_result, val_result):
    """Update metrics history"""
    for k, v in train_result.items():
        metrics_history[f"train_{k}"].append(float(v))

    for k, v in val_result.items():
        metrics_history[f"val_{k}"].append(float(v))


def save_best_model_if_improved(model, val_result, best_acc, epoch_num, checkpoint_dir):
    """Save model if it's the best so far"""
    current_acc = float(val_result["accuracy"])

    if current_acc > best_acc:
        checkpoint_path = os.path.join(  # noqa: PTH118
            checkpoint_dir,
            f"best_model_Epoch_{epoch_num + 1}_Acc_{current_acc:.6f}",
            "state",
        )
        save_model(model, checkpoint_path)
        print(f"🎉 New best model saved with accuracy: {current_acc * 100:.6f}%")
        return current_acc

    return best_acc


def plot_training_metrics(metrics_history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(metrics_history["train_loss"]) + 1)

    # Plot loss
    ax1.plot(epochs, metrics_history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(epochs, metrics_history["val_loss"], label="Val Loss", marker="s")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)  # noqa: FBT003

    # Plot accuracy
    ax2.plot(
        epochs, metrics_history["train_accuracy"], label="Train Accuracy", marker="o"
    )
    ax2.plot(epochs, metrics_history["val_accuracy"], label="Val Accuracy", marker="s")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)  # noqa: FBT003

    plt.tight_layout()
    plt.show()


def print_dataset_info(train_dataset, val_dataset):
    """Print dataset information"""
    print("📊 Dataset Info:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Classes: {train_dataset.classes}")
    print(f"  Number of classes: {len(train_dataset.classes)}")


def main():
    """Main training function"""
    set_seed(params["seed"])

    print("🚀 Starting training with ResNet18...")
    print(f"📋 Configuration: {params}")

    # Create datasets and dataloaders
    print("\n📂 Loading datasets...")
    train_dataset, val_dataset = create_datasets(params)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, params)

    print_dataset_info(train_dataset, val_dataset)

    # Create model and optimizer
    print("\n🏗️ Creating model and optimizer...")
    model = create_model(params["seed"], params["num_classes"])

    optimizer = nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
        ),
    )

    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    # Initialize tracking variables
    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_acc = -1.0

    # Training loop
    print(f"\n🏃 Starting training for {params['num_epochs']} epochs...")

    for epoch in range(params["num_epochs"]):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{params['num_epochs']}")
        print(f"{'=' * 60}")

        # Train and validate
        train_result = train_epoch(model, optimizer, metrics, train_loader, epoch)
        val_result = validate_epoch(model, metrics, val_loader, epoch)

        # Update metrics history
        update_metrics_history(metrics_history, train_result, val_result)

        # Save best model
        best_acc = save_best_model_if_improved(
            model, val_result, best_acc, epoch, params["checkpoint_dir"]
        )

    print("\n🎯 Training completed!")
    print(f"🏆 Best validation accuracy: {best_acc * 100:.6f}%")

    # Plot results
    print("\n📈 Plotting training metrics...")
    plot_training_metrics(metrics_history)


if __name__ == "__main__":
    main()
