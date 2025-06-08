import csv
import os
import random
from pathlib import Path

import albumentations as A  # noqa: N812
import cv2
import grain
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
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
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)


def get_image_extensions() -> tuple[str, ...]:
    return (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


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
    """优化的图像加载器，增加错误处理"""  # noqa: RUF002

    def map(self, element: tuple[str, int]) -> tuple[np.ndarray, int]:
        img_path, label = element

        try:
            img = cv2.imread(img_path)
            if img is None:
                msg = f"无法加载图像: {img_path}"
                raise ValueError(msg)  # noqa: TRY301

            # 检查图像是否损坏
            if img.size == 0:
                msg = f"图像为空: {img_path}"
                raise ValueError(msg)  # noqa: TRY301

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb, label  # noqa: TRY300

        except Exception as e:  # noqa: BLE001
            print(f"警告: 跳过损坏的图像 {img_path}: {e}")
            # 返回一个黑色图像作为占位符
            placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
            return placeholder, label


class ImageFolderDataSource:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)  # Use pathlib.Path
        self.samples: list[tuple[str, int]] = []
        self.classes: list[str] = []
        self._load_samples()

    def _load_samples(self) -> None:
        if not self.root_dir.exists():
            msg = f"Root directory {self.root_dir} does not exist."
            raise FileNotFoundError(msg)

        class_to_idx = {}
        valid_extensions = get_image_extensions()

        class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        class_dirs.sort(key=lambda x: x.name)

        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = len(class_to_idx)
            class_to_idx[class_name] = class_idx
            self.classes.append(class_name)

            # Move image file collection inside the class_dir loop
            for ext in valid_extensions:
                for img_path in class_dir.glob(f"*{ext}"):
                    self.samples.append((str(img_path), class_idx))

        if not self.samples:
            msg = f"No valid images found in directory '{self.root_dir}'"
            raise RuntimeError(msg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.samples[index]


def create_datasets(params):
    """Create training and validation datasets"""
    train_dataset = ImageFolderDataSource(params["train_data_path"])

    val_dataset = ImageFolderDataSource(params["val_data_path"])

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, params):
    """Create data loaders"""

    def create_batch_fn(batch):
        images = jnp.stack([element[0] for element in batch], axis=0)
        labels = jnp.stack([element[1] for element in batch], axis=0)
        return images, labels

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
            batch_fn=create_batch_fn,
        )
    )
    val_loader = (
        grain.MapDataset.source(val_dataset)
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
            batch_fn=create_batch_fn,
        )
    )

    return train_loader, val_loader


def create_model(seed, num_classes):
    """Create a new model"""
    return resnet50(rngs=nnx.Rngs(seed), num_classes=num_classes)


def create_optimizer(model, learining_rate: float, weight_decay: float):
    """Create an optimizer for the model"""
    return nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=learining_rate,
            weight_decay=weight_decay,
        ),
    )


def save_model(model, path_str: str):
    """Save model state"""
    model_path = Path(path_str)
    model_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    checkpointer = ocp.PyTreeCheckpointer()
    state = nnx.state(model)
    checkpointer.save(model_path, state)


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

    optimizer = create_optimizer(
        model,
        learining_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    train_metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )
    val_metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    # Initialize tracking variables

    # Initialize best accuracy
    best_acc = -1.0

    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    # Training loop
    csv_path = os.path.join(params["checkpoint_dir"], "train_log.csv")  # noqa: PTH118
    with Path.open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
        )
    cached_train_step = nnx.cached_partial(train_step, model, optimizer, train_metrics)
    cached_eval_step = nnx.cached_partial(eval_step, model, val_metrics)
    print(f"\n🏃 Starting training for {params['num_epochs']} epochs...")

    for epoch in range(params["num_epochs"]):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{params['num_epochs']}")
        print(f"{'=' * 60}")

        # Train and validate
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            cached_train_step(batch)
        train_result = train_metrics.compute()
        train_metrics.reset()
        print(
            f"✅ Train Loss: {train_result['loss']:.6f}, Acc: {train_result['accuracy'] * 100:.6f}%"
        )

        model.eval()
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation"):
            cached_eval_step(batch)
        val_result = val_metrics.compute()
        val_metrics.reset()
        print(
            f"📊 Val Loss: {val_result['loss']:.6f}, Acc: {val_result['accuracy'] * 100:.6f}%"
        )

        for k, v in train_result.items():
            metrics_history[f"train_{k}"].append(v)
        for k, v in val_result.items():
            metrics_history[f"val_{k}"].append(v)
        # Save model if validation accuracy improved

        current_acc = float(val_result["accuracy"])
        if current_acc > best_acc:
            best_acc = current_acc
            checkpoint_path = os.path.join(  # noqa: PTH118
                params["checkpoint_dir"],
                f"best_model_Epoch_{epoch + 1}_Acc_{current_acc:.6f}",
                "state",
            )
            save_model(model, checkpoint_path)
            print(f"🎉 New best model saved with accuracy: {current_acc * 100:.6f}%")
        with open(csv_path, mode="a", newline="") as f:  # noqa: PTH123
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    train_result["loss"],
                    train_result["accuracy"],
                    val_result["loss"],
                    val_result["accuracy"],
                ]
            )
    print("\n🎯 Training completed!")
    print(f"🏆 Best validation accuracy: {best_acc * 100:.6f}%")


if __name__ == "__main__":
    main()
