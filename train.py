import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
import csv
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

from dataset import ImageFolderDataSource
from jaxvision.models.resnet import resnet18
from jaxvision.transforms import AlbumentationsTransform, LoadImageMap

# Configuration
params = {
    "num_epochs": 300,
    "batch_size": 64,
    "target_size": 224,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "seed": 42,
    "num_workers": 8,
    "num_classes": 16,
    "train_data_path": Path("./MpoxData/train"),
    "val_data_path": Path("./MpoxData/validation"),
    "checkpoint_dir": Path("/Users/billy/Documents/DLStudy/JaxVision/checkpoints"),
}


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.default_rng(seed)
    random.seed(seed)


def create_transforms(target_size, *, is_training=True) -> A.Compose:
    transforms_list = [
        A.Resize(height=target_size, width=target_size, p=1.0),
    ]

    if is_training:
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_REFLECT_101),  # 使用 cv2.BORDER_REFLECT_101 填充边缘
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,  # 减小 hue 的变化范围
                    p=0.5,
                ),
                A.RandomResizedCrop(
                    size=(target_size, target_size),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.33),
                    p=0.5,
                ),
            ],
        )

    transforms_list.append(
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet 均值
            std=[0.229, 0.224, 0.225],  # ImageNet 标准差
            max_pixel_value=255.0,
        ),
    )

    return A.Compose(transforms_list)


def create_datasets(params):
    """Create training and validation datasets."""
    train_dataset = ImageFolderDataSource(params["train_data_path"])

    val_dataset = ImageFolderDataSource(params["val_data_path"])

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, params):
    """Create data loaders."""

    def create_batch_fn(batch):
        images, labels = zip(*batch, strict=True)
        return np.stack(images, axis=0), np.stack(labels, axis=0)

    train_loader = (
        grain.MapDataset.source(train_dataset)
        .shuffle(seed=params["seed"])
        .map(LoadImageMap())
        .map(AlbumentationsTransform(create_transforms(params["target_size"], is_training=True)))
        .to_iter_dataset()
        .batch(
            batch_size=params["batch_size"],
            drop_remainder=True,
            batch_fn=create_batch_fn,
        )
        .mp_prefetch(options=grain.multiprocessing.MultiprocessingOptions(num_workers=params["num_workers"]))
    )
    val_loader = (
        grain.MapDataset.source(val_dataset)
        .map(LoadImageMap())
        .map(AlbumentationsTransform(create_transforms(params["target_size"], is_training=False)))
        .to_iter_dataset()
        .batch(
            batch_size=params["batch_size"],
            drop_remainder=False,
            batch_fn=create_batch_fn,
        )
        .mp_prefetch(options=grain.multiprocessing.MultiprocessingOptions(num_workers=params["num_workers"]))
    )

    return train_loader, val_loader


def create_model(seed, num_classes):
    """Create a new model."""
    return resnet18(rngs=nnx.Rngs(seed), num_classes=num_classes)


def create_optimizer(model, learining_rate: float, weight_decay: float):
    """Create an optimizer for the model."""
    return nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=learining_rate,
            weight_decay=weight_decay,
        ),
    )


def save_model(model, path_str: str):
    """Save model state."""
    model_path = Path(path_str)
    model_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    checkpointer = ocp.PyTreeCheckpointer()
    state = nnx.state(model)
    checkpointer.save(model_path, state)


def load_model(path, num_classes):
    """Load model from checkpoint."""
    model = nnx.eval_shape(lambda: create_model(0, num_classes))
    state = nnx.state(model)

    checkpointer = ocp.PyTreeCheckpointer()
    state = checkpointer.restore(path, item=state)

    nnx.update(model, state)
    return model


def loss_fn(model, batch):
    """Calculate loss and logits."""
    images, labels = batch
    logits = model(images)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
    return loss, logits


@nnx.jit
def train_step(model, optimizer, metrics, batch):
    """Single training step."""
    # Convert numpy arrays to jnp.array on GPU
    x, y_true = jnp.asarray(batch[0]), jnp.asarray(batch[1])
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, (x, y_true))
    metrics.update(loss=loss, logits=logits, labels=y_true)
    optimizer.update(grads)


@nnx.jit
def eval_step(model, metrics, batch):
    """Single evaluation step."""
    # convert numpy arrays to jnp.array on GPU
    x, y_true = jnp.asarray(batch[0]), jnp.asarray(batch[1])
    loss, logits = loss_fn(model, (x, y_true))
    metrics.update(loss=loss, logits=logits, labels=y_true)


def print_dataset_info(train_dataset, val_dataset):
    """Print dataset information."""
    print("📊 Dataset Info:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Classes: {train_dataset.classes}")
    print(f"  Number of classes: {len(train_dataset.classes)}")


def main():
    """Main training function."""
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

    # Training loop
    csv_path = params["checkpoint_dir"] / "train_log.csv"
    with Path.open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
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
        print(f"✅ Train Loss: {train_result['loss']:.6f}, Acc: {train_result['accuracy'] * 100:.6f}%")

        model.eval()
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation"):
            cached_eval_step(batch)
        val_result = val_metrics.compute()
        val_metrics.reset()
        print(f"📊 Val Loss: {val_result['loss']:.6f}, Acc: {val_result['accuracy'] * 100:.6f}%")

        # Save model if validation accuracy improved
        current_acc = float(val_result["accuracy"])
        if current_acc > best_acc:
            best_acc = current_acc
            checkpoint_path = params["checkpoint_dir"] / f"best_model_Epoch_{epoch + 1}_Acc_{current_acc:.6f}" / "state"
            save_model(model, checkpoint_path)
            print(f"🎉 New best model saved with accuracy: {current_acc * 100:.6f}%")
        with Path.open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    train_result["loss"],
                    train_result["accuracy"],
                    val_result["loss"],
                    val_result["accuracy"],
                ],
            )
    print("\n🎯 Training completed!")
    print(f"🏆 Best validation accuracy: {best_acc * 100:.6f}%")


if __name__ == "__main__":
    main()
