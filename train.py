import os
import random

import albumentations as A  # noqa: N812
import cv2
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
import torch
import torchvision
from flax import nnx
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.resnet import resnet18

# Configuration
params = {
    "num_epochs": 5,
    "batch_size": 64,
    "target_size": 224,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "seed": 42,
    "num_workers": 2,
    "num_classes": 6,
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


def numpy_collate(batch):
    """
    Collates a batch of samples into a single array or nested list of arrays.

    This function recursively processes a batch of samples, stacking NumPy arrays, and collating lists or tuples by grouping elements together. If the batch consists of NumPy arrays, they are stacked. If the batch contains tuples or lists, the function recursively applies the collation.

    This collate function is taken from the `JAX tutorial with PyTorch Data Loading <https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html>`_.

    Parameters
    ----------
    batch : List[Union[np.ndarray, Tuple, List]]
        A batch of samples where each sample is either a NumPy array, a tuple, or a list. It depends on the
        data loader.

    Returns
    -------
    np.ndarray
        The collated batch, either as a stacked NumPy array or as a nested structure of arrays.

    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    if isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch, strict=False)
        return [numpy_collate(samples) for samples in transposed]
    return np.asarray(batch)


def load_image(img_path):
    """Load image in RGB format"""
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def create_transforms(target_size, is_training=True):  # noqa: FBT002
    """Create data augmentation transforms"""
    transforms_list = [
        # Fix resize issue - use direct resize instead of SmallestMaxSize + Crop
        A.Resize(height=target_size, width=target_size, p=1.0),
    ]

    if is_training:
        transforms_list.extend(
            [
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
            ]
        )

    transforms_list.append(
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        )
    )

    return A.Compose(transforms_list)


class ImageTransforms:
    """Wrapper for albumentations transforms"""

    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.transforms(image=image)["image"]


def create_datasets(params):
    """Create training and validation datasets"""
    train_transforms = create_transforms(params["target_size"], is_training=True)
    val_transforms = create_transforms(params["target_size"], is_training=False)

    train_dataset = torchvision.datasets.ImageFolder(
        params["train_data_path"],
        transform=ImageTransforms(train_transforms),
        loader=load_image,
    )

    val_dataset = torchvision.datasets.ImageFolder(
        params["val_data_path"],
        transform=ImageTransforms(val_transforms),
        loader=load_image,
    )

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, params):
    """Create data loaders"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=numpy_collate,
        num_workers=params["num_workers"],
        pin_memory=True,
        drop_last=True,  # Avoid batch size issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=numpy_collate,
        num_workers=params["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader


def create_model(seed, num_classes):
    """Create a new model"""
    return resnet18(rngs=nnx.Rngs(seed), num_classes=num_classes)


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
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
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
    # Convert numpy arrays to jnp.array on GPU
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
    ax2.plot(epochs, metrics_history["train_accuracy"], label="Train Accuracy", marker="o")
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
    print(f"  Number of classes:ß {len(train_dataset.classes)}")


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
        best_acc = save_best_model_if_improved(model, val_result, best_acc, epoch, params["checkpoint_dir"])

    print("\n🎯 Training completed!")
    print(f"🏆 Best validation accuracy: {best_acc * 100:.6f}%")

    # Plot results
    print("\n📈 Plotting training metrics...")
    plot_training_metrics(metrics_history)


if __name__ == "__main__":
    main()
