import csv
from pathlib import Path

import albumentations as A
import grain.python as grain
import optax
from flax import nnx
from tqdm import tqdm

from dataset import create_datasets
from jaxvision.models.vision_transformer import vit_b_16
from jaxvision.transforms import AlbumentationsTransform, OpenCVLoadImageMap
from jaxvision.utils import create_model, print_dataset_info, save_model, set_seed

params = {
    "num_epochs": 300,
    "batch_size": 64,
    "target_size": 224,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "seed": 42,
    "num_workers": 64,
    "num_classes": 6,
    "train_data_path": Path(
        "/root/JaxVision/Original Images/Original Images/FOLDS/fold1/Train",
    ),
    "val_data_path": Path(
        "/root/JaxVision/Original Images/Original Images/FOLDS/fold1/Valid",
    ),
    "checkpoint_dir": Path(
        "/root/JaxVision/checkpoints",
    ),
}


def create_transforms(target_size, *, is_training=True) -> A.Compose:
    """Create image augmentation and normalization transformations.

    Args:
        target_size: The desired height and width for the images.
        is_training: A boolean indicating whether to apply training-specific augmentations.

    Returns:
        An `A.Compose` object containing the sequence of transformations.

    """
    transforms_list = [
        A.Resize(height=target_size, width=target_size, p=1.0),
    ]

    if is_training:
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
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
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
    )

    return A.Compose(transforms_list)


def create_dataloaders(train_dataset, val_dataset, params):
    """Create data loaders for training and validation datasets using Grain.

    Args:
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        params: A dictionary containing configuration parameters like seed, batch_size, etc.

    Returns:
        A tuple containing the training and validation Grain DataLoaders.

    """
    train_sampler = grain.IndexSampler(
        len(train_dataset),
        shuffle=True,
        seed=params["seed"],
        shard_options=grain.ShardByJaxProcess(),
        num_epochs=8,
    )

    val_sampler = grain.IndexSampler(
        len(val_dataset),
        shuffle=False,
        seed=params["seed"],
        shard_options=grain.ShardByJaxProcess(),
        num_epochs=1,
    )

    train_loader = grain.DataLoader(
        data_source=train_dataset,
        sampler=train_sampler,
        worker_count=params["num_workers"],
        worker_buffer_size=4,
        operations=[
            OpenCVLoadImageMap(),
            AlbumentationsTransform(
                create_transforms(
                    target_size=params["target_size"],
                    is_training=True,
                ),
            ),
            grain.Batch(
                params["batch_size"],
                drop_remainder=True,
            ),
        ],
    )

    val_loader = grain.DataLoader(
        data_source=val_dataset,
        sampler=val_sampler,
        worker_count=params["num_workers"],
        worker_buffer_size=4,
        operations=[
            OpenCVLoadImageMap(),
            AlbumentationsTransform(
                create_transforms(
                    target_size=params["target_size"],
                    is_training=False,
                ),
            ),
            grain.Batch(params["batch_size"]),
        ],
    )

    return train_loader, val_loader


def create_optimizer(model, learining_rate: float, weight_decay: float) -> nnx.Optimizer:
    """Create an Optax optimizer for the given model.

    Args:
        model: The model to optimize.
        learining_rate: The learning rate for the optimizer.
        weight_decay: The weight decay (L2 regularization) strength.

    Returns:
        An `nnx.Optimizer` instance.

    """
    return nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=learining_rate,
            weight_decay=weight_decay,
        ),
    )


def loss_fn(model, batch):
    """Calculate the loss and model logits for a given batch.

    Args:
        model: The neural network model.
        batch: A tuple containing (images, labels).

    Returns:
        A tuple containing the calculated loss and the model's logits.

    """
    images, labels = batch
    logits = model(images)

    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
    return loss, logits


@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Performs a single training step, including forward pass, loss calculation,
    gradient computation, and parameter update.

    Args:
        model: The neural network model.
        optimizer: The Optax optimizer.
        metrics: The metrics tracker for training.
        batch: A tuple containing (images, labels) for the current training batch.

    """
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)

    metrics.update(loss=loss, logits=logits, labels=batch[1])

    optimizer.update(grads)


@nnx.jit
def eval_step(model, metrics: nnx.MultiMetric, batch):
    """Performs a single evaluation step, including forward pass and loss calculation.
    No gradient computation or parameter updates occur here.

    Args:
        model: The neural network model.
        metrics: The metrics tracker for evaluation.
        batch: A tuple containing (images, labels) for the current evaluation batch.

    """
    loss, logits = loss_fn(model, batch)

    metrics.update(loss=loss, logits=logits, labels=batch[1])


def main():
    """Main function to orchestrate the training and validation process of the model."""
    set_seed(params["seed"])

    print(f"📋 Configuration: {params}")

    print("\n📂 Loading datasets...")

    train_dataset, val_dataset = create_datasets(params)

    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, params)

    print_dataset_info(train_dataset, val_dataset)

    print("\n🏗️ Creating model and optimizer...")

    model = create_model(vit_b_16, params["seed"], params["num_classes"])

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

    best_acc = -1.0

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

        model.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            cached_train_step(batch)
        train_result = train_metrics.compute()
        train_metrics.reset()
        print(
            f"✅ Train Loss: {train_result['loss']:.6f}, Acc: {train_result['accuracy'] * 100:.6f}%",
        )

        model.eval()

        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation"):
            cached_eval_step(batch)
        val_result = val_metrics.compute()
        val_metrics.reset()
        print(
            f"📊 Val Loss: {val_result['loss']:.6f}, Acc: {val_result['accuracy'] * 100:.6f}%",
        )

        current_acc = float(val_result["accuracy"])
        if current_acc > best_acc:
            best_acc = current_acc

            checkpoint_path = params["checkpoint_dir"] / f"best_model_Epoch_{epoch + 1}_Acc_{current_acc:.6f}" / "state"
            save_model(model, checkpoint_path)
            print(f"🎉 New best model saved with accuracy: {current_acc * 100:.6f}%")

        with csv_path.open(mode="a", newline="") as f:
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
