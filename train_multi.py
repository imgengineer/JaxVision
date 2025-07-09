import os

os.environ["PJRT_DEVICE"] = "TPU"
import csv
from pathlib import Path

import albumentations as A
import grain.python as grain
import jax
import optax
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from dataset import create_datasets
from jaxvision.transforms import AlbumentationsTransform, OpenCVLoadImageMap
from jaxvision.utils import print_dataset_info, set_seed
from resnet import resnet50

params = {
    "num_epochs": 300,
    "batch_size": 720,
    "target_size": 224,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "seed": 42,
    "num_classes": 6,
    "train_data_path": Path(
        "/root/JaxVision/Original Images/Original Images/FOLDS/fold1/Train",
    ),
    "val_data_path": Path(
        "/root/JaxVision/Original Images/Original Images/FOLDS/fold1/Test",
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
        shard_options=grain.NoSharding(),
        num_epochs=200,
    )

    val_sampler = grain.IndexSampler(
        len(val_dataset),
        shuffle=False,
        seed=params["seed"],
        shard_options=grain.NoSharding(),
        num_epochs=1,
    )

    train_loader = grain.DataLoader(
        data_source=train_dataset,
        sampler=train_sampler,
        worker_count=64,
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
        worker_count=64,
        worker_buffer_size=4,
        operations=[
            OpenCVLoadImageMap(),
            AlbumentationsTransform(
                create_transforms(
                    target_size=params["target_size"],
                    is_training=False,
                ),
            ),
            grain.Batch(
                params["batch_size"],
                drop_remainder=False,
            ),
        ],
    )

    return train_loader, val_loader


def create_optimizer(model, learining_rate: float, weight_decay: float) -> nnx.Optimizer:
    return nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=learining_rate,
            weight_decay=weight_decay,
        ),
    )


def loss_fn(model, batch):
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

    mesh = Mesh(mesh_utils.create_device_mesh((2, 2)), ("batch", "model"))
    model = resnet50(num_classes=params["num_classes"], rngs=nnx.Rngs(params["seed"]))

    for _, m in model.iter_modules():
        if isinstance(m, nnx.Conv):
            m.kernel_init = nnx.with_partitioning(
                nnx.initializers.variance_scaling(2.0, "fan_out", "truncated_normal"),
                NamedSharding(
                    mesh,
                    P(None, "model"),
                ),
            )
            m.bias_init = nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P("model")))
        elif isinstance(m, nnx.BatchNorm | nnx.GroupNorm):
            m.scale_init = nnx.with_partitioning(nnx.initializers.constant(1), NamedSharding(mesh, P("model")))
            m.bias_init = nnx.with_partitioning(nnx.initializers.constant(0), NamedSharding(mesh, P("model")))
        elif isinstance(m, nnx.Linear):
            m.kernel_init = nnx.with_partitioning(
                nnx.initializers.xavier_uniform(),
                NamedSharding(mesh, P(None, "model")),
            )
            m.bias_init = nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P("model")))

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

    csv_path = params["checkpoint_dir"] / "train_log.csv"

    with Path.open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])

    print(f"\n🏃 Starting training for {params['num_epochs']} epochs...")

    cached_train_step = nnx.cached_partial(train_step, model, optimizer, train_metrics)
    nnx.cached_partial(eval_step, model, val_metrics)
    for epoch in range(params["num_epochs"]):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{params['num_epochs']}")
        print(f"{'=' * 60}")

        model.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            images, labels = batch
            sharded_images = jax.device_put(images, NamedSharding(mesh, P("batch", None)))
            sharded_labels = jax.device_put(labels, NamedSharding(mesh, P("batch")))
            cached_train_step(
                (sharded_images, sharded_labels),
            )
        train_result = train_metrics.compute()
        train_metrics.reset()
        print(
            f"✅ Train Loss: {train_result['loss']:.6f}, Acc: {train_result['accuracy'] * 100:.6f}%",
        )


if __name__ == "__main__":
    main()
