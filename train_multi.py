import os
from functools import partial

from jaxvision.models.resnet import resnet50

os.environ["PJRT_DEVICE"] = "TPU"
import csv
from pathlib import Path

import albumentations as A  # noqa: N812
import grain.python as grain
import jax
import optax
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from data import create_datasets
from jaxvision.transforms import AlbumentationsTransform, OpenCVLoadImageMap
from jaxvision.utils import print_dataset_info, save_model, set_seed
from params import Config

params = Config()


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
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, p=0.5),
                A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.5),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
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
        seed=params.seed,
        shard_options=grain.ShardByJaxProcess(),
        num_epochs=200,
    )

    val_sampler = grain.IndexSampler(
        len(val_dataset),
        shuffle=False,
        seed=params.seed,
        shard_options=grain.ShardByJaxProcess(),
        num_epochs=200,
    )

    train_loader = grain.DataLoader(
        data_source=train_dataset,
        sampler=train_sampler,
        worker_count=64,
        worker_buffer_size=2,
        operations=[
            OpenCVLoadImageMap(),
            AlbumentationsTransform(
                create_transforms(
                    target_size=params.target_size,
                    is_training=True,
                ),
            ),
            grain.Batch(
                params.batch_size,
                drop_remainder=True,
            ),
        ],
    )

    val_loader = grain.DataLoader(
        data_source=val_dataset,
        sampler=val_sampler,
        worker_count=64,
        worker_buffer_size=2,
        operations=[
            OpenCVLoadImageMap(),
            AlbumentationsTransform(
                create_transforms(
                    target_size=params.target_size,
                    is_training=False,
                ),
            ),
            grain.Batch(
                params.batch_size,
                drop_remainder=False,
            ),
        ],
    )

    return train_loader, val_loader


def create_optimizer(model, learning_rate: float, weight_decay: float) -> nnx.Optimizer:
    tx = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return nnx.Optimizer(
        model,
        tx,
    )


def loss_fn(model, batch):
    logits = model(batch[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch[1]).mean()
    return loss, logits


@partial(jax.jit, donate_argnames="state")
def train_step_jax(graphdef, state, batch):
    model, optimizer = nnx.merge(graphdef, state)
    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
    optimizer.update(grads)
    state = nnx.state((model, optimizer))
    return loss, logits, state


# @partial(jax.jit, donate_argnames="state")
@jax.jit
def eval_step_jax(graphdef, state, batch):
    model, optimizer = nnx.merge(graphdef, state)
    loss, logits = loss_fn(model, batch)
    # state = nnx.state((model, optimizer))
    return loss, logits


def main():  # noqa: PLR0915
    """Main function to orchestrate the training and validation process of the model."""
    set_seed(params.seed)

    print("\n📂 Loading datasets...")

    train_dataset, val_dataset = create_datasets(params)

    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, params)

    print_dataset_info(train_dataset, val_dataset)

    print("\n🏗️ Creating model and optimizer...")

    # Data Parallel
    mesh = Mesh(mesh_utils.create_device_mesh((4,)), ("data",))
    data_sharding = NamedSharding(mesh, P("data"))
    model_sharding = NamedSharding(mesh, P())

    model = resnet50(num_classes=params.num_classes, rngs=nnx.Rngs(params.seed))
    optimizer = create_optimizer(
        model,
        learning_rate=params.learning_rate,
        weight_decay=params.weight_decay,
    )

    # replicate state
    state = nnx.state((model, optimizer))
    state = jax.device_put(state, model_sharding)
    nnx.update((model, optimizer), state)

    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )
    best_acc = -1.0

    csv_path = params.checkpoint_dir / "train_log.csv"

    with Path.open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])

    print(f"\n🏃 Starting training for {params.num_epochs} epochs...")

    graphdef, state = nnx.split((model, optimizer))
    for epoch in range(params.num_epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1} / {params.num_epochs}")
        print(f"{'=' * 60}")
        model.train()
        state = nnx.state((model, optimizer))
        for step, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1} Training"):
            loss, logits, state = train_step_jax(graphdef, state, jax.device_put(batch, data_sharding))
            metrics.update(
                loss=loss,
                logits=logits,
                labels=batch[1],
            )
            print(
                f"\r[train] epoch: {epoch + 1}/{params.num_epochs}, loss: {loss.item():.4f} ",
                end="",
            )
            print("\r", end="")
        train_result = metrics.compute()
        metrics.reset()
        print(
            f"\n✅ Train Loss: {train_result['loss']:.6f}, Acc: {train_result['accuracy'] * 100:.6f}%",
        )
        model, optimizer = nnx.merge(graphdef, state)
        model.eval()
        state = nnx.state((model, optimizer))
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation"):
            loss, logits = eval_step_jax(graphdef, state, batch)
            metrics.update(
                loss=loss,
                logits=logits,
                labels=batch[1],
            )
        val_result = metrics.compute()
        metrics.reset()
        print(
            f"📊 Val Loss: {val_result['loss']:.6f}, Acc: {val_result['accuracy'] * 100:.6f}%",
        )

        current_acc = float(val_result["accuracy"])
        if current_acc > best_acc:
            best_acc = current_acc

            checkpoint_path = params.checkpoint_dir / f"best_model_Epoch_{epoch + 1}_Acc_{current_acc:.6f}" / "state"
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
