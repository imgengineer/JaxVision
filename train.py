import os  # Import the os module for interacting with the operating system

# Set XLA flags for performance optimization on GPU.
# XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra
# that optimizes TensorFlow computations.
os.environ["XLA_FLAGS"] = (
    # Links to documentation for XLA performance flags.
    # https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/GPU_performance.md
    # https://docs.jax.dev/en/latest/gpu_performance_tips.html#xla-performance-flags
    # Enable Triton-based matrix multiplication. Triton is a language and compiler
    # for writing highly efficient custom GPU kernels. This can improve the
    # performance of matrix multiplications (GEMM).
    "--xla_gpu_triton_gemm_any=True "
    # Enable latency hiding optimizations. This flag allows XLA to overlap
    # computation with data transfers, potentially reducing idle time.
    "--xla_gpu_enable_latency_hiding_scheduler=true "
)
# Set the fraction of GPU memory that JAX should pre-allocate.
# "0.95" means 95% of the available GPU memory will be pre-allocated for JAX operations.
# This can prevent out-of-memory errors and improve performance by reducing
# memory allocation overheads during runtime.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
import csv  # Import the csv module for reading and writing CSV files
from pathlib import Path  # # Import Path from pathlib for object-oriented filesystem paths

import albumentations as A  # Import Albumentations for image augmentations.  # noqa: N812
import grain.python as grain  # Import Grain for building data pipelines
import jax
import optax  # Import Optax for gradient processing and optimization
from flax import nnx  # Import Flax NNX for neural network modules and utilities
from tqdm import tqdm  # Import tqdm for displaying progress bars

# Import custom modules from the current project
from dataset import create_datasets  # Function to create training and validation datasets
from jaxvision.models.resnet import resnet18  # ResNet18 model from JAXVision library
from jaxvision.transforms import AlbumentationsTransform, OpenCVLoadImageMap  # Custom transforms
from jaxvision.utils import create_model, print_dataset_info, save_model, set_seed  # Utility functions

# Configuration parameters for the training process
params = {
    "num_epochs": 300,  # Total number of training epochs
    "batch_size": 64,  # Number of samples per batch
    "target_size": 224,  # Target size (height and width) for image resizing
    "learning_rate": 5e-4,  # Learning rate for the optimizer
    "weight_decay": 1e-4,  # Weight decay (L2 regularization) to prevent overfitting
    "seed": 42,  # Random seed for reproducibility
    "num_workers": 8,  # Number of worker processes for data loading
    "num_classes": 16,  # Number of output classes for classification
    "train_data_path": Path("./MpoxData/train"),  # Path to the training data directory
    "val_data_path": Path("./MpoxData/validation"),  # Path to the validation data directory
    "checkpoint_dir": Path(
        "/Users/billy/Documents/DLStudy/JaxVision/checkpoints",
    ),  # Directory to save model checkpoints
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
        # Resize all images to the target_size. This is a mandatory transformation.
        A.Resize(height=target_size, width=target_size, p=1.0),
    ]

    if is_training:
        # Add training-specific augmentations if `is_training` is True.
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),  # Randomly flip images horizontally with 50% probability
                A.VerticalFlip(p=0.5),  # Randomly flip images vertically with 50% probability
                A.Rotate(limit=30, p=0.5),  # Randomly rotate images by up to 30 degrees with 50% probability
                A.ColorJitter(  # Randomly change the brightness, contrast, saturation, and hue
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                    p=0.5,
                ),
                A.RandomResizedCrop(  # Crop a random part of the image and resize it
                    size=(target_size, target_size),
                    scale=(0.8, 1.0),  # Scale factor of the cropped area
                    ratio=(0.75, 1.33),  # Aspect ratio of the cropped area
                    p=0.5,
                ),
            ],
        )

    # Always apply normalization as the last step.
    transforms_list.append(
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean values for R, G, B channels
            std=[0.229, 0.224, 0.225],  # ImageNet standard deviation values for R, G, B channels
            max_pixel_value=255.0,  # Maximum pixel value in the input images (e.g., 255 for 8-bit images)
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
    # Create an `grain.IndexSampler` for the training data.
    # This sampler determines the order in which samples are drawn from the dataset.
    train_sampler = grain.IndexSampler(
        len(train_dataset),  # The total number of samples in the data source.
        shuffle=True,  # Shuffle the data to randomize the order of samples for training.
        seed=params["seed"],  # Set a seed for reproducibility of shuffling.
        shard_options=grain.NoSharding(),  # No sharding is applied since this is a single-device setup.
        num_epochs=1,  # Iterate over the dataset for one epoch per call to the data loader.
    )
    # Create an `grain.IndexSampler` for the validation data.
    val_sampler = grain.IndexSampler(
        len(val_dataset),
        shuffle=False,  # Do not shuffle validation data to ensure consistent evaluation.
        seed=params["seed"],
        shard_options=grain.NoSharding(),
        num_epochs=1,
    )

    # Create the training DataLoader.
    train_loader = grain.DataLoader(
        data_source=train_dataset,  # The dataset to load data from.
        sampler=train_sampler,  # The sampler to determine how to access the data.
        worker_count=params["num_workers"],  # Number of child processes launched to parallelize data transformations.
        worker_buffer_size=2,  # Count of output batches to produce in advance per worker to reduce pipeline stalls.
        operations=[
            # Custom operation to load images using OpenCV.
            OpenCVLoadImageMap(),
            # Apply Albumentations transformations.
            AlbumentationsTransform(
                # For training data, apply training-specific augmentations.
                create_transforms(
                    target_size=params["target_size"],
                    is_training=True,  # IMPORTANT: Use is_training=True for the training loader
                ),
            ),
            # Batch the samples together.
            grain.Batch(
                params["batch_size"],  # The desired batch size.
                drop_remainder=True,  # Drop the last batch if its size is less than `batch_size`.
            ),
        ],
    )
    # Create the validation DataLoader.
    val_loader = grain.DataLoader(
        data_source=val_dataset,
        sampler=val_sampler,
        worker_count=params["num_workers"],
        worker_buffer_size=2,
        operations=[
            OpenCVLoadImageMap(),
            AlbumentationsTransform(
                # For validation data, apply only resizing and normalization (no augmentations).
                create_transforms(
                    target_size=params["target_size"],
                    is_training=False,  # IMPORTANT: Use is_training=False for the validation loader
                ),
            ),
            grain.Batch(params["batch_size"]),  # Batch the samples.
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
        model,  # The model whose parameters will be optimized
        optax.adamw(  # Using the AdamW optimizer, which combines Adam with weight decay
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
    images, labels = batch  # Unpack images and labels from the batch
    logits = model(images)  # Get raw predictions (logits) from the model
    # Calculate softmax cross-entropy loss with integer labels.
    # The .mean() calculates the average loss over the batch.
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
    return loss, logits  # Return both loss and logits for potential use in metrics


@nnx.jit  # JIT compile this function for performance.
def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Performs a single training step, including forward pass, loss calculation,
    gradient computation, and parameter update.

    Args:
        model: The neural network model.
        optimizer: The Optax optimizer.
        metrics: The metrics tracker for training.
        batch: A tuple containing (images, labels) for the current training batch.

    """
    # Define a function that computes the loss and its gradient with respect to model parameters.
    # `has_aux=True` indicates that `loss_fn` returns auxiliary data (logits in this case)
    # along with the loss, which should not be differentiated.
    # Compute the loss, logits, and gradients.
    batch = jax.device_put(batch)
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    # Update the training metrics with the current batch's loss, logits, and labels.
    metrics.update(loss=loss, logits=logits, labels=batch[1])
    # Apply the computed gradients to update the model's parameters using the optimizer.
    optimizer.update(grads)


@nnx.jit  # JIT compile this function for performance.
def eval_step(model, metrics: nnx.MultiMetric, batch):
    """Performs a single evaluation step, including forward pass and loss calculation.
    No gradient computation or parameter updates occur here.

    Args:
        model: The neural network model.
        metrics: The metrics tracker for evaluation.
        batch: A tuple containing (images, labels) for the current evaluation batch.

    """
    batch = jax.device_put(batch)
    # Calculate the loss and logits for the current batch.
    loss, logits = loss_fn(model, batch)
    # Update the evaluation metrics with the current batch's loss, logits, and labels.
    metrics.update(loss=loss, logits=logits, labels=batch[1])


def main():
    """Main function to orchestrate the training and validation process of the model."""
    set_seed(params["seed"])  # Set the random seed for reproducibility

    print("🚀 Starting training with ResNet18...")
    print(f"📋 Configuration: {params}")

    # Create datasets and dataloaders
    print("\n📂 Loading datasets...")
    # Call the custom function to create training and validation datasets
    train_dataset, val_dataset = create_datasets(params)
    # Create data loaders from the datasets
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, params)

    print_dataset_info(train_dataset, val_dataset)  # Print information about the datasets

    # Create model and optimizer
    print("\n🏗️ Creating model and optimizer...")
    # Create the ResNet18 model with specified seed and number of classes
    model = create_model(resnet18, params["seed"], params["num_classes"])

    # Create the optimizer
    optimizer = create_optimizer(
        model,
        learining_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    # Initialize metrics for training and validation.
    # nnx.MultiMetric allows tracking multiple metrics simultaneously.
    train_metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),  # Track accuracy for training
        loss=nnx.metrics.Average("loss"),  # Track average loss for training
    )
    val_metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),  # Track accuracy for validation
        loss=nnx.metrics.Average("loss"),  # Track average loss for validation
    )

    # Initialize best accuracy for saving the best model
    best_acc = -1.0

    # Training loop setup
    # Define the path for the training log CSV file
    csv_path = params["checkpoint_dir"] / "train_log.csv"
    # Open the CSV file in write mode and write the header row
    with Path.open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])

    # Create cached partial functions for `train_step` and `eval_step`.
    # `nnx.cached_partial` can optimize JIT compilation for functions with fixed arguments.
    cached_train_step = nnx.cached_partial(train_step, model, optimizer, train_metrics)
    cached_eval_step = nnx.cached_partial(eval_step, model, val_metrics)

    print(f"\n🏃 Starting training for {params['num_epochs']} epochs...")

    # Main training loop
    for epoch in range(params["num_epochs"]):
        print(f"\n{'=' * 60}")  # Print a separator line
        print(f"Epoch {epoch + 1}/{params['num_epochs']}")  # Print current epoch number
        print(f"{'=' * 60}")  # Print another separator line

        # --- Training Phase ---
        model.train()  # Set the model to training mode (e.g., enable dropout if present)
        # Iterate over the training data loader with a progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            cached_train_step(batch)  # Perform a single training step
        train_result = train_metrics.compute()  # Compute final training metrics for the epoch
        train_metrics.reset()  # Reset training metrics for the next epoch
        print(
            f"✅ Train Loss: {train_result['loss']:.6f}, Acc: {train_result['accuracy'] * 100:.6f}%",
        )  # Print training results

        # --- Validation Phase ---
        model.eval()  # Set the model to evaluation mode (e.g., disable dropout)
        # Iterate over the validation data loader with a progress bar
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation"):
            cached_eval_step(batch)  # Perform a single evaluation step
        val_result = val_metrics.compute()  # Compute final validation metrics for the epoch
        val_metrics.reset()  # Reset validation metrics for the next epoch
        print(
            f"📊 Val Loss: {val_result['loss']:.6f}, Acc: {val_result['accuracy'] * 100:.6f}%",
        )  # Print validation results

        # Save model if validation accuracy improved
        current_acc = float(val_result["accuracy"])  # Get current validation accuracy
        if current_acc > best_acc:
            best_acc = current_acc  # Update best accuracy
            # Define checkpoint path including epoch and accuracy
            checkpoint_path = params["checkpoint_dir"] / f"best_model_Epoch_{epoch + 1}_Acc_{current_acc:.6f}" / "state"
            save_model(model, checkpoint_path)  # Save the model state
            print(f"🎉 New best model saved with accuracy: {current_acc * 100:.6f}%")

        # Log results to CSV
        with csv_path.open(mode="a", newline="") as f:  # Open CSV in append mode
            writer = csv.writer(f)
            writer.writerow(  # Write epoch results
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
    main()  # Run the main training function when the script is executed
