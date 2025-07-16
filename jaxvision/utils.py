import random
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp
from flax import nnx


def create_model(model, seed, num_classes) -> nnx.Module:
    """Create a new model instance.

    This function initializes a neural network model with a given random seed
    and the number of output classes. It's designed to work with Flax NNX models.

    Args:
        model: The model class or function to instantiate (e.g., `resnet18`).
               It's expected to accept `rngs` for random number generation
               and `num_classes` for the output layer.
        seed: An integer seed for random number generation within the model
              (e.g., for weight initialization).
        num_classes: The number of output classes for the model's final layer.

    Returns:
        A new instance of the `nnx.Module` (the initialized neural network model).

    """
    return model(rngs=nnx.Rngs(seed), num_classes=num_classes)


def set_seed(seed):
    """Set random seeds for various random number generators to ensure reproducibility.

    This is crucial for research and debugging, allowing experiments to be
    replicated with the same results.

    Args:
        seed: An integer seed value.

    """
    np.random.default_rng(seed)

    random.seed(seed)


def load_model(path, num_classes):
    """Load a model from a previously saved checkpoint.

    This function reconstructs a model's structure and then loads its saved
    parameters from a specified path using Orbax Checkpoint.

    Args:
        path: The file path to the model checkpoint.
        num_classes: The number of classes the model was trained with.
                     This is necessary to correctly reconstruct the model's shape.

    Returns:
        The loaded `nnx.Module` with its parameters restored from the checkpoint.

    """
    model = nnx.eval_shape(lambda: create_model(0, num_classes))

    state = nnx.state(model)

    checkpointer = ocp.PyTreeCheckpointer()

    state = checkpointer.restore(path, item=state)

    nnx.update(model, state)
    return model


def save_model(model, path_str: str):
    """Save the current state (parameters and variables) of a model to a checkpoint.

    This function creates the necessary directory structure and then uses
    Orbax Checkpoint to save the model's state.

    Args:
        model: The `nnx.Module` instance whose state is to be saved.
        path_str: The string representation of the file path where the model
                  state should be saved.

    """
    model_path = Path(path_str)

    model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpointer = ocp.PyTreeCheckpointer()

    state = nnx.state(model)

    checkpointer.save(model_path, state)


def print_dataset_info(train_dataset, val_dataset):
    """Print summary information about the training and validation datasets.

    This helps in quickly understanding the size and characteristics of the datasets.

    Args:
        train_dataset: The training dataset object. It's expected to have `__len__`
                       and `classes` attributes.
        val_dataset: The validation dataset object. It's expected to have `__len__`
                     and `classes` attributes.

    """
    print("📊 Dataset Info:")

    print(f"  Train samples: {len(train_dataset)}")

    print(f"  Validation samples: {len(val_dataset)}")

    print(f"  Classes: {train_dataset.classes}")

    print(f"  Number of classes: {len(train_dataset.classes)}")
