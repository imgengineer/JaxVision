import random  # Import the random module for generating random numbers
from pathlib import Path  # Import Path from pathlib for object-oriented filesystem paths

import numpy as np  # Import numpy for numerical operations, especially for random number generation
import orbax.checkpoint as ocp  # Import Orbax Checkpoint for saving and loading model states
from flax import nnx  # Import Flax NNX for neural network modules and utilities


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
    # Initialize the model. nnx.Rngs(seed) creates a random number generator
    # specifically for the model's internal operations (e.g., weight initialization, dropout).
    return model(rngs=nnx.Rngs(seed), num_classes=num_classes)


def set_seed(seed):
    """Set random seeds for various random number generators to ensure reproducibility.

    This is crucial for research and debugging, allowing experiments to be
    replicated with the same results.

    Args:
        seed: An integer seed value.

    """
    # Set the seed for NumPy's default random number generator.
    np.random.default_rng(seed)
    # Set the seed for Python's built-in `random` module.
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
    # Create an "empty" model shape using `nnx.eval_shape`.
    # This creates a model with the correct structure but without initializing actual parameters,
    # as parameters will be loaded from the checkpoint.
    # The `lambda: create_model(0, num_classes)` provides a callable that returns
    # the model, with `0` as a placeholder for the seed as it's not used for shape evaluation.
    model = nnx.eval_shape(lambda: create_model(0, num_classes))
    # Extract the "state" (trainable parameters and mutable variables) from the model.
    state = nnx.state(model)

    # Initialize a PyTreeCheckpointer from Orbax, which handles saving/loading of JAX PyTrees.
    checkpointer = ocp.PyTreeCheckpointer()
    # Restore the model state from the given path. The `item=state` argument tells
    # Orbax to use the structure of the `state` object for restoration.
    state = checkpointer.restore(path, item=state)

    # Update the `model`'s internal state with the loaded `state`.
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
    model_path = Path(path_str)  # Convert the string path to a Path object
    # Ensure the parent directory for the model checkpoint exists.
    # `parents=True` creates any necessary intermediate directories.
    # `exist_ok=True` prevents an error if the directory already exists.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Initialize a PyTreeCheckpointer.
    checkpointer = ocp.PyTreeCheckpointer()
    # Extract the current state (parameters and mutable variables) from the model.
    state = nnx.state(model)
    # Save the extracted state to the specified path.
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
    # Print the number of samples in the training dataset.
    print(f"  Train samples: {len(train_dataset)}")
    # Print the number of samples in the validation dataset.
    print(f"  Validation samples: {len(val_dataset)}")
    # Print the list of class names from the training dataset.
    print(f"  Classes: {train_dataset.classes}")
    # Print the total number of unique classes.
    print(f"  Number of classes: {len(train_dataset.classes)}")
