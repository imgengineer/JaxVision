import random
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp
from flax import nnx


def create_model(model, seed, num_classes) -> nnx.Module:
    """Create a new model."""
    return model(rngs=nnx.Rngs(seed), num_classes=num_classes)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.default_rng(seed)
    random.seed(seed)


def load_model(path, num_classes):
    """Load model from checkpoint."""
    model = nnx.eval_shape(lambda: create_model(0, num_classes))
    state = nnx.state(model)

    checkpointer = ocp.PyTreeCheckpointer()
    state = checkpointer.restore(path, item=state)

    nnx.update(model, state)
    return model


def save_model(model, path_str: str):
    """Save model state."""
    model_path = Path(path_str)
    model_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    checkpointer = ocp.PyTreeCheckpointer()
    state = nnx.state(model)
    checkpointer.save(model_path, state)


def print_dataset_info(train_dataset, val_dataset):
    """Print dataset information."""
    print("📊 Dataset Info:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Classes: {train_dataset.classes}")
    print(f"  Number of classes: {len(train_dataset.classes)}")
