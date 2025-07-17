from pathlib import Path


class Config:
    batch_size: int = 256
    target_size: int = 224
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    num_workers: int = 64
    seed: int = 42
    num_classes: int = 6
    num_epochs: int = 300
    train_data_path: Path = Path("/root/JaxVision/data/Original Images/Original Images/FOLDS/fold1/Train")
    val_data_path: Path = Path("/root/JaxVision/data/Original Images/Original Images/FOLDS/fold1/Valid")
    checkpoint_dir: Path = Path("/root/JaxVision/checkpoints")
