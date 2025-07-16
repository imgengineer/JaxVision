from pathlib import Path


class Config:
    batch_size = 32
    target_size = 224
    learning_rate = 5e-4
    weight_decay = 1e-4
    num_workers = 64
    seed = 42
    num_classes = 6
    num_epochs = 300
    train_data_path = Path("/root/JaxVision/data/Original Images/Original Images/FOLDS/fold1/Train")
    val_data_path = Path("/root/JaxVision/data/Original Images/Original Images/FOLDS/fold1/Valid")
    checkpoint_dir = Path("/root/JaxVision/checkpoints")
