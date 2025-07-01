from pathlib import Path


def get_image_extensions() -> tuple[str, ...]:
    """Returns a tuple of common image file extensions.

    This function provides a centralized list of supported image file types,
    making it easy to manage and update.

    Returns:
        A tuple of strings, where each string is an image file extension
        (e.g., ".jpg", ".png").

    """
    return (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


class ImageFolderDataSource:
    """A data source class that mimics the structure of torchvision.datasets.ImageFolder.

    It expects the root directory to contain subdirectories, where each subdirectory
    represents a class and contains images belonging to that class.

    Example directory structure:
    root_dir/
    ├── class_a/
    │   ├── image1.jpg
    │   └── image2.png
    └── class_b/
        ├── image3.jpeg
        └── image4.bmp
    """

    def __init__(self, root_dir: str):
        """Initializes the ImageFolderDataSource.

        Args:
            root_dir: The path to the root directory containing class subdirectories.

        """
        self.root_dir = Path(root_dir)
        self.samples: list[tuple[str, int]] = []
        self.classes: list[str] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Loads image file paths and their corresponding class indices from the root directory.

        This method iterates through subdirectories, treats each subdirectory name as a class,
        and collects all valid image files within them.
        """
        if not self.root_dir.exists():

            msg = f"Root directory {self.root_dir} does not exist."
            raise FileNotFoundError(msg)

        class_to_idx = {}
        valid_extensions = get_image_extensions()



        class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        class_dirs.sort(key=lambda x: x.name)

        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = len(class_to_idx)
            class_to_idx[class_name] = class_idx
            self.classes.append(class_name)


            for ext in valid_extensions:

                for img_path in class_dir.glob(f"*{ext}"):

                    self.samples.append((str(img_path), class_idx))

        if not self.samples:

            msg = f"No valid images found in directory '{self.root_dir}'"
            raise RuntimeError(msg)

    def __len__(self) -> int:
        """Returns the total number of samples (images) in the dataset.
        This allows the dataset object to be used with `len()`.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[str, int]:
        """Returns a sample (image path and its class index) at the given index.
        This allows the dataset object to be indexed like a list (e.g., `dataset[0]`).
        """
        return self.samples[index]


def create_datasets(params):
    """Creates training and validation dataset instances using ImageFolderDataSource.

    Args:
        params: A dictionary containing configuration parameters, specifically
                "train_data_path" and "val_data_path".

    Returns:
        A tuple containing the initialized training dataset and validation dataset objects.

    """
    train_dataset = ImageFolderDataSource(params["train_data_path"])


    val_dataset = ImageFolderDataSource(params["val_data_path"])

    return train_dataset, val_dataset


def print_dataset_info(train_dataset, val_dataset):
    """Prints summary information about the training and validation datasets.

    Args:
        train_dataset: The initialized training dataset object (an instance of ImageFolderDataSource).
        val_dataset: The initialized validation dataset object (an instance of ImageFolderDataSource).

    """
    print("📊 Dataset Info:")

    print(f"  Train samples: {len(train_dataset)}")

    print(f"  Validation samples: {len(val_dataset)}")

    print(f"  Classes: {train_dataset.classes}")

    print(f"  Number of classes: {len(train_dataset.classes)}")
