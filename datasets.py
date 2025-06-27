from pathlib import Path  # Import the Path class from the pathlib module for object-oriented filesystem paths


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
        self.root_dir = Path(root_dir)  # Convert the input string path to a pathlib.Path object
        self.samples: list[tuple[str, int]] = []  # List to store (image_path, class_index) tuples
        self.classes: list[str] = []  # List to store class names in sorted order
        self._load_samples()  # Automatically load samples during initialization

    def _load_samples(self) -> None:
        """Loads image file paths and their corresponding class indices from the root directory.

        This method iterates through subdirectories, treats each subdirectory name as a class,
        and collects all valid image files within them.
        """
        if not self.root_dir.exists():
            # Raise an error if the specified root directory does not exist.
            msg = f"Root directory {self.root_dir} does not exist."
            raise FileNotFoundError(msg)

        class_to_idx = {}  # Dictionary to map class names to integer indices
        valid_extensions = get_image_extensions()  # Get the tuple of valid image extensions

        # Get all subdirectories within the root_dir and sort them by name.
        # This ensures a consistent mapping of class names to indices.
        class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        class_dirs.sort(key=lambda x: x.name)

        for class_dir in class_dirs:
            class_name = class_dir.name  # The name of the subdirectory is the class name
            class_idx = len(class_to_idx)  # Assign a unique integer index to the class
            class_to_idx[class_name] = class_idx  # Store the mapping
            self.classes.append(class_name)  # Add the class name to the ordered list of classes

            # Iterate over all valid image extensions.
            for ext in valid_extensions:
                # Use `glob` to find all files ending with the current extension within the class directory.
                for img_path in class_dir.glob(f"*{ext}"):
                    # Add the absolute path of the image and its corresponding class index to the samples list.
                    self.samples.append((str(img_path), class_idx))

        if not self.samples:
            # Raise a runtime error if no valid images are found after scanning all directories.
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
    # Create an instance of ImageFolderDataSource for the training data
    train_dataset = ImageFolderDataSource(params["train_data_path"])

    # Create an instance of ImageFolderDataSource for the validation data
    val_dataset = ImageFolderDataSource(params["val_data_path"])

    return train_dataset, val_dataset


def print_dataset_info(train_dataset, val_dataset):
    """Prints summary information about the training and validation datasets.

    Args:
        train_dataset: The initialized training dataset object (an instance of ImageFolderDataSource).
        val_dataset: The initialized validation dataset object (an instance of ImageFolderDataSource).

    """
    print("📊 Dataset Info:")
    # Print the total number of samples found in the training dataset
    print(f"  Train samples: {len(train_dataset)}")
    # Print the total number of samples found in the validation dataset
    print(f"  Validation samples: {len(val_dataset)}")
    # Print the list of unique class names identified in the training dataset
    print(f"  Classes: {train_dataset.classes}")
    # Print the total count of unique classes
    print(f"  Number of classes: {len(train_dataset.classes)}")
