from pathlib import Path


def get_image_extensions() -> tuple[str, ...]:
    return (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


class ImageFolderDataSource:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)  # Use pathlib.Path
        self.samples: list[tuple[str, int]] = []
        self.classes: list[str] = []
        self._load_samples()

    def _load_samples(self) -> None:
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

            # Move image file collection inside the class_dir loop
            for ext in valid_extensions:
                for img_path in class_dir.glob(f"*{ext}"):
                    self.samples.append((str(img_path), class_idx))

        if not self.samples:
            msg = f"No valid images found in directory '{self.root_dir}'"
            raise RuntimeError(msg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.samples[index]
