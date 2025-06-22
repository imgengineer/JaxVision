import cv2
import grain
import numpy as np
from PIL import Image


class AlbumentationsTransform(grain.transforms.Map):
    def __init__(self, transforms):
        self.transforms = transforms

    def map(self, element: tuple[np.ndarray, int]) -> tuple[np.ndarray, int]:
        image, label = element
        transformed_image = self.transforms(image=image)["image"]
        return transformed_image, label


class OpenCVLoadImageMap(grain.transforms.Map):
    def map(self, element: tuple[str, int]) -> tuple[np.ndarray, int]:
        img_path, label = element
        img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        return img, label


class PILoadImageMap(grain.transforms.Map):
    def map(self, element: tuple[str, int]) -> tuple[np.ndarray, int]:
        img_path, label = element
        img = np.asarray(Image.open(img_path).convert(mode="RGB"))
        return img, label
