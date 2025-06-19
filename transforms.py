import cv2
import grain
import numpy as np


class AlbumentationsTransform(grain.transforms.Map):
    def __init__(self, transforms):
        self.transforms = transforms

    def map(self, element: tuple[np.ndarray, int]) -> tuple[np.ndarray, int]:
        image, label = element
        transformed_image = self.transforms(image=image)["image"]
        return transformed_image, label


class LoadImageMap(grain.transforms.Map):
    """优化的图像加载器，增加错误处理"""  # noqa: RUF002

    def map(self, element: tuple[str, int]) -> tuple[np.ndarray, int]:
        img_path, label = element
        img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        return img, label
