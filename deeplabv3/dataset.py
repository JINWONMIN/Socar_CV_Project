from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch.utils.data
from PIL import Image
from torchvision.datasets.vision import VisionDataset

class CustomDataset(torch.utils.data.Dataset):
    """The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folder: str,
                 class_: str = "dent",
                 transforms: Optional[Callable] = None,
                 subset: str = None,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "grayscale") -> None:

        """
        Params:
            :param root (str): Root directory path.
            :param image_folder (str): Name of the folder that contains the images in the root directory.
            :param mask_folder (str): Name of the folder that contains the masks in the root directory.
            :param class_ (str, optional): Name of the class folder. 'dent' or 'scratch' or 'spacing'.
            Defaults to 'dent'.
            :param transforms (optional[Callable], optional): A function/transform that takes in a sample and
            returns a transformed version.
            E.g, ''transforms.ToTensor'' for images. Defaults to None.
            :param subset (str, optional): 'Train' or 'Test' or 'Valid' to select the appropriate set. Defaults to None:
            :param image_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'rgb'.
            :param mask_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.
        Raises:
            :raise ValueError: If class_ is not either 'dent' or 'scratch' or 'spacing'
            :raise OSError: If image folder doesn't exist in root.
            :raise OSError: If mask folder doesn't exist in root.
            :raise ValueError: If subset is not either 'Train' or 'Test' or 'Valid'
            :raise ValueError: If image_color_mode and mask_color_mode are either 'rgb' or 'grayscale'
        """
        self.root = root

        image_folder_path = Path(self.root) / class_ / subset / image_folder
        mask_folder_path = Path(self.root) / class_ / subset / mask_folder
        # exception
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist.")

        if not class_ not in ["dent", "scratch", "spacing"]:
            raise ValueError(
                f"{class_} is an invalid choice. enter from dent or scratch or spacing"
            )
        self.class_ = class_

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. enter from rgb or grayscale"
            )
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. enter from rgb or grayscale"
            )

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        if subset not in ["Train", "Test", "Valid"]:
            raise (ValueError(
                f"{subset} is not a valid input. Acceptable values are Train and Test, Valid"
            ))
        self.image_names = image_folder_path.glob("*")
        self.mask_names = mask_folder_path.glob("*")

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            image = Image.open(image_file)
            if self.image_color_mode == "rgb":
                image = image.convert("RGB")
            elif self.image_color_mode == "grayscale":
                image = image.convert("L")
            mask = Image.open(mask_file)
            if self.mask_color_mode == "rgb":
                mask = mask.convert("RGB")
            elif self.mask_color_mode == "grayscale":
                mask = mask.convert("L")
            data = {"image": image, "mask": mask}
            if self.transforms:
                data["image"] = self.transforms(data["image"])
                data["mask"] = self.transforms(data["mask"])
            return data













