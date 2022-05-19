import os
from glob import glob

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import transforms

class DatasetV2(Dataset):
    def __init__(self, imgs_dir, mask_dir, transform=None):
        self.img_dir = imgs_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = []
        self.masks  = []

        images_path, masks_path = [], []

        for ext in ('*.jpeg', '*.png', '*.jpg'):
            images_path.extend(sorted(glob(os.path.join(imgs_dir, ext))))
            masks_path.extend(sorted(glob(os.path.join(mask_dir, ext))))
        
        for i, m  in zip(images_path, masks_path):
            self.images.extend([Image.open(i).convert('RGB')])
            self.masks.extend([Image.open(m).convert('L')])

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask  = self.masks[idx]

        np_image = np.array(image)
        np_mask  = np.array(mask)
        
        if self.transform:
            transformed = self.transform(image=np_image, mask=np_mask)
            np_image = transformed["image"]
            np_mask = transformed["mask"]
            np_mask = np_mask.long()
        
        ret = {
            'img': np_image,
            'label': np_mask,
        }
        
        return ret

if __name__ == '__main__':

    transform = A.Compose([
                A.Resize(512, 512),
                A.Normalize(mean=0.5, std=0.5),
                transforms.ToTensorV2()
            ])

    img_dir = os.path.join(os.getcwd(), "segmentation_dataset_v1", 'dent', 'train', 'images')
    mask_dir = os.path.join(os.getcwd(), "segmentation_dataset_v1", 'dent', 'train', 'masks')

    train_dataset = DatasetV2(img_dir, mask_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=0)

    #For shape test
    for ret in iter(dataloader):
        print(ret['img'].shape, ret['label'].shape, ret['label'].type)
        break