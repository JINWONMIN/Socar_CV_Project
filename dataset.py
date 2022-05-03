import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, datasets


from util import *

## dataloader 구현
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, class_, mode, data_dir="accida_segmentation_dataset_v1", transform=None, task=None, opts=None):
        self.class_ = class_
        self.mode= mode
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts
        # self.to_tensor = ToTensor()

    # input / label data를 나눠서 불러오기
        lst_data = os.listdir(self.data_dir + '/'+ class_+ '/' + mode + '/images')
        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('png') | f.endswith('jpeg')]   # 확장자가 jpg, png인 파일만 로드


        # lst_data.sort()

        self.lst_data = lst_data    # declare lst_data instance

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):   # index를 인자로 받아서 index에 해당하는 파일을 로드해서 리턴하는 형태로 정의

        img_path = "accida_segmentation_dataset_v1/" + self.class_ + '/' + self.mode + '/images'
        mask_path = "accida_segmentation_dataset_v1/" + self.class_ + '/' + self.mode + '/masks'

        img = plt.imread(os.path.join(img_path, self.lst_data[index]))
        mask = plt.imread(os.path.join(mask_path, self.lst_data[index]))
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        # img = np.array(Image.open(os.path.join(img_path, self.lst_data[index])).convert('RGB'))
        # mask = np.array(Image.open(os.path.join(mask_path, self.lst_data[index])).convert('L'), dtype=np.float32)

        # 이미지를 normalization 해주는 경우는 data type이 uint8인 경우에만 해줘서 data type이 uint8인 경우에만
        # 정규화 해주는 코드 작성
        if img.dtype == np.uint8:
            img = img/255.0

        if mask.dtype == np.uint8:
            mask = mask/255.0

        if img.ndim == 2:   # input 값은 최소 3d로 넣어야 해서 2d인 경우
            img = img[:, :, np.newaxis]     # 마지막 axis 임의로 생성

        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]

        # label = mask.transpose((2, 0, 1))
        # input = img.transpose((2, 0, 1))
        input = img
        label = mask

        if self.task == "denoising":
            input = add_noise(img, type=self.opts[0], opts=self.opts[1])
        elif self.task == "inpainting":
            input = add_sampling(img, type=self.opts[0], opts=self.opts[1])
        elif self.task == "super_resolution":
            input = add_blur(img, type=self.opts[0], opts=self.opts[1])

        data = {'input': input, 'label': label}

        if self.transform:  # transform function를 data loader의 argument로 넣어주고,
            data = self.transform(data) # transform 함수가 정의 되어 이싸면, transform 함수르 통과한 data를 리턴

        return data

class ToTensor(object):
    def __call__(self, data):
        input, label = data['input'], data['label']

        input = input.transpose((2, 0, 1)).astype(np.float32)
        label = label.transpose((2, 0, 1)).astype(np.float32)

        data = {'input': torch.from_numpy(input), 'label': torch.from_numpy(label)}

        return data

class Normalization(object):    # https://jh-bk.tistory.com/24
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input, label = data['input'], data['label']

        input = (input - self.mean) / self.std
        label = (label - self.mean) / self.std

        data = {'input': input, 'label': label}

        return data

if __name__ == "__main__":
    test_transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
    test = CustomDataset(class_="spacing", mode="valid", transform=test_transform)
    print(len(test))
    loader = torch.utils.data.DataLoader(test)
    for idx in loader:
        print(idx['input'].shape)
        print(idx['label'].shape)
        break