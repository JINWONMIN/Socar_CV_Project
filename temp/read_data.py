import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# 데이터 불러오기
dent_data_dir = './accida_segmentation_dataset_v1/dent/train'

dent_train_input = '/images/*.jpg'
dent_train_label = '/masks/*.jpg'

dent_train_img_input = Image.open(os.path.join(dent_data_dir, dent_train_input))
dent_train_img_label = Image.open(os.path.join(dent_data_dir, dent_train_label))

ny, nx = dent_train_img_label.size  # 512 x 512 size







class Dataset(torch.utils.data.Dataset):    # Dataset 클래스에 torch.utils.data.Dataset 클래스를 상속
    def __init__(self, data_dir, transform=None):   # 할당 받을 인자 선언 (첫 선언)
        self.data_dir = data_dir
        self.transform = transform

        # prefixed word를 이용해 prefixed 되어 있는 input / label data를 나눠서 불러오기
        lst_data = os.listdir(self.data_dir) # os.listdir 메소드를 이용해서 data_dir에 있는 모든 파일을 불러온다.

        # prefixed 되어있는 word를 기반으로 label / input list를 정렬
        lst_label = [f for f in lst_data if f.startswith('label')]  # startswith 메소드를 통해
        lst_input = [f for f in lst_data if f.startswith('input')]  # prefixed 돼있는 리스트만 따로 정리.

        lst_label.sort()
        lst_label.sort()

        self.lst_label = lst_label  # 정렬된 리스트를
        self.lst_input = lst_input  # 해당 클래스의 파라미터로 가져온다.

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):   # index를 인자로 받아서 index에 해당하는 파일을 로드해서 리턴하는 형태로 정의
        label = np.load(os.path.join(self.data_dir, self.lst_label[index])) # NumPy 형식으로 저장되어 있어 np.load로 불러온다.
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label / 255.0   # 데이터가 0 ~ 255 사이의 값으로 있기 때문에
        input = input / 255.0   # 이를 0 ~ 1 사이로 normalize

        if label.ndim == 2:     # input 값은 최소 3차원으로 넣어야 해서 2차원인 경우
            label = label[:, :, np.newaxis]     # 마지막 axis 임의로 생성
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label} # 생성된 label 과 input을 딕셔너리 형태로 저장

        if self.transform:  # transform function을 data loader의 argument로 넣어주고,
            data = self.transform(data)     # transform 함수가 정의 되어 있다면, transform 함수를 통과한 data를 리턴

        return data

