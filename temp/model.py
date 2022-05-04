import os
import numpy as np

import torch
import torch.nn as nn

from layer import *


## 네트워크 구축
# U-Net Layer Create
class UNet(nn.Module):  # nn.Module이라는 클래스를 UNet 클래스에 상속
    def __init__(self, nch, nker, norm="bnorm", learning_type= "plain"):
        super(UNet, self).__init__()
        self.learning_type = learning_type

        # Contracting path
        '''
        kernel_size, stride, padding, bias는 cbr layer를 생성함에 있어서 항상 변하지 않고, 고정이 되어 있기 때문에 
        위에서 미리 predefined를 해 놓음. 그래서 삭제해도 된다.
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)                       
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True) 
        '''
        self.enc1_1 = CBR2d(in_channels=nch, out_channels=1 * nker, norm=norm)     # 1 of Encoder stage 1
        self.enc1_2 = CBR2d(in_channels=1 * nker, out_channels=1 * nker,  norm=norm)    # 2 of Encoder stage 1

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=1 *nker, out_channels=2 * nker,  norm=norm)   # 1 of Encoder stage 2
        self.enc2_2 = CBR2d(in_channels=2 * nker, out_channels=2 * nker,  norm=norm)  # 2 of Encoder stage 2

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=2 * nker, out_channels=4 * nker,  norm=norm)  # 1 of Encoder stage 3
        self.enc3_2 = CBR2d(in_channels=4 * nker, out_channels=4 * nker,  norm=norm)  # 2 of Encoder stage 3

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=4 * nker, out_channels=8 * nker,  norm=norm)  # 1 of Encoder stage 4
        self.enc4_2 = CBR2d(in_channels=8 * nker, out_channels=8 * nker,  norm=norm)  # 2 of Encoder stage 4

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=8 * nker, out_channels=16 * nker,  norm=norm)  # final Encoder stage


        # Expansive path  <encoder와 매칭을 해주기 위해 encoder 역순으로 진행>
        self.dec5_1 = CBR2d(in_channels=16 * nker, out_channels=8 * nker,  norm=norm)   # Decoder stage 5

        self.unpool4 = nn.ConvTranspose2d(in_channels=8 * nker, out_channels=8 * nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 8 * nker, out_channels=8 * nker,  norm=norm)  # bottom stage 512 + 2 of Encoder stage 4 512 = 1024가 input channel (first start decoder part)
        self.dec4_1 = CBR2d(in_channels=8 * nker, out_channels=4 * nker,  norm=norm)

        self.unpool3 = nn.ConvTranspose2d(in_channels=4 * nker, out_channels=4 * nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 4 * nker, out_channels=4 * nker,  norm=norm) # skip connection volumn이 존재하므로 input_channels 두배가 된다. <enc3_2 output channel=256>
        self.dec3_1 = CBR2d(in_channels=4 * nker, out_channels=2 * nker,  norm=norm)

        self.unpool2 = nn.ConvTranspose2d(in_channels=2 * nker, out_channels=2 * nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 2 * nker, out_channels=2 * nker,  norm=norm) # skip connection volumn이 존재하므로 input_channels 두배가 된다. <enc2_2 output channel=128>
        self.dec2_1 = CBR2d(in_channels=2 * nker, out_channels=1 * nker,  norm=norm)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 1 * nker, out_channels=1 * nker,  norm=norm) # skip connection volumn이 존재하므로 input_channels 두배가 된다. <enc1_2 output channel=64>
        self.dec1_1 = CBR2d(in_channels=1 * nker, out_channels=1 * nker,  norm=norm)

        # segmentation에 필요한 n개의 클래스에 대한 output을 만들어주기 위해 up-conv(2x2) 정의
        self.fc = nn.Conv2d(in_channels=1 * nker, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    # designed layer connection
    def forward(self, x):   # x is input images
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)    # First encoder stage
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)    # Second encoder stage
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)    # Third encoder stage
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)    # Fourth encoder stage
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)     # Final encoder stage

        dec5_1 = self.dec5_1(enc5_1)    # First decoder stage

        unpool4 = self.unpool4(dec5_1)  # First up-conv <파란색 부분으로 출력>
        # 흰색 부분과 파란색 부분을 연결해주는 레이어 생성 (일반적으로 채널방향으로 연결해주는 함수를 cat이라 부름)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)  # dim=[0:batch, 1:channel, 2:height(y), 3:width(x)]
        dec4_2 = self.dec4_2(cat4)  # conv(1x1) 선미에 있는 블럭을 연결해주는 코드
        dec4_1 = self.dec4_1(dec4_2)    # 두 번째 conv(1x1) 선미에 있는 블럭을 연결해 주는 코드

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        if self.learning_type == "plain":
            x = self.fc(dec1_1)
        elif self.learning_type == "residual":
            x = x + self.fc(dec1_1)

        return x