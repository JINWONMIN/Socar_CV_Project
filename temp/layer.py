import torch
import torch.nn as nn

class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias)]  # 컨볼루션 레이어 정의

        if not norm is None:  # norm 이 None 아닌 경우에
            if norm == "bnorm":  # bnorm 으로 설정되어 있으면 batchnormalization 추가
                layers += [nn.BatchNorm2d(num_features=out_channels)]  # Batch normalization 정의
            elif norm == "inorm":  # inorm (instance normalization)으로 설정되어 있으면,
                layers += [nn.InstanceNorm2d(num_features=out_channels)]  # InstanceNorm2d 추가

        if not relu is None:
            layers += [nn.ReLU() if relu == 0.0 else nn.LeakyReLU(relu)]  # activation fuction(using ReLU) 정의

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)
