## import libraries
import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import UNet
from dataset import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms, datasets
import albumentations as A

## Parser 생성
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 파라미터 세팅
parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

# class 설정
parser.add_argument("--class_", default="dent", choices=["dent", "scratch", "spacing"], type=str, dest="class_")

# train / test 모드 설정
parser.add_argument("--mode", default="train", type=str, dest="mode") # train or test를 구분하기 위한 파서
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

# data augmentation method
parser.add_argument("--task", default="denoising", choices=["denoising", "inpainting", "super_resolution"], type=str, dest="task")
parser.add_argument("--opts", nargs='+', default=["random", 30.0], dest="opts")

# 입력을 받는 이미지에 대한 사이즈를 설정할 수 있는 인자 설정 (to do 가변적으로 변경)
parser.add_argument("--ny", default=320, type=int, dest="ny")
parser.add_argument("--nx", default=480, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")

# file path
parser.add_argument("--data_dir", default="accida_segmentation_dataset_v1", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./results", type=str, dest="result_dir")

# network를 선택할 수 있는 argument 추가
parser.add_argument("--network", default="unet", choices=["unet", "resnet", "autoencoder"], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

args = parser.parse_args()

# local directory path
data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

# class type
class_ = args.class_

## training paremeters setting
mode = args.mode
train_continue = args.train_continue


lr = args.lr    # 1e-3
batch_size = args.batch_size    # 4
num_epoch = args.num_epoch  # 100

task = args.task
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]   # 옵션은 두 가지 타입으로 나누어서 저장을 한다. 옵션의 첫 번째 인자는 항상 타입을 설정하는 값을 따로 떼서 받고, 그 외에 변수들은 항상 숫자로 받기 때문에 array형태로 받도록 설정한다.

ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nker

network = args.network
learning_type = args.learning_type

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("mode: %s" % mode)

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)

print("class type: %s" % class_)

print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)

print("task: %s" % task)
print("pots: %s" % opts)

print("network: %s" % network)
print("learning type: %s" % learning_type)


## 디렉토리 생성
# 원래는 tensorboard로 결과를 받기 위해 result_dir을 만들었지만, 코랩으로 텐서보드를 확인하기가 불편해 png로 따로 결과를 저장하기 위한 디렉토리 생성
result_dir_train = os.path.join(result_dir, 'train')
result_dir_val = os.path.join(result_dir, 'val')
result_dir_test = os.path.join(result_dir, 'test')


if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir_train, 'png'))
    os.makedirs(os.path.join(result_dir_val, 'png'))

    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'numpy'))

## 네트워크 학습하기
# 학습시킬 데이터를 불러오기 위해 transform 함수 정의
# if __name__ == "__main__":
if mode == 'train':
    transform_train = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()]) # transforms.Compose: 여러 개의 transform 함수들을 묶어서 사용할 수 있다.
    transform_val = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_train = CustomDataset(class_=class_, mode="train", transform=transform_train)  # train set 불러오기
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

    dataset_val = CustomDataset(class_=class_, mode="valid", transform=transform_val)  # val set 불러오기
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)

    ## 그 밖의 부수적인 variables 설정
    # training / val set의 갯수를 설정하는 변수 설정
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    transform_test = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])  # transforms.Compose: 여러 개의 transform 함수들을 묶어서 사용할 수 있다.

    dataset_test = CustomDataset(class_=class_, mode=test, transform=transform_test, task=task, opts=opts)  # test set 불러오기
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

## 네트워크 생성하기
if network == "unet":
    net = UNet(nch=nch, nker=nker, norm="bnorm", learning_type=learning_type).to(device)
# elif network == "resnet":
#     net = ResNet().to(device)

## 손실함수 정의하기
# fn_loss = nn.BCEWithLogitsLoss().to(device)   # crossentropy는 classification을 하기 위한 loss function.

fn_loss = nn.MSELoss().to(device)   # L2 Loss

## Optimizer 설정
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그밖의 부수적인 functions 설정
# .transpose(0, 2, 3, 1)
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)    # tonumpy: from tensor to numpy
fn_denorm = lambda x, mean, std: (x * std) + mean   # denormalization()
# fn_class = lambda  x: 1.0 * (x > 0.5)   # classification() using thresholding (p=0.5) <네트워크 output 이미지를 binary class로 분류해주는 function

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## 네트워크 학습시키기
st_epoch = 0    # 트레이닝이 시작되는 epoch의 position을 0으로 설정

# TRAIN MODE
if mode == 'train':
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train() # network에게 training mode임을 알려주는 train() 활성화
        loss_mse = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            input = data['input'].to(device)
            label = data['label'].to(device)

            output = net(input)

            # backward pass
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # 손실함수 계산

            loss_mse += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_mse)))


            # Tensorboard 저장 (label, input, output)
            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            input = np.clip(input, a_min=0, a_max=1)
            output = np.clip(output, a_min=0, a_max=1)

            id = num_batch_train * (epoch - 1) + batch

            plt.imsave(os.path.join(result_dir_train, 'png', '%04d_label.png' % id), label[0].squeeze(), cmap='gray', vmin=0, vmax=255)  # batch로 설정된 이미지 중 첫 번째 batch에 들어있는 이미지만 저장
            plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input.png' % id), input[0].squeeze())
            plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0].squeeze())

            writer_train.add_image('label', label, num_batch_train * (epoch -1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch -1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch -1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_mse), epoch)   # loss를 텐서보드에 저장
        # network validation
        with torch.no_grad(): # validation 부붑은 backpropergation하는 부분이 없기때문에 backpropergation을 막기 위해 torch.no_grad() 을 활성화.
            net.eval()  # network에게 validation mode임을 알려주는 train() 활성화
            loss_mse = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # 손실함수 계산
                loss = fn_loss(output, label)

                loss_mse += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_mse)))

                # Tensorboard 저장 (label, input, output)
                label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

                input = np.clip(input, a_min=0, a_max=1)
                output = np.clip(output, a_min=0, a_max=1)

                id = num_batch_val * (epoch - 1) + batch

                plt.imsave(os.path.join(result_dir_val, 'png', '%04d_label.png' % id), label[0].squeeze(), cmap='gray', vmin=0, vmax=255)
                plt.imsave(os.path.join(result_dir_val, 'png', '%04d_input.png' % id), input[0].squeeze())
                plt.imsave(os.path.join(result_dir_val, 'png', '%04d_output.png' % id), output[0].squeeze())

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

            writer_val.add_scalar('loss', np.mean(loss_mse), epoch)  # loss를 텐서보드에 저장

            if epoch % 50 == 0: # (n(50)번 마다 저장하고 싶을 때 사용)
                save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)  # epoch가 한 번 진행될때마다 네트워크 저장

        writer_train.close() # 학습이 완료되면 tensorboard를 저장하기 위해 생성했던 두 개의 writer를 close()
        writer_val.close()

# TEST MODE
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():  # validation 부붑은 backpropergation하는 부분이 없기때문에 backpropergation을 막기 위해 torch.no_grad() 을 활성화.
        net.eval()  # network에게 validation mode임을 알려주는 eval() 활성화
        loss_mse = []

        for batch, data in enumerate(loader_test, 1):
            # forward path
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # 손실함수 계산
            loss = fn_loss(output, label)

            loss_mse += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_mse)))

            # Tensorboard 저장 (label, input, output)
            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                label_ = label[j]
                input_ = input[j]
                output_ = output[j]

                np.save(os.path.join(result_dir, 'numpy', '%04d_label.png' % id), label_)
                np.save(os.path.join(result_dir, 'numpy', '%04d_input.png' % id), input_)
                np.save(os.path.join(result_dir, 'numpy', '%04d_output.png' % id), output_)

                label_ = np.clip(label_, a_min=0, a_max=1)
                input_ = np.clip(input_, a_min=0, a_max=1)
                output_ = np.clip(output_, a_min=0, a_max=1)

                # save as png type
                plt.imsave(os.path.join(result_dir, 'png', '%04d_label.png' % id), label_)
                plt.imsave(os.path.join(result_dir, 'png', '%04d_input.png' % id), input_)
                plt.imsave(os.path.join(result_dir, 'png', '%04d_output.png' % id), output_)


    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" % (batch, num_batch_test, np.mean(loss_mse)))
