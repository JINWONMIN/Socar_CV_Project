import os
import numpy as np
from scipy.stats import poisson
from skimage.transform import rescale, resize

import torch
import torch.nn as nn


## 네트워크 저장
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))


## 네트워크 로드
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch


## smapling 하기
def add_sampling(img, type="random", opts=None):
    # global dst
    sz = img.shape

    if type == "uniform":
        ds_y = opts[0].astype(np.int)  # interval of y
        ds_x = opts[1].astype(np.int)  # interval of x

        msk = np.zeros(sz)  # create mask
        msk[::ds_y, ::ds_x, :] = 1  # sampling at interval ds_y and ds_x

        dst = img * msk  # mak의 간격으로 sampling하기 위해 곱해준다.
    elif type == "random":
        rnd = np.random.rand(sz[0], sz[1], sz[2])  # 1) random sampling을 하기 위해 uniform size와 동일안 random variables 생성
        prob = opts[0]  # sampling 할 비율
        msk = (rnd > prob).astype(np.float)

        # rnd = np.random.rand(sz[0], sz[1], 1)  # 채널 방향으로 복사해주면 된다. (채널 방향으로는 동일한 마스크가 진행된다.)
        # prob = 0.5
        # msk = (rnd > prob).astype(np.float)
    elif type == "gaussian":
        x0 = opts[0]  # center를 기준
        y0 = opts[1]  # center를 기준
        sgmx = opts[2]
        sgmy = opts[3]
        a = opts[4]

        ly = np.linspace(-1, 1, sz[0])
        lx = np.linspace(-1, 1, sz[1])

        x, y = np.meshgrid(lx, ly)

        gaus = a * np.exp(
            - ((x - x0) ** 2 / (2 * sgmx ** 2) + (y - y0) ** 2 / (2 * sgmy ** 2)))  # 2d gaussian distribution 공식
        gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, sz[2]))
        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd < gaus).astype(np.float)

        # gaus = a * np.exp(- ((x - x0) ** 2 / (2 * sgmx ** 2) + (y - y0) ** 2 / (2 * sgmy ** 2)))
        # gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, 1))
        # rnd = np.random.rand(sz[0], sz[1], 1)
        # msk = (rnd < gaus).astype(np.float)
        # msk = np.tile(msk, (1, 1, sz[2]))

        dst = img * msk
    return dst


## Noise 추가하기
def add_noise(img, type="random", opts=None):
    # global dst
    sz = img.shape

    if type == "random":
        sgm = opts[0]
        noise = sgm / 255.0 * np.random.randn(sz[0], sz[1], sz[
            2])  # 이미지가 0~1 사이로 normalization이 되어 있어서 시그마에도 normalization을 해줘서 스케일을 맞춰준다.
        dst = img + noise
    elif type == "poisson":
        dst = poisson.rvs(
            255.0 * img) / 255.0  # https://hyperpolyglot.org/numerical-analysis2 (MATLAB 과 python가 동일한 함수가 정리되어 있음.)
        noise = dst - img

    return dst


## Blurring 추가하기
def add_blur(img, type="bilinear", opts=None):
    '''
    -------------------------
    order options
    -------------------------
    0: Nearest-neighbor
    1: Bi-linear (default)
    2: Bi-quadratic
    3: Bi-cubic
    4: Bi-quartic
    5: Bi-quintic
    '''
    if type == "nearest":
        order = 0
    elif type == "bilinear":
        order = 1
    elif type == "biquadratic":
        order = 2
    elif type == "bicubic":
        order = 3
    elif type == "biquartic":
        order = 4
    elif type == "quintic":
        order = 5

    sz = img.shape

    ds = opts[0]

    if len(opts) == 1:  # opts 변수의 2번째 인자를 설정 하지 않은 경우에
        keepdim = True  # dim을 input 차원과 같게 다시 upsampling을 한다.
    else:  # 그게 아니라면
        keepdim = opts[1]  # opts 변수의 두 번재 인자를 이전 값으로 설정

    # dst = rescale(img, scale(dw, dw, 1), order=order)   # rescale은 인자로 넣은 dw ratio를 스케일로 활용을 해서 크기를 조정
    dst = resize(img, output_shape=(sz[0] // opts[0], sz[1] // opts[0], sz[2]), order=order)  # resize는 output shape를 고정해주면 그 shape에 맞게 조정
    '''
    rescale 와 resize 함수는 동일하게 동작을 한다. 하지만 rescale함수에서 스케일 벡터로 사이즈를 조절 하다보면 고정된 크기로
    스케일러 스케일 다운이 되지 않는 경우들이 종종 존재하게 된다.
    그래서 output size를 고정을 해서 사용할 수 있는 resize 함수를 활용하는 것이 좋다.

    이제 keepdim 이라는 플래그에 근간하여 이미지를 upsampling 할지 아니면 destination matrix 그래도 리턴할지 확인하는
    if문 작성
    '''
    if keepdim:  # 만약 dimension을 그대로 유지한다면
        dst = resize(dst, output_shape=(sz[0], sz[1], sz[2]), order=order)  # 오리지널 사이즈로 사이즈를 다시 리사이즈 해주는

    return dst  # destination matrix로 리턴








