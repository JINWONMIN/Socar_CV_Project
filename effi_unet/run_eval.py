import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import transforms

import wandb
import sweep_eval

from efficientunet import *
from dataset import *
from config import *
from adabelief_pytorch import AdaBelief
from losses import FocalLoss

def wandb_setting(sweep_config=None):
    wandb.init(config=sweep_config)
    w_config = wandb.config
    name_str = str(w_config.model) + ' | ' +  str(w_config.img_size) + ' | ' +  str(w_config.batch_size) 
    wandb.run.name = name_str

    #########Random seed 고정해주기###########
    random_seed = w_config.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    ###########################################

    test_transform = A.Compose([
        A.Resize(w_config.img_size, w_config.img_size),
        A.Normalize(mean=(0.485), std=(0.229)),
        transforms.ToTensorV2(),
    ])

    ##########################################데이터 로드 하기#################################################
    batch_size= w_config.batch_size

    test_dataset = DatasetV2(imgs_dir=TEST_IMGS_DIR, mask_dir=TEST_LABELS_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    #############################################################################################################

    test_data_num = len(test_dataset)
    test_batch_num = int(np.ceil(test_data_num / batch_size)) 

    if w_config.model == 'imagenet-b1':
        net = get_efficientunet_b1(out_channels=2, concat_input=True, pretrained=True).to(device)
    elif w_config.model == 'stfd-ssl-b4':
        net = get_stanford_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True).to(device)
    elif w_config.model == 'socar-ssl-b4':
        net = get_socar_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True).to(device)
    
    # Loss Function
    if w_config.loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss().to(device)
    elif w_config.loss == 'focal':
        criterion = FocalLoss(gamma=2, alpha=0.5).to(device)

    # Optimizer
    if w_config.optimizer == 'sgd':
        optimizer_ft = torch.optim.SGD(net.parameters(), lr=w_config.learning_rate, momentum=0.9)# optimizer 종류 정해주기
    elif w_config.optimizer == 'adam':
        optimizer_ft = torch.optim.Adam(params=net.parameters(), lr=w_config.learning_rate)
    elif w_config.optimizer == 'adabelief':
        optimizer_ft = AdaBelief(net.parameters(), lr=w_config.learning_rate, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = True)
    
    ckpt_dir = CKPT_DIR + name_str

    wandb.watch(net, log='all') 

    sweep_eval.eval_model(test_loader, test_batch_num, net, criterion, optimizer_ft, ckpt_dir, wandb, w_config=w_config)

project_name = '[VIAI] EVAL' # 프로젝트 이름을 설정해주세요.
entity_name  = 'viai' # 사용자의 이름을 설정해주세요.
sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity_name)

wandb.agent(sweep_id, wandb_setting, count=20)