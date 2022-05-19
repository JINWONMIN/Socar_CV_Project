import numpy as np
import matplotlib.pyplot as plt

import torch
import copy
import time

from efficientunet.efficientunet import *
from dataset import *
from config import *
from utils import *
from metric import iou_score

def train_model(dataloaders, batch_num, net, criterion, optim, ckpt_dir, wandb, w_config):
    wandb.watch(net, criterion, log='all', log_freq=10)

    since = time.time()
    
    best_model_wts = copy.deepcopy(net.state_dict())
    best_iou = 0
    num_epoch = w_config.epochs

    for epoch in range(1, num_epoch+1):
        net.train()  # Train Mode
        train_loss_arr = []
        
        for batch_idx, data in enumerate(dataloaders['train'], 1):
            # Forward Propagation
            img = data['img'].to(device)
            label = data['label'].to(device)
            
            label = label // 255
            
            output = net(img)

            # Backward Propagation
            optim.zero_grad()
            
            loss = criterion(output, label)

            loss.backward()
            
            optim.step()
            
            # Calc Loss Function
            train_loss_arr.append(loss.item())

            print_form = '[Train] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'
            print(print_form.format(epoch, num_epoch, batch_idx, batch_num['train'], train_loss_arr[-1]))

        train_loss_avg = np.mean(train_loss_arr)

        # Validation (No Back Propagation)
        example_images = []
        with torch.no_grad():
            net.eval()  # Evaluation Mode
            val_loss_arr, val_iou_arr = [], []
            
            for batch_idx, data in enumerate(dataloaders['val'], 1):
                # Forward Propagation
                img = data['img'].to(device)
                label = data['label'].to(device)
                
                label = label // 255

                output = net(img)
                output_t = torch.argmax(output, dim=1).float()
                
                # Calc Loss Function
                loss = criterion(output, label)
                iou = iou_score(output_t, label)

                val_loss_arr.append(loss.item())
                val_iou_arr.append(iou.item())

                example_images.append(wandb.Image(
                    img[0], caption="Pred: {} Truth: {}".format(output[0].item(), label[0])
                ))
                
                print_form = '[Validation] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f} | IoU: {:.4f}'
                print(print_form.format(epoch, num_epoch, batch_idx, batch_num['val'], val_loss_arr[-1], iou))

        val_loss_avg = np.mean(val_loss_arr)
        val_iou_avg =  np.mean(val_iou_arr)
        # val_writer.add_scalar(tag='loss', scalar_value=val_loss_avg, global_step=epoch)
        
        if  best_iou < val_iou_avg:
            best_iou = val_iou_avg
            best_model_wts = copy.deepcopy(net.state_dict())    

        wandb.log({'train_epoch_loss': train_loss_avg,
                   'val_epoch_loss': val_loss_avg,
                   'val_epoch_iou': val_iou_avg,
                   'Examples': example_images}, step=epoch)

        print_form = '[Epoch {:0>4d}] Training Avg Loss: {:.4f} | Validation Avg Loss: {:.4f} | Validation Avg IoU: {:.4f}'
        print(print_form.format(epoch, train_loss_avg, val_loss_avg, val_iou_avg))
        
        save_net(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val IoU: {:4f}'.format(best_iou))

    wandb.log({'Best val IoU': best_iou}, commit=False)

    net.load_state_dict(best_model_wts)
    save_net(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch, is_best=True, best_iou=best_iou)
