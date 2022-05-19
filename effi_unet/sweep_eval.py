import numpy as np
import matplotlib.pyplot as plt

import torch
from efficientunet import *
from dataset import *
from config import *
from utils import *
from metric import iou_score

def eval_model(test_loader, test_batch_num, net, criterion, optim, ckpt_dir, wandb, w_config):
    # Load Checkpoint File
    if os.listdir(ckpt_dir):
        net, optim, ckpt_path = load_net(ckpt_dir=ckpt_dir, net=net, optim=optim)

    result_dir = os.path.join(INFER_DIR, 'test_' + ckpt_path.split('/')[-1][:-4])

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Evaluation
    with torch.no_grad():
        net.eval()  # Evaluation Mode
        loss_arr, iou_arr = [], []

        for batch_idx, data in enumerate(test_loader, 1):
            # Forward Propagation
            img = data['img'].to(device)
            label = data['label'].to(device)

            label = label // 255

            output = net(img)
            output_t = torch.argmax(output, dim=1).float()

            # Calc Loss Function
            loss = criterion(output, label)
            iou = iou_score(output_t, label)

            loss_arr.append(loss.item())
            iou_arr.append(iou.item())
            
            print_form = '[Test] | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f} | IoU: {:.4f}'
            print(print_form.format(batch_idx, test_batch_num, loss_arr[-1], iou))

            img = to_numpy(denormalization(img, mean=0.5, std=0.5))
            # 이미지 캐스팅
            img = np.clip(img, 0, 1) 

            label = to_numpy(label)
            output_t = to_numpy(classify_class(output_t))
            
            for j in range(label.shape[0]):
                crt_id = int(w_config.batch_size * (batch_idx - 1) + j)
                
                plt.imsave(os.path.join(result_dir, f'img_{crt_id:04}.png'), img[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, f'label_{crt_id:04}.png'), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, f'output_{crt_id:04}.png'), output_t[j].squeeze(), cmap='gray')
    
    eval_loss_avg = np.mean(loss_arr)
    eval_iou_avg  = np.mean(iou_arr)
    print_form = '[Result] | Avg Loss: {:0.4f} | Avg IoU: {:0.4f}'
    wandb.log({'eval_loss': eval_loss_avg , 'eval_iou': eval_iou_avg}, commit=False)
    print(print_form.format(eval_loss_avg, eval_iou_avg))