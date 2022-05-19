import os
import torch

sweep_config = {
    'method': 'grid',
    'name':'grid-socar_sweep_v13_spacing',
    'metric' : {
        'name': 'Best val IoU',
        'goal': 'maximize'   
        },
    'parameters' : {
        'epochs': {
            'value' : 5},
        'batch_size': {
            'value' : 16},
        'optimizer': { 
            'value': 'adam'},
        'model': {
            'value': 'imagenet-b1'},
        'loss': { 
            'value': 'focal'},
        'img_size': { 
            'value': 224},
        'seed':{
            'value': 0},
        'learning_rate': {
            'value': 1e-3}, 
    }
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def mkdir(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

data_folder = 'dent'

ROOT_DIR = os.path.join('/content/drive/MyDrive/socar/models-main/models-main/EfficientUnet_VIAI')
DATA_DIR = os.path.join(ROOT_DIR, 'segmentation_dataset_v1', data_folder)
CKPT_DIR = os.path.join(ROOT_DIR, 'checkpoints_dir', 'checkpoints_')

RESULTS_DIR = os.path.join(ROOT_DIR, 'test_results_dir', 'test_results_')
INFER_DIR  = os.path.join(ROOT_DIR, 'inference_dir', data_folder)

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_IMGS_DIR = os.path.join(TRAIN_DIR, 'images')
VAL_IMGS_DIR = os.path.join(VAL_DIR, 'images')
TEST_IMGS_DIR = os.path.join(TEST_DIR, 'images')

TRAIN_LABELS_DIR = os.path.join(TRAIN_DIR, 'masks')
VAL_LABELS_DIR = os.path.join(VAL_DIR, 'masks')
TEST_LABELS_DIR = os.path.join(TEST_DIR, 'masks')

mkdir(
    CKPT_DIR, RESULTS_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, TRAIN_IMGS_DIR, VAL_IMGS_DIR, 
    TEST_IMGS_DIR, TRAIN_LABELS_DIR, VAL_LABELS_DIR, TEST_LABELS_DIR, 
    )

# Hyper parameters
class Config:
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 4
    NUM_EPOCHS = 30
