"""
Search script training models for different parameters.
"""

# System
import os, sys, io, glob, time, datetime

# Model
from models.uNet import *
from helpers.training import training
from datasets.aerial_dataset import AerialDataset
from datasets.patched_aerial_dataset import PatchedAerialDataset
from visualization.helpers import imshow_tensor, imshow_tensor_gt, generate_predictions, concatenate_images, img_crop
from preprocessing.augmentation_config import ImageAugmentationConfig
from preprocessing.rotation import *
from kaggle.mask_to_submission import *
import random

# ML Library
import numpy as np
import torch, torchvision
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

TRAIN_IMAGE_DATA = './data/train/images/'
TRAIN_LABEL_DATA = './data/train/groundtruth/'
TEST_IMAGE_DATA = './data/test/images/'
TEST_LABEL_DATA = './data/test/predictions/'

BATCH_SIZE = 64
NUM_EPOCHS = 50
TESTING_SIZE = 50

DATA_SIZE = 100
TRAINING_SIZE = 80

TEST_IMG_W, TEST_IMG_H, TEST_PATCH_SIZE_W, TEST_PATCH_SIZE_H = 608, 608, 16, 16
THRESHOLD = 0.25

CUDA = True

def main():
    total = len(PATCH_SIZES) * len(AUGMENTATIONS) * len(UNET_CONFIGS)
    current = 0
    
    with open('./search_new.txt', 'w') as file:
        file.writelines('Starting time: {} \n\n'.format(datetime.datetime.now()))
    
        indices = np.arange(1, DATA_SIZE + 1)
        train_indices = indices[:TRAINING_SIZE]
        validation_indices = indices[TRAINING_SIZE:]
                
        for patch_size in PATCH_SIZES:
            for aug_idx, (aug, input_channels) in enumerate(AUGMENTATIONS):
                trainset = PatchedAerialDataset(TRAIN_IMAGE_DATA, TRAIN_LABEL_DATA, train_indices, patch_size, aug)
                validationset = PatchedAerialDataset(TRAIN_IMAGE_DATA, TRAIN_LABEL_DATA, validation_indices, patch_size, aug)

                trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
                validationloader = DataLoader(validationset, batch_size=1, shuffle=False)
                
                datasets = {'train':trainset, 'val':validationset}
                dataloaders = {'train':trainloader, 'val':validationloader}

                for unet_conf in UNET_CONFIGS:
                    nbr_layers = 2 if (unet_conf['CHANNELS_L4'] == -1) else 3
                    unet_conf['CHANNELS_L0'] = input_channels

                    model = UNet(nbr_layers, unet_conf).cuda(1)
                    criterion = torch.nn.BCEWithLogitsLoss().cuda(1)
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

                    acc, best_model = training(NUM_EPOCHS, model, criterion, optimizer, lr_scheduler, datasets, dataloaders, patch_size, cuda=CUDA)
                    
                    channels = '{}-{}-{}-{}-{}'.format(unet_conf['CHANNELS_L0'], unet_conf['CHANNELS_L1'], unet_conf['CHANNELS_L2'], unet_conf['CHANNELS_L3'], unet_conf['CHANNELS_L4'])
                    file.writelines('Model {}/{} \n'.format(current, total))
                    file.writelines('Params: {}, {}, {} \n'.format(patch_size, aug_idx, channels))
                    file.writelines('Accs: {} \n'.format(acc))
                    file.writelines('Best acc: {} \n'.format(np.max(acc)))
                    file.writelines('\n')
                    file.flush()
                    
                    # Save model
                    torch.save(model, './checkpoints/search|{}|{}|{}.pt'.format(patch_size, aug_idx, channels))
                    
                    current += 1
                
                
        file.writelines('Ending time: {} \n'.format(datetime.datetime.now()))

def img_augs():
    aug_config1 = ImageAugmentationConfig()
    aug_config1.rotation(range(10, 190, 10))
    
    aug_config2 = ImageAugmentationConfig()
    aug_config2.rotation(range(10, 190, 10))
    aug_config2.edge()
    
    aug_config3 = ImageAugmentationConfig()
    aug_config3.rotation(range(10, 190, 10))
    aug_config3.blur()
    
    aug_config4 = ImageAugmentationConfig()
    aug_config4.rotation(range(10, 190, 10))
    aug_config4.edge()
    aug_config4.blur()
    
    return [(aug_config1, 3), (aug_config2, 6), (aug_config3, 6), (aug_config4, 9)]

AUGMENTATIONS = img_augs()
PATCH_SIZES = [104, 120]
UNET_CONFIGS = [
    {
        'CHANNELS_L1': 8,
        'CHANNELS_L2': 16,
        'CHANNELS_L3': 32,
        'CHANNELS_L4': -1
    },
    
    {
        'CHANNELS_L1': 16,
        'CHANNELS_L2': 32,
        'CHANNELS_L3': 64,
        'CHANNELS_L4': -1
    },
    
    {
        'CHANNELS_L1': 8,
        'CHANNELS_L2': 16,
        'CHANNELS_L3': 32,
        'CHANNELS_L4': 64
    },
    
    {
        'CHANNELS_L1': 16,
        'CHANNELS_L2': 32,
        'CHANNELS_L3': 64,
        'CHANNELS_L4': 128
    },
]

if __name__ == '__main__':
    main()