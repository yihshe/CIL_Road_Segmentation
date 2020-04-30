"""
Train the final model and generate predictions and Kaggle submission.
"""

import argparse
import numpy as np

import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from models.simpleu import UNet
from helpers.training import training
from helpers.prediction import predict
from datasets.aerial_dataset import AerialDataset
from datasets.patched_aerial_dataset import PatchedAerialDataset
from preprocessing.augmentation_config import ImageAugmentationConfig
from visualization.helpers import labels_to_patches, extract_patches, generate_predictions
from kagglefunc.mask_to_submission import generate_submission_csv

SUBMISSION_PATH, CHECKPOINT_PATH = '../submissions/', '../checkpoints/'
TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH = '../data/training/images/', '../data/training/groundtruth/'
TEST_IMAGE_PATH, TEST_LABEL_PATH = '../data/test/images/', '../data/test/predictions/'
DATA_SIZE = 100 # 100
TRAINING_SIZE, TESTING_SIZE = 90, 94 # 80, 50
PERM = True
BATCH_SIZE, NUM_EPOCHS = 16, 70
PATCH_SIZE, OVERLAP, OVERLAP_AMOUNT = 256, True, 64
TEST_IMG_SIZE, TEST_PATCH_SIZE = 608, 16
ROAD_THRESHOLD = 0.25

def run(cuda, gpu_idx, train):
    """
    Main run function.
    """
    # Get augmentation configuration
    aug_config = augmentation_config()

    # Create Unet, criterion and optimzer, use cuda if requested
    if cuda:
        model = UNet(*unet_config()).cuda(gpu_idx)
    else:
        model = UNet(*unet_config())

    # Train or load
    if train:
        model = train_model(model, cuda, gpu_idx, aug_config)
    else:
        model = load_model(model)

    # Load testing data
    test_indices = np.arange(1, TESTING_SIZE + 1)
    testset = AerialDataset(TEST_IMAGE_PATH, test_indices, aug_config, majority_voting=False)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Predict labels
    predicted_labels = predict(model, testloader, cuda, gpu_idx)

    # Transform pixel-wise prediction to patchwise
    patched_images = [labels_to_patches(labels, TEST_IMG_SIZE, TEST_PATCH_SIZE, ROAD_THRESHOLD) for labels in predicted_labels]

    # Extract each patch
    img_patches_submit = extract_patches(patched_images, TEST_PATCH_SIZE)

    # Generate submission
    generate_predictions(TESTING_SIZE, TEST_IMG_SIZE, TEST_PATCH_SIZE, img_patches_submit, TEST_LABEL_PATH)
    generate_submission_csv(SUBMISSION_PATH, TEST_LABEL_PATH)

    print('Done')
    print('Predictions generated in: {}'.format(TEST_LABEL_PATH))
    print('CSV submission generated in: {}'.format(SUBMISSION_PATH))

def train_model(model, cuda, gpu_idx, aug_config):
    """
    Train model from scratch.
    """
    # Create dataset and dataloader
    if PERM:
        indices = np.random.permutation(np.arange(1, DATA_SIZE + 1))
    else:
        indices = np.arange(1, DATA_SIZE + 1)
    train_indices = indices[:TRAINING_SIZE]
    validation_indices = indices[TRAINING_SIZE:]
    trainset = PatchedAerialDataset(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, train_indices, PATCH_SIZE, OVERLAP, OVERLAP_AMOUNT, aug_config)
    validationset = PatchedAerialDataset(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, validation_indices, PATCH_SIZE, OVERLAP, OVERLAP_AMOUNT, aug_config)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    validationloader = DataLoader(validationset, batch_size=1, shuffle=False)

    # Create criterion, optimizer and scheduler
    if cuda:
        criterion = torch.nn.BCEWithLogitsLoss().cuda(gpu_idx)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    # Train model
    datasets, dataloaders = {'train': trainset, 'val':validationset}, {'train': trainloader, 'val':validationloader}
    training(NUM_EPOCHS, model, criterion, optimizer, lr_scheduler, datasets, dataloaders, PATCH_SIZE, validate=True, cuda=cuda, gpu_idx=gpu_idx)

    return model

def load_model(model):
    """
    Load pre-trained model.
    """
    print('Loading pre-trained model...')
    model.load_state_dict(torch.load(CHECKPOINT_PATH + 'best_model.pt').state_dict())
    return model

def unet_config():
    """
    Generate the Unet channel configuration.
    """
    nbr_layers = 3
    channel_sizes = {
        'CHANNELS_L0': 3,
        'CHANNELS_L1': 16,
        'CHANNELS_L2': 32,
        'CHANNELS_L3': 64,
        'CHANNELS_L4': 128
    }

    return (nbr_layers, channel_sizes)

def augmentation_config():
    """
    Generate the image augmentation configuration.
    """
    aug_config = ImageAugmentationConfig()
    aug_config.rotation([45, 90, 135, 180, 225, 270, 315])
    aug_config.flip()
    return aug_config


def parse_args():
    """
    Parse command line flags.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', action='store_true', default=False, dest='cuda', help='Run with Cuda enabled')
    parser.add_argument('-gpu2', action='store_true', default=False, dest='gpu2', help='Use second GPU instead of first one')
    parser.add_argument('-train', action='store_true', default=False, dest='train', help='Train model from scratch')
    results = parser.parse_args()

    return {'cuda': results.cuda and torch.cuda.is_available(), 'train': results.train, 'gpu2': results.gpu2}

if __name__ == '__main__':
    args = parse_args()
    gpu_idx = 1 if args['gpu2'] else 0
    run(args['cuda'], gpu_idx, args['train'])
