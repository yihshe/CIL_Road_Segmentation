"""
Full aerial images dataset.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from preprocessing.prepare_images import prepare_test_images

class AerialDataset(Dataset):
    """
    Data set of arial images for a given path, indices and image augmentation.
    It is possible to use majority voting (allows to predict a pixel label by voting of the 4 main rotation predictions).
    """
    def __init__(self, images_path, indices, augmentation_config, majority_voting=False):
        # Load images
        self.images, self.nblist = prepare_test_images(images_path, indices, augmentation_config, majority_voting)

        # Transormation applied before getting an element
        self.images_transform = torch.from_numpy

    def __getitem__(self, index):
        image = np.transpose(self.images[index], (2, 0, 1))

        if self.images_transform != None:
            image = self.images_transform(image)

        return image.float()

    def __len__(self):
        return len(self.images)
