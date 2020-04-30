"""
Labeling related helpers.
"""

import torch

LABELING_THRESHOLD = 128

class Relabel:
    """
    Transforms grayscale image label to binary.
    """
    def __call__(self, tensor):
        assert isinstance(tensor, torch.FloatTensor), 'tensor needs to be FloatTensor'
        tensor[tensor < LABELING_THRESHOLD] = 0
        tensor[tensor >= LABELING_THRESHOLD] = 1
        return tensor


class ToLabel:
    """
    Transforms numpy labels to pytorch tensors.
    """
    def __call__(self, image):
        return torch.from_numpy(image).float().unsqueeze(0)
