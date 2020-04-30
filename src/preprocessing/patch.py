"""
Patch extraction helpers.
"""

import numpy as np
from skimage.util import view_as_windows

def patchify_overlap(im, patch_len, overlap_amount):
    """
    Extract overlapping patches from image.
    """
    is_2D = len(im.shape) == 2
    
    if is_2D:
        window_shape = (patch_len, patch_len)
    else:
        window_shape = (patch_len, patch_len, 3)
        
    overlapped_windows = view_as_windows(im, window_shape, overlap_amount)
    
    patches = []
    for i in range(0, overlapped_windows.shape[0], 1):
        for j in range(0, overlapped_windows.shape[1], 1):
            if is_2D:
                patches.append(overlapped_windows[i][j])
            else:
                patches.append(overlapped_windows[i][j][0])
    
    return patches

def patchify(im, patch_len):
    """
    Extract non-overlapping patches from image.
    If patch_len is not multiple of the image size, image is mirrored.
    """
    patches = []
    width, height = im.shape[0], im.shape[1]
    is_2D = len(im.shape) == 2

    # Mirror image for a length of one patch to make sure we get the whole image in patches
    im = mirror(im, patch_len)

    # Build the right number of patches out of the mirrored image
    for i in range(0, width, patch_len):
        for j in range(0, height, patch_len):
            if is_2D:
                patch = im[j:j+patch_len, i:i+patch_len]
            else:
                patch = im[j:j+patch_len, i:i+patch_len, :]

            patches.append(patch)
    return patches


def mirror(im, length):
    """
    Mirror an image on the right on length pixels
    """
    width, height = im.shape[0], im.shape[1]
    is_2D = len(im.shape) == 2

    if is_2D:
        right_flipped = np.fliplr(im[:, width - length:])
    else:
        right_flipped = np.fliplr(im[:, width - length:, :])

    right_mirrored = np.concatenate((im, right_flipped), axis=1)

    if is_2D:
        bottom_flipped = np.flipud(right_mirrored[height - length:, :])
    else:
        bottom_flipped = np.flipud(right_mirrored[height - length:, :, :])

    mirrored = np.concatenate((right_mirrored, bottom_flipped), axis=0)
    return mirrored
