"""
Image rotation related helpers.
"""

import numpy as np
from imgaug import augmenters as iaa
from preprocessing.patch import patchify, patchify_overlap

def rotate_images(images, angles):
    """
    Rotates all the images by all the given angles.
    """
    result = []
    for im in images:
        for angle in angles:
            rotated = iaa.Affine(rotate=angle).augment_image(im)
            result.append(rotated)

    return result

def patchify_no_corner(img, label, patch_size, overlap, overlap_amount):
    """
    Patchify and remove invalid corners due to rotation for both image and label.
    """
    if overlap:
        patches = zip(patchify_overlap(img, patch_size, overlap_amount), patchify_overlap(label, patch_size, overlap_amount))
    else:
        patches = zip(patchify(img, patch_size), patchify(label, patch_size))
    
    valid_img, valid_label = [], []

    for patch_img, patch_label in patches:
        if not is_corner(patch_img):
            valid_img.append(patch_img)
            valid_label.append(patch_label)

    return valid_img, valid_label

def is_corner(patch):
    """
    True of the given patch is likely to be in a rotation corner.
    """
    lt = not np.any(patch[0][0])
    rt = not np.any(patch[0][-1])
    lb = not np.any(patch[-1][0])
    rb = not np.any(patch[-1][-1])

    return lt or rt or lb or rb
