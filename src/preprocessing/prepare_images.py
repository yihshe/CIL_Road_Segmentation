"""
Main functions responsible for loading and transforming images.
"""

from imgaug import augmenters as iaa

from preprocessing.patch import *
from preprocessing.loading import *
from preprocessing.rotation import *

def prepare_test_images(images_path, indices, aug_config, majority_voting=False):
    """
    Load and augment images for testing.
    Note that only channels augmentations are performed.
    """
    # Load images
    #NOTE: needed np.arrays to append that stuff easily
    images, img_nblist = extract_test_images(images_path, indices)
    images = np.array(images)

    if majority_voting:
        #This has been chosen so that the structure of images is
        #50 images no rotate | 50 images rotate 90 | 50 images rotate 180 | 50 images rotate 270
        #instead of
        #img0 | img0 rotate 90 | img0 rotate 180 | img0 rotate 270 | img1 | img1 rotate 90 | ...
        #This is easier then to rotate back the images afters model eval
        r1 = np.array(rotate_images(images, [90]))
        r2 = np.array(rotate_images(images, [180]))
        r3 = np.array(rotate_images(images, [270]))
        images = np.concatenate((images, r1, r2, r3),axis=0)

    return images, img_nblist

def prepare_train_patches(images_path, labels_path, indices, patch_size, overlap, overlap_amount, aug_config):
    """
    Load, patchify and augment images and labels for training.
    """

    # Load images and labels
    images = extract_images(images_path, indices)
    labels = extract_images(labels_path, indices)

    # Get patches
    if overlap:
        image_patches = [patch for im in images for patch in patchify_overlap(im, patch_size, overlap_amount)]
        label_patches = [patch for label in labels for patch in patchify_overlap(label, patch_size, overlap_amount)]
    else:
        image_patches = [patch for im in images for patch in patchify(im, patch_size)]
        label_patches = [patch for label in labels for patch in patchify(label, patch_size)]

    if not aug_config:
        return image_patches, label_patches

    patches = zip(image_patches, label_patches)

    # Rotation needs to be applied on whole image
    if aug_config.do_rotation:
        images_rot = rotate_images(images, aug_config.rotation_angles)
        labels_rot = rotate_images(labels, aug_config.rotation_angles)

        for im, label in zip(images_rot, labels_rot):
            p = patchify_no_corner(im, label, patch_size, overlap, overlap_amount)
            image_patches.extend(p[0])
            label_patches.extend(p[1])

    # Flip each patch horizontally
    images_flipped = []
    labels_flipped = []
    if aug_config.do_flip:
        flip_hor = iaa.Fliplr(0.5).to_deterministic()
        flip_ver = iaa.Flipud(0.5).to_deterministic()
        images_flipped.extend(flip_hor.augment_images(image_patches))
        images_flipped.extend(flip_ver.augment_images(image_patches))
        labels_flipped.extend(flip_hor.augment_images(label_patches))
        labels_flipped.extend(flip_ver.augment_images(label_patches))

    image_patches.extend([im.copy() for im in images_flipped])
    label_patches.extend([im.copy() for im in labels_flipped])

    return image_patches, label_patches
