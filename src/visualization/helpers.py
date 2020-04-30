"""
Image visualization / transformation helpers.
"""

import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy

PIXEL_DEPTH = 255
IMG_PATCH_SIZE = 16

def imshow_tensor(img):
    """
    Display an image given as tensor.
    """
    npimg = img.numpy()
    plt.imshow(numpy.transpose(npimg, (1, 2, 0)))
    plt.show()

def imshow_tensor_gt(img):
    """
    Display label image given as tensor.
    """
    npimg = img.numpy()
    plt.imshow(npimg[0])
    plt.show()

def generate_predictions(testing_size, test_image_size, test_patch_size, labels, path, numberlist):
    """
    Generate prediction image from labels.
    """
    STEP = int(len(labels) / testing_size)
    assert STEP == (test_image_size**2 / test_patch_size**2)


    for i in range(0, len(labels), STEP):
        labels_for_patch = labels[i: i+STEP]
        prediction = label_to_img(test_image_size, test_image_size, test_patch_size, test_patch_size, labels_for_patch)

        k = int((i+STEP)/STEP)
        img = Image.fromarray(to_rgb(prediction))
        img.save(path + "test_%d" % numberlist[k-1] + ".png")

def label_to_img(imgwidth, imgheight, w, h, labels):
    """
    Transform labels to image.
    """
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            array_labels[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return array_labels

def img_crop(im, w, h):
    """
    Extract patches from given image.
    """
    list_patches = []
    imgwidth, imgheight = im.shape[0], im.shape[1]

    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)

    return list_patches


def extract_patches(images, patch_size):
    """
    Extract all patches from the images.
    """
    patches = []
    for im in images:
        patches.extend(img_crop(im, patch_size, patch_size))

    return patches

def img_float_to_uint8(img):
    """
    Transform float image to uint image.
    """
    rimg = img - numpy.min(img)
    if numpy.max(rimg) > 1e-5:
        rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    else:
        rimg = numpy.zeros_like(rimg)
    return rimg

def to_rgb(gt_img):
    """
    Get RGB image from groundtruth.
    """
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    gt_img8 = img_float_to_uint8(gt_img)
    gt_img_3c[:,:,0] = gt_img8
    gt_img_3c[:,:,1] = gt_img8
    gt_img_3c[:,:,2] = gt_img8
    return gt_img_3c

def make_img_overlay(img, predicted_img):
    """
    Display prediction overlay on top of original image.
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def concatenate_images(img, gt_img):
    """
    Concatenate two images side by side.
    """
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def labels_to_patches(image, img_size, p_size, threshold):
    """
    Transform pixel-wise label image, to patch-wise label image.
    """
    array_labels = numpy.zeros([img_size, img_size])
    for i in range(0, img_size, p_size):
        for j in range(0, img_size, p_size):
            mean = numpy.mean(image[i : i+p_size, j : j+p_size])
            if mean > threshold: array_labels[i : i+p_size, j : j+p_size] = 1
            else: array_labels[i : i+p_size, j : j+p_size] = 0
    return array_labels
