"""
Images loading helpers.
"""

import os
import re
import glob
import numpy as np
import matplotlib.image as mpimg

def extract_images(filepath, indices):
    """
    Load images with given indices from the filepath.
    Note that the classical file name convention of the project is used.
    """
    imgs = []
    print ('Loading {} aerial images...'.format(len(indices)), end='')

    # Load all images
    for i in indices:
        filename = filepath + 'satImage_{:03d}.png'.format(i)
        if os.path.isfile(filename):
            img = mpimg.imread(filename)
            imgs.append(img)
        else:
            print('File {} does not exists'.format(filename))

    print('done')
    return np.asarray(imgs)

def extract_test_images(filepath, indices):
    """
    load test images
    """
    imgs = []
    nbs = []
    print('Loading {} aerial images...'.format(len(indices)), end='')

    # Load all images
    filelist = glob.glob(filepath+'*.png')
    for filename in filelist:
        if os.path.isfile(filename):
            img = mpimg.imread(filename)
            imgs.append(img)
            img_number = int(re.search(r"\d+", filename).group(0))
            nbs.append(img_number)
        else:
            print('File {} does not exists'.format(filename))
    print('done')
    return np.asarray(imgs), nbs

