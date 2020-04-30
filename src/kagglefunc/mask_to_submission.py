"""
Helpers functions for Kaggle submission generation.
"""
import os
import numpy as np
import matplotlib.image as mpimg
import re
import time

# Percentage of pixels > 1 required to assign a foreground label to a patch
foreground_threshold = 0.25

def mask_to_submission_strings(image_filename):
    """
    Reads a single image and outputs the strings that should go into the submission file.
    """
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """
    Converts images into a submission file.
    """
    with open(submission_filename, 'w+') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


def generate_submission_csv(testing_size, submission_path, predictions_path, numberlist):
    """
    Generate CSV submission from predicted images.
    """
    today = time.strftime("%d-%m-%Y")
    submission_filename = submission_path + 'submission_' + today +'.csv'
    image_filenames = []
    for i in range(testing_size):
        image_filename = predictions_path + 'test_' + '%d' % numberlist[i] + '.png'
        print(image_filename)
        image_filenames.append(image_filename)
        masks_to_submission(submission_filename, *image_filenames)

def patch_to_label(patch):
    """
    Assign label to a patch.
    """
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0
