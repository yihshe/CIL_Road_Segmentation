"""
Majority voting main function.
"""

from preprocessing.rotation import *

def majority_voting(predicted_labels, testing_size):
    """
    Perform majority voting on predicted_labels.
    predicted_labels should be of the following form: 
        [No rotation | Rotation 90° | Rotation 180° | Rotation 270° ]
    where each section contains testing_size images.
    """
    n = len(predicted_labels)
    width, height = predicted_labels[0].shape
    
    # Rotate back images
    ROTATIONS_N = 4
    r1_back = rotate_images(predicted_labels[testing_size : 2*testing_size], [-90])
    r2_back = rotate_images(predicted_labels[2*testing_size : 3*testing_size], [-180])
    r3_back = rotate_images(predicted_labels[3*testing_size : 4*testing_size], [-270])
    
    # Concatenate them
    predicted_labels_rotated_back = np.concatenate((predicted_labels[: testing_size], r1_back, r2_back, r3_back),axis=0)
    
    # Perform actual voting
    predicted_labels_majority = np.full_like(predicted_labels[: testing_size], -1)
    for k in range(testing_size):
        print(str(k) + ',', end='')
        
        # Get the 'same' 4 images (idx 0, 50, 100, 150; then 1, 51, 101, 151, ...)
        rge = [k + testing_size*idx for idx in range(ROTATIONS_N)]
        imgs = predicted_labels_rotated_back[rge]
        
        # For each pixel, do majority
        for i in range(width):
            for j in range(height):
                # [0.0, 1.0, 1.0, 1.0]
                l = [imgs[idx][i][j] for idx in range(ROTATIONS_N)]
                # [1, 3]
                counts = np.bincount(l)
                # 1.0
                predicted_labels_majority[k][i][j] = np.argmax(counts)
                
    return predicted_labels_majority