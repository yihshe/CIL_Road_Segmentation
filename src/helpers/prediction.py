"""
Prediction helpers.
"""

import numpy as np
from torch.autograd import Variable

def predict(model, loader, cuda, gpu_idx):
    """
    Predict label given a model and dataloader.
    """
    predicted_labels = []
    model.eval()
    
    for data in loader:
        # Create patches tensor
        patches = data
        if cuda: patches = patches.cuda(gpu_idx)
        patches = Variable(patches)

        # Feed model with patches
        outputs = model(patches)
        predicted = np.rint(outputs.squeeze().data.cpu().numpy())
        predicted_labels.append(predicted)
        
    return predicted_labels