"""
UNet model structure.
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 1
DROPOUT_P = 0.2

class UNet(nn.Module):
    """
    Define UNet structure, with down and up phases. 
    Can be configured with 2 or 3 layers, and different channel sizes. 
    One must be careful of useful compatible channel sizes with provided images.
    All output values are passed into the sigmoid fonction for a binary classification.
    """
    def __init__(self, nbr_layers, channels_sizes):
        super(UNet, self).__init__()
        
        # Unpack configuration
        self.nbr_layers = nbr_layers
        CHANNELS_L0 = channels_sizes['CHANNELS_L0']
        CHANNELS_L1 = channels_sizes['CHANNELS_L1']
        CHANNELS_L2 = channels_sizes['CHANNELS_L2']
        CHANNELS_L3 = channels_sizes['CHANNELS_L3']
        CHANNELS_L4 = channels_sizes['CHANNELS_L4']
        
        self.down1 = DownStep(CHANNELS_L0, CHANNELS_L1)
        self.down2 = DownStep(CHANNELS_L1, CHANNELS_L2, dropout=(self.nbr_layers==2))
        
        if nbr_layers == 2:
            self.center = UpStep(CHANNELS_L2, CHANNELS_L3, CHANNELS_L2)
        elif nbr_layers == 3:
            self.down3 = DownStep(CHANNELS_L2, CHANNELS_L3, dropout=True)
            self.center = UpStep(CHANNELS_L3, CHANNELS_L4, CHANNELS_L3)
            self.up3 = UpStep(CHANNELS_L4, CHANNELS_L3, CHANNELS_L2)
        else:
            raise ValueError('UNet accepts 2 or 3 layers.')
            
        self.up2 = UpStep(CHANNELS_L3, CHANNELS_L2, CHANNELS_L1)
        self.up1 = UpStep(CHANNELS_L2, CHANNELS_L1, CHANNELS_L0, mode='final')
        
    def forward(self, x):
        down1, bridge1 = self.down1(x)
        down2, bridge2 = self.down2(down1)
        
        if self.nbr_layers == 2:
            center = self.center(down2)
            up2 = self.up2(torch.cat([center, bridge2], 1))
        elif self.nbr_layers == 3:
            down3, bridge3 = self.down3(down2)
            center = self.center(down3)
            up3 = self.up3(torch.cat([center, bridge3], 1))        
            up2 = self.up2(torch.cat([up3, bridge2], 1))  
        else: pass
                   
        up1 = self.up1(torch.cat([up2, bridge1], 1))
        
        return nn.Sigmoid()(up1)

class DownStep(nn.Module):
    """
    Define a single UNet down step, using convolutions and maxpooling.
    Dropout can be enabled.
    """
    def __init__(self, in_channels, out_channels, dropout=False):
        super(DownStep, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=DROPOUT_P)

    def forward(self, x):
        x = self.batchnorm1(x)
        x = self.relu(self.conv1(x))
        
        x = self.batchnorm2(x)
        x = self.relu(self.conv2(x))
        
        to_concat = x.clone()
        
        if self.use_dropout:
            x = self.dropout(x)
            
        return self.maxpooling(x), to_concat

class UpStep(nn.Module):
    """
    Define a single UNet up step, using convolutions and maxpooling.
    """
    def __init__(self, in_channels, middle_channels, out_channels, mode='transpose'):
        super(UpStep, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.batchnorm2 = nn.BatchNorm2d(middle_channels)
        
        # Either upsample or produce final image
        self.final_step = False
        if mode == 'transpose':
            self.upconv = nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2)
        elif mode == 'final':
            self.final_step = True
            self.final = nn.Conv2d(middle_channels, NUM_CLASSES, kernel_size=1)
            
    def forward(self, x):
        x = self.batchnorm1(x)
        x = self.relu(self.conv1(x))
        
        x = self.batchnorm2(x)
        x = self.relu(self.conv2(x))
        
        if self.final_step:
            return self.final(x)
        else:
            return self.upconv(x)