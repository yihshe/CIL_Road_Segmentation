#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Activation, MaxPooling2D, Dropout, BatchNormalization, Concatenate, Add
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam

from metrics import f1

def conv2d_block(inputs, n_filter, kernel_size=3, batchnorm=True, activation='relu'):
    # first layer
    res = Conv2D(n_filter, kernel_size=1, kernel_initializer="he_normal", padding="same")(inputs)
    x = Conv2D(n_filter, kernel_size=kernel_size, kernel_initializer="he_normal",
               padding="same")(inputs)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    # second layers
    x = Conv2D(n_filter, kernel_size=kernel_size, kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Add()([x, res])
    x = Activation(activation)(x)

    return x


def unet(pretrained_weights = None,
         input_size = (None,None,3),
         n_filter=16,
         activation='relu',
         dropout=True, dropout_rate=0.5,
         batchnorm=True,
         loss=binary_crossentropy,
         optimizer=Adam(lr=1e-4)):
    """Build a standard UNet model.
    
    Arguments:
        pretrained_weights {str} -- path of the pretrained weights (default: {None})
        input_size {tuple} -- size of input images (default: {(None,None,3)})
        n_filter {int} -- number of filter of the first layer (default: {16})
        activation {str} -- activation function to use (default: {'relu'})
        dropout {bool} -- whether to use dropout layer (default: {True})
        dropout_rate {float} -- dropout rate (default: {0.5})
        batchnorm {bool} -- whether to use batch normalization layer (default: {True})
        loss {keras.losses} -- loss function to use (default: {binary_crossentropy})
        optimizer {keras.optimizers} -- optimizer to use (default: {Adam(lr=1e-4)})
    
    Returns:
        keras.models -- UNet model
    """

  
    # 3
    inputs = Input(input_size)
    
    # down path
    # n_filter
    conv1 = conv2d_block(inputs, n_filter, kernel_size=3, batchnorm=batchnorm, activation=activation)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # n_filter*2
    conv2 = conv2d_block(pool1, n_filter*2, kernel_size=3, batchnorm=batchnorm, activation=activation)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # n_filter*4
    conv3 = conv2d_block(pool2, n_filter*4, kernel_size=3, batchnorm=batchnorm, activation=activation)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # n_filter*8
    conv4 = conv2d_block(pool3, n_filter*8, kernel_size=3, batchnorm=batchnorm, activation=activation)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # central path
    # n_filter*16
    conv5 = conv2d_block(pool4, n_filter*16, kernel_size=3, batchnorm=batchnorm, activation=activation)

    # up path
    # n_filter*8
    up6 = Conv2DTranspose(n_filter*8, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv5)
    merge6 = concatenate([conv4, up6], axis = 3)
    merge6 = Dropout(dropout_rate)(merge6) if dropout else merge6
    conv6 = conv2d_block(merge6, n_filter*8, kernel_size=3, batchnorm=False, activation=activation)
    
    # n_filter*4
    up7 = Conv2DTranspose(n_filter*4, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv6)
    merge7 = concatenate([conv3, up7], axis = 3)
    merge7 = Dropout(dropout_rate)(merge7) if dropout else merge7
    conv7 = conv2d_block(merge7, n_filter*4, kernel_size=3, batchnorm=False, activation=activation)
    
    # n_filter*2
    up8 = Conv2DTranspose(n_filter*2, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv7)
    merge8 = concatenate([conv2, up8], axis = 3)
    merge8 = Dropout(dropout_rate)(merge8) if dropout else merge8
    conv8 = conv2d_block(merge8, n_filter*2, kernel_size=3, batchnorm=False, activation=activation)
    
    # n_filter
    up9 = Conv2DTranspose(n_filter, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv8)
    merge9 = concatenate([conv1, up9], axis = 3)
    merge9 = Dropout(dropout_rate)(merge9) if dropout else merge9
    conv9 = conv2d_block(merge9, n_filter, kernel_size=3, batchnorm=False, activation=activation)
    
    # classifier
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = optimizer, loss = loss, metrics = [f1, 'accuracy'])
    
    if(pretrained_weights):
        model.load_weights(filepath=pretrained_weights)

    return model



def bottleneck(x, n_filter, depth=6, kernel_size=3, activation='relu'):
    """Bottle neck of UNet with dilated convolution."""
    dilated_layers = []
    for i in range(depth):
        x = Conv2D(n_filter, kernel_size, 
                    activation=activation, padding='same', dilation_rate=2**i)(x)
        dilated_layers.append(x)
    return add(dilated_layers)

def unet_dilated(pretrained_weights = None,
                 input_size = (None,None,3),
                 n_filter=16,
                 activation='relu',
                 dropout=True, dropout_rate=0.5,
                 batchnorm=True,
                 loss=binary_crossentropy,
                 optimizer=Adam(lr=1e-4)):
    """Build a standard UNet model with dilated convolution.
    
    Arguments:
        pretrained_weights {str} -- path of the pretrained weights (default: {None})
        input_size {tuple} -- size of input images (default: {(None,None,3)})
        n_filter {int} -- number of filter of the first layer (default: {16})
        activation {str} -- activation function to use (default: {'relu'})
        dropout {bool} -- whether to use dropout layer (default: {True})
        dropout_rate {float} -- dropout rate (default: {0.5})
        batchnorm {bool} -- whether to use batch normalization layer (default: {True})
        loss {keras.losses} -- loss function to use (default: {binary_crossentropy})
        optimizer {keras.optimizers} -- optimizer to use (default: {Adam(lr=1e-4)})
    
    Returns:
        keras.models -- UNet model with dilated conv bottlenect
    """
    # 3
    inputs = Input(input_size)
    
    # down path
    # n_filter
    conv1 = conv2d_block(inputs, n_filter, kernel_size=3, batchnorm=batchnorm, activation=activation)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # n_filter*2
    conv2 = conv2d_block(pool1, n_filter*2, kernel_size=3, batchnorm=batchnorm, activation=activation)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # n_filter*4
    conv3 = conv2d_block(pool2, n_filter*4, kernel_size=3, batchnorm=batchnorm, activation=activation)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # central path
    # n_filter*8
    dilated = bottleneck(pool3, n_filter*8, depth=6, kernel_size=3, activation=activation)

    # up path
    # n_filter*4
    up4 = Conv2DTranspose(n_filter*4, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(dilated)
    merge4 = concatenate([conv3, up4], axis = 3)
    merge4 = Dropout(dropout_rate)(merge4) if dropout else merge4
    conv4 = conv2d_block(merge4, n_filter*4, kernel_size=3, batchnorm=False, activation=activation)
    
    # n_filter*2
    up5 = Conv2DTranspose(n_filter*2, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv4)
    merge5 = concatenate([conv2, up5], axis = 3)
    merge5 = Dropout(dropout_rate)(merge5) if dropout else merge5
    conv5 = conv2d_block(merge5, n_filter*2, kernel_size=3, batchnorm=False, activation=activation)
    
    # n_filter
    up6 = Conv2DTranspose(n_filter, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv5)
    merge6 = concatenate([conv1, up6], axis = 3)
    merge6 = Dropout(dropout_rate)(merge6) if dropout else merge6
    conv6 = conv2d_block(merge6, n_filter, kernel_size=3, batchnorm=False, activation=activation)
    
    # classifier
    conv7 = Conv2D(1, 1, activation='sigmoid')(conv6)

    model = Model(inputs = inputs, outputs = conv7)

    model.compile(optimizer = optimizer, loss = loss, metrics = [f1, 'accuracy'])
    
    if(pretrained_weights):
        model.load_weights(filepath=pretrained_weights)

    return model


def UNetTransfer(base_model, skip_connection_names, encoder_last_layer_name, up_filters, pretrained_weights = None, loss=binary_crossentropy, optimizer=Adam(lr=1e-4)):

  inputs = Input((None, None, 3), name="input_image")
  
  # Define encoder
  encoder = base_model(input_tensor = inputs, include_top=False, weights=None) # do not load the weights
  # encoder = base_model(input_tensor=inputs, include_top=False, weights=None)
  encoder_output =  encoder.get_layer(encoder_last_layer_name).output
  encoder.trainable = True # set the architecture as trainable
  x = encoder_output

  # Define bottleneck, layers of bottleneck can be reduced or omitted to reduce trainable parameters
  x = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
  x = BatchNormalization()(x)
  x = Dropout(0.3)(x)
  x = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
  x = BatchNormalization()(x)

  # Define decoder
  for i in range(1, len(skip_connection_names)+1, 1):
    # Define dropout rate for layers
    if i==len(skip_connection_names):
      drop_rate = 0.1
    else:
      drop_rate = 0.2

    # Up-sampling using transpose convolution
    x = Conv2DTranspose(up_filters[-i], (2, 2), strides=(2, 2), padding='same')(x)
    x_skip = encoder.get_layer(skip_connection_names[-i]).output
    x = Concatenate()([x, x_skip])

    # Add two convolutional layers in the block
    x = Conv2D(up_filters[-i], (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x)
    # x = Activation("relu")(x)

    x = Conv2D(up_filters[-i], (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    # x = Dropout(drop_rate)(x)
    # x = Activation("relu")(x)

  # Define output layer (1x1 convolution)
  outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

  # return the model
  model = Model(inputs, outputs)
  model.compile(optimizer = optimizer, loss = loss, metrics = [f1, 'accuracy'])
  if(pretrained_weights):
    model.load_weights(filepath=pretrained_weights)
  return model