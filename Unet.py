import numpy as np
from keras.models import Model
from keras.layers import BatchNormalization, Input, Conv2D, MaxPooling2D, UpSampling2D, Activation
from keras.layers.merge import Concatenate

def convblock(x, filters, batch_norm=True):
    """A convolutional block with a 2D convolution, 
    optional batch normalization and ReLU activation
    
    Parameters
    ----------
    
    x : Keras computation
        The input to the convolution
        
    filters : int
        The filters in the convolutional layer
        
    batch_norm : bool, optional
        If batch normalization should be applied
        
    Returns
    -------
    
    Keras computation
        The sequence of convolution, (batch norm) and ReLU.
    
    """
    x = Conv2D(filters, 
               (3, 3), 
               use_bias=False, 
               padding="same",
               strides=1,
               kernel_initializer='he_uniform')(x) #glorot_uniform
    
    if batch_norm:
        x = BatchNormalization()(x)
    
    x = Activation("relu")(x)
    
    return x

def Unet(filters=64, layers=4, activation='sigmoid', classes=1, input_shape=None):
    """
    Building a U-Net [1]_.
    
    
    Parameters
    ----------
    
    filters : int, optional
        The number of filters in the first layer.
        The subsequent layers have multiples of this filter number.
        Default is 64.
        
    layers : int, optional
        The number of encoding and decoding layers. Default is 4.
    
    activation : str, optional
        The activation function in the last layer. Default is sigmoid.
        
    classes : int, optional
        The number of classes in the last layer. Default is 1.
        
    input_shape : tuple, optional
        The input shape of the data. We train the network to have arbitraty
        input shapes, default is None. Otherwise, the tuple has to follow
        the following criterion: (X, Y, channels)
        
        
    Returns
    -------
    
    Keras Model
        A Keras Model containing the U-Net structure.
        
        
    References
    ----------
    
    [1] Ronneberger, O., Fischer, P., & Brox, T. (2015, October). 
    U-net: Convolutional networks for biomedical image segmentation. 
    In International Conference on Medical image computing and 
    computer-assisted intervention (pp. 234-241). Springer, Cham.
    
    """
    if input_shape is None:
        input_shape = (None, None, 1)
        
    model_input = Input(shape=input_shape)
    
    to_concat = []
    
    x = model_input
    
    # Encoding 
    for block in range(layers):
        x = convblock(x, filters*(2**block))
        x = convblock(x, filters*(2**block))
        to_concat.append(x)
        
        x = MaxPooling2D(pool_size=(2, 2))(x)   
        
    x = convblock(x, filters*(2**(block+1)))
        
    # Decoding
    for block, filter_factor in enumerate(np.arange(layers)[::-1]):
        x = convblock(x, filters*(2**filter_factor))
        
        x = UpSampling2D(size = (2,2))(x)
        x = Concatenate()([x, to_concat[::-1][block]])
        
        x = convblock(x, filters*(2**filter_factor))
        
    x = convblock(x, filters*(2**filter_factor))
    
    # Final output, 1x1 convolution 
    model_output = Conv2D(classes, 
                          (1, 1), 
                          use_bias=False, 
                          padding="same",
                          activation=activation,
                          strides=1,
                          kernel_initializer='glorot_uniform')(x)

    return Model(model_input, model_output)