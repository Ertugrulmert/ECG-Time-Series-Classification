from tensorflow import Tensor
from keras.layers import Dropout, Conv1D, ReLU, add, MaxPool1D, BatchNormalization
from typing import Optional


def residual_block(
    X : Tensor, 
    filters : int, 
    kernel_size : int = 3, 
    stride : int = 1, 
    downsample : bool = False, 
    dropout : float = 0.1, 
    padding : str = "same", 
    pool_size : int = 2, 
    maxpool_strides : Optional[int] = None
) -> Tensor:

    """
    This function returns a 1-D residual block for a ResNet model
    :param X: Tensor of previous layer
    :param filters: number of filters
    :param kernel_size: kernel size
    :param stride: stride for convolutional layers
    :param downsample: boolean for downsampling
    :param dropout: dropout rate
    :param padding: padding for convolutional layers
    :param pool_size: pool size for MaxPool
    :param maxpool_strides: strides for MaxPool
    :return:
    """

    out = Conv1D(filters=filters,kernel_size=kernel_size, strides=stride, padding=padding)(X)
    out = Dropout(dropout)(out)
    out = ReLU()(out)
    out = BatchNormalization()(out)
    out = Conv1D(filters=filters,kernel_size=kernel_size, strides=stride, padding=padding)(out)
    out = Dropout(dropout)(out)

    if downsample:
        X = Conv1D(kernel_size=kernel_size, strides=1, filters=filters, padding="same")(X)

    out = add([X, out])
    out = ReLU()(out)
    out = BatchNormalization()(out)
    out = MaxPool1D(pool_size, maxpool_strides)(out)
    return out
