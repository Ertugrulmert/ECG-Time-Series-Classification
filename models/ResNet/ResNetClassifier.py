from keras import losses, activations
from tensorflow.keras import optimizers
from keras.layers import Dense, Input, Conv1D, ReLU, Flatten
from keras import Model

from typing import List

from .residual_block import residual_block
from .. import BaseClassifier


"""
Base Class for ResNet Models

:param classes: number of classes
:param first_filters: number of filters for the first convolutional layer
:param res_layers: list of parameters for residual blocks
:param dense_layers: list of parameters for dense layers
:param optimizer: optimizer used for fitting the model
"""
class ResNetBase(BaseClassifier):

    def __init__(
            self,
            classes: int,
            first_filters: int,
            res_layers: List,
            dense_layers: List,
            optimizer: optimizers = optimizers.Adam(),
            *args, **kwargs
    ) -> None:

        self.classes = classes
        self.first_filters = first_filters
        self.res_layers = res_layers
        self.dense_layers = dense_layers
        self.optimizer = optimizer
        super().__init__(*args, **kwargs)

    def model_builder(self) -> Model:

        if self.classes == 1:
            final_activation = activations.sigmoid
            loss_function = losses.binary_crossentropy
        else:
            final_activation = activations.softmax
            loss_function = losses.sparse_categorical_crossentropy

        # Add first layers
        inp = Input(shape=(187, 1))
        X = Conv1D(filters=self.first_filters, kernel_size=3, strides=1, padding="same")(inp)
        X = ReLU()(X)

        # Add residual bolcks
        for res_l in self.res_layers:
            X = residual_block(X, **res_l)

        # Flatten layers before adding dense layers
        X = Flatten()(X)
        for dense_l in self.dense_layers:
            X = Dense(**dense_l)(X)

        # Number of outputs equals number of classes
        # Use softmax for self.classes > 1 and sigmoid for binary classification with one output
        output = Dense(self.classes, activation=final_activation)(X)

        # Compile and return model
        # Use binary or categorical crossentropy depending on number of outputs
        model = Model(inputs=inp, outputs=output)
        model.compile(loss=loss_function, optimizer=self.optimizer, metrics=["acc"])

        # model.summary()

        return model

"""
ResNet with 5 residual blocks and 16 filters in each layer and

:param classes: number of distinct classes for classification task
:param filters: filter size
:param optimizer: optimizer used for fitting the model
"""
class ResNetSmall(ResNetBase):


    def __init__(
            self,
            classes: int,
            filters: int = 16,
            optimizer: optimizers = optimizers.Adam(),
            *args, **kwargs
    ) -> Model:

        res_layers = [{"filters": filters} for _ in range(5)]
        dense_layers = [{"units": units, "activation": "relu"} for units in [48, 32]]
        super().__init__(classes, filters, res_layers, dense_layers, optimizer, *args, **kwargs)


"""
ResNet with 5 residual blocks and 32 filters in each layer and

:param classes: number of distinct classes for classification task
:param filters: filter size
:param optimizer: optimizer used for fitting the model
"""
class ResNetStandard(ResNetBase):

    def __init__(
            self,
            classes: int,
            dropout: float,
            filters: int = 32,
            optimizer: optimizers = optimizers.Adam(),
            *args, **kwargs
    ) -> Model:
        res_layers = [{"filters": filters, "dropout": dropout} for _ in range(5)]
        dense_layers = [{"units": units, "activation": "relu"} for units in [48, 32]]
        super().__init__(classes, filters, res_layers, dense_layers, optimizer, *args, **kwargs)

"""
ResNet with Downsampling Structure (7 residual blocks with an increasing number of filters)
:param classes: number of distinct classes for classification task
:param optimizer: optimizer used for fitting the model
"""
class ResNetDS(ResNetBase):


    def __init__(
            self,
            classes: int,
            optimizer: optimizers = optimizers.Adam(),
            *args, **kwargs
    ) -> Model:


        res_layers = [{"filters": f, "downsample": d} for f, d in
                      [(32, False), (32, False), (64, True), (64, False), (64, False), (128, True), (128, False)]]
        dense_layers = [{"units": units, "activation": "relu"} for units in [64, 32]]
        super().__init__(classes, 32, res_layers, dense_layers, optimizer, *args, **kwargs)
