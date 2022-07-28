from keras.models import Model
from keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from keras import losses, activations
import numpy as np

from .. import BaseClassifier

class LatentClassifier(BaseClassifier):

    def __init__(self, classes, width, depth,  encoder, input_size = (1024,), padded_size = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.set_params(
            classes=classes, width=width, depth=depth, 
            encoder=encoder, input_size=input_size, 
            padded_size=padded_size
        )

    def model_builder(self) -> Model:

        if self.classes == 1:
            final_activation = activations.sigmoid
            loss_function = losses.binary_crossentropy
        else:
            final_activation = activations.softmax
            loss_function = losses.sparse_categorical_crossentropy

        inputs = Input(self.input_size)
        drop_1 = Dropout(0.2)(inputs)
        dense_1 = Dense(128, activation=activations.relu, name="dense_1")(drop_1)
        dense_1 = Dense(128, activation=activations.relu, name="dense_1.5")(dense_1)
        drop_2 = Dropout(0.2)(dense_1)
        dense_2 = Dense(128, activation=activations.relu, name="dense_2")(drop_2)
        dense_3 = Dense(self.classes, activation=final_activation, name="dense_3")(dense_2)

        model = Model(inputs=inputs, outputs=dense_3)
        opt = Adam(1e-4)

        model.compile(optimizer=opt, loss=loss_function, metrics=['acc'])
        # model.summary()
        return model
    
    def initialize(self, X, y):
        
        if not self.encoder.initialized_:
            self.encoder.initialize(X, y)

        if not self.padded_size is None:
            X = self._add_padding(X)

        return super().initialize(self.encoder.encode(X), y)

    def fit(self, X, y):
        
        if not self.encoder.initialized_:
            self.encoder.initialize(X, y)
        
        if not self.padded_size is None:
            X = self._add_padding(X)

        return super().fit(self.encoder.encode(X), y)

    def predict(self, X):
        
        if not self.padded_size is None:
            X = self._add_padding(X)

        return super().predict(self.encoder.encode(X))

    def predict_proba(self, X):
        
        if not self.padded_size is None:
            X = self._add_padding(X)

        return super().predict_proba(self.encoder.encode(X))
    
    def _add_padding(self, X):
        return np.concatenate([X, np.zeros((X.shape[0], self.padded_size - X.shape[1], 1))], axis=1)
