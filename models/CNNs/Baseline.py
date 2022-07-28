from keras import losses, activations, Model
from tensorflow.keras import optimizers
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D

from .BaseCNN import BaseCNN


"""
Keras baseline model, adapted from https://github.com/CVxTz/ECG_Heartbeat_Classification/blob/master/code/baseline_mitbih.py
                                   https://github.com/CVxTz/ECG_Heartbeat_Classification/blob/master/code/baseline_ptbdb.py

:param nclass: number of distinct classes for classification task
:return: keras model
"""
class Baseline(BaseCNN):

    def model_builder(self) -> Model:

        
        if self.classes == 1:
            final_activation = activations.sigmoid
            loss_function = losses.binary_crossentropy
        else:
            final_activation = activations.softmax
            loss_function = losses.sparse_categorical_crossentropy

        inp = Input(shape=(187, 1))
        img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
        img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = GlobalMaxPool1D()(img_1)
        img_1 = Dropout(rate=0.2)(img_1)

        dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
        dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
        dense_1 = Dense(self.classes, activation=final_activation, name="dense_3_ptbdb")(dense_1)

        model = Model(inputs=inp, outputs=dense_1)
        opt = optimizers.Adam(0.001)
        model.compile(optimizer=opt, loss=loss_function, metrics=['acc'])

        return model
