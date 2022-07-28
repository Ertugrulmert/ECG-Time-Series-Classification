from keras import losses, activations, Model
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten
from keras.models import Sequential
from tensorflow.keras import optimizers

from .BaseCNN import BaseCNN

"""
VanillaCNN structure: Repeating of convolutional and maxpooling layers with an increasing number of filters


:param classes: number of distinct classes for classification task
:param dropout: dropout rate after convolutional layers
:param optimizer: optimizer used for fitting the model
"""
class VanillaCNN(BaseCNN):


    def __init__(
            self,
            classes: int,
            dropout: float,
            optimizer: optimizers,
            *args, **kwargs
    ) -> None:

        self.dropout = dropout
        self.optimizer = optimizer
        super().__init__(classes, *args, **kwargs)

    def model_builder(self) -> Model:
        """
        returns a VanillaCNN Model
        :return: keras model
        """

        if self.classes == 1:
            final_activation = activations.sigmoid
            loss_function = losses.binary_crossentropy
        else:
            final_activation = activations.softmax
            loss_function = losses.sparse_categorical_crossentropy

        model = Sequential()
        model.add(Conv1D(input_shape=(187, 1), filters=8, kernel_size=3, strides=1, padding="valid",
                         activation=activations.relu))
        model.add(Dropout(rate=self.dropout))
        model.add(MaxPool1D(pool_size=2, strides=2, padding="valid"))
        model.add(Conv1D(16, kernel_size=3, strides=1, padding="valid", activation=activations.relu))
        model.add(Dropout(rate=self.dropout))
        model.add(MaxPool1D(pool_size=2, strides=2, padding="valid"))
        model.add(Conv1D(32, kernel_size=3, strides=1, padding="valid", activation=activations.relu))
        model.add(Dropout(rate=self.dropout))
        model.add(MaxPool1D(pool_size=2, strides=2, padding="valid"))
        model.add(Conv1D(64, kernel_size=3, strides=1, padding="valid", activation=activations.relu))
        model.add(Dropout(rate=self.dropout))
        model.add(MaxPool1D(pool_size=2, strides=2, padding="valid"))
        model.add(Conv1D(128, kernel_size=3, strides=1, padding="valid", activation=activations.relu))
        model.add(Dropout(rate=self.dropout))
        model.add(MaxPool1D(pool_size=2, strides=2, padding="valid"))
        model.add(Conv1D(256, kernel_size=3, strides=1, padding="valid", activation=activations.relu))
        model.add(Dropout(rate=self.dropout))
        model.add(Flatten())
        model.add(Dense(30, activation=activations.relu))
        model.add(Dense(self.classes, activation=final_activation))

        model.compile(loss=loss_function, optimizer=self.optimizer, metrics=['acc'])

        return model
