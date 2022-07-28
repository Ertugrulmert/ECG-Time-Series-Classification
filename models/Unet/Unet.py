from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dropout, Convolution1D, MaxPool1D, UpSampling1D, concatenate, GlobalMaxPool1D

from .. import BaseRegressor

class Unet(BaseRegressor):

    def __init__(self, input_size, loss, *args, **kwargs) -> None:
        self.set_params(input_size = input_size, loss = loss, **kwargs)
        super().__init__(*args, **kwargs)

    def model_builder(self) -> Model:
        
        inputs = Input(self.input_size)
        conv1 = Convolution1D(64, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(inputs)
        conv1 = Convolution1D(64, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPool1D(pool_size=2)(conv1)
        conv2 = Convolution1D(128, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(pool1)
        conv2 = Convolution1D(128, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPool1D(pool_size=2)(conv2)
        conv3 = Convolution1D(256, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(pool2)
        conv3 = Convolution1D(256, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPool1D(pool_size=2)(conv3)
        conv4 = Convolution1D(512, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(pool3)
        conv4 = Convolution1D(512, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPool1D(pool_size=2)(drop4)

        conv5 = Convolution1D(1024, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(pool4)
        conv5 = Convolution1D(1024, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Convolution1D(512, 2, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(drop5))
        merge6 = concatenate([drop4,up6], axis = 2)
        conv6 = Convolution1D(512, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(merge6)
        conv6 = Convolution1D(512, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(conv6)

        up7 = Convolution1D(256, 2, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv6))
        merge7 = concatenate([conv3,up7], axis = 2)
        conv7 = Convolution1D(256, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(merge7)
        conv7 = Convolution1D(256, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(conv7)

        up8 = Convolution1D(128, 2, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv7))
        merge8 = concatenate([conv2,up8], axis = 2)
        conv8 = Convolution1D(128, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(merge8)
        conv8 = Convolution1D(128, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(conv8)

        up9 = Convolution1D(64, 2, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv8))
        merge9 = concatenate([conv1,up9], axis = 2)
        conv9 = Convolution1D(64, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(merge9)
        conv9 = Convolution1D(64, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(conv9)
        conv9 = Convolution1D(2, 3, activation = 'relu', padding = "same", kernel_initializer = 'he_normal')(conv9)
        conv10 = Convolution1D(1, 1, activation = 'linear')(conv9)

        model = Model(inputs = inputs, outputs = conv10)

        model.compile(
            optimizer = Adam(learning_rate = 1e-4), 
            loss = self.loss
        )

        # Encoder for extracting 
        pooled_out = GlobalMaxPool1D()(drop5)
        self.encoder_ = Model(inputs = inputs, outputs = pooled_out)

        return model

    def encode(self, X):
        return self.encoder_.predict(X)

    def initialize(self, X, y):
        super().initialize(X, y.reshape((y.shape[0], -1)))

    def fit(self, X, y):
        self.initialize(X, y)
        self.model_.fit(X, y, *self.args, **self.kwargs)
