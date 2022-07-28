import numpy as np
from copy import deepcopy

from tensorflow.keras.optimizers import Adam
from keras import losses, activations, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, GlobalMaxPool1D
from sklearn.metrics import accuracy_score
from torch import classes

from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import AdaBoostClassifier

def get_model(classes, n_filters = 16, n_dense = 16, n_dense_layers = 1, kernel_size=5):

    if classes == 1:
        final_activation = activations.sigmoid
        loss_function = losses.binary_crossentropy
    else:
        final_activation = activations.softmax
        loss_function = losses.sparse_categorical_crossentropy

    inp = Input(shape=(187, 1))
    img_1 = Convolution1D(n_filters, kernel_size=kernel_size, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(n_filters, kernel_size=kernel_size, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    dense = Dropout(rate=0.2)(img_1)

    for _ in range(n_dense_layers):
        dense = Dense(n_dense, activation=activations.relu)(dense)
    dense = Dense(classes, activation=final_activation)(dense)

    model = models.Model(inputs=inp, outputs=dense)
    opt = Adam(0.001)

    model.compile(optimizer=opt, loss = loss_function, metrics=["acc"])
    
    return model

class BoostingCNN: 

    def __init__(self, **kwargs):
        self.set_params(**kwargs)  
      
    def set_params(self, n_estimators, **kwargs):

        self.classes = classes
        self.args = deepcopy(kwargs)
        self.args["n_estimators"] = n_estimators

        early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=0)
        redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=0)
        callbacks_list = [early, redonplat]
        
        ann_estimator = KerasClassifier(
              model=lambda: get_model(**kwargs), 
              epochs=100, verbose=0, 
              callbacks=callbacks_list, validation_split=0.1
        )

        self.model = AdaBoostClassifier(base_estimator=ann_estimator, n_estimators=n_estimators)

        return self

    def fit(self, X_train, Y_train):
        
        # Helper for ensuring that all classes are present
        X_train = np.concatenate([X_train, np.zeros((self.args["classes"] , *X_train.shape[1:]), dtype=X_train.dtype)])
        Y_train = np.concatenate([Y_train, np.arange(0, self.args["classes"] , dtype=Y_train.dtype)])

        self.model.fit(X_train, Y_train)
        return self

    def predict(self, X_pred):
        return self.model.predict(X_pred)

    def score(self, X_val, Y_val):
        pred_test = self.model.predict(X_val)
        return accuracy_score(Y_val, pred_test)

    def get_params(self, deep=False):
        return self.args
