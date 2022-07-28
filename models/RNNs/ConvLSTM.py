from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras import backend as K
from keras import losses, Model
import keras

from . import BaseRNN

"""
The CNN + LSTM model aims to use convolutional layers to extract features 
from the time series and feed them to the dollowing sequential part of the model,
i.e the LSTM cell. Hidden state size of teh LSTM and convolutional layer sizes kept
equal for simplicity. Conv. layer and dense layer numbers are tunable.
"""
class ConvLSTM(BaseRNN):

    def __init__(self, num_conv : int, *args, **kwargs) -> None:
        self.num_conv = num_conv
        super().__init__(*args, **kwargs)
    

    def model_builder(self) -> Model:
        
        model = Sequential()
    
        # Adding the convolutional layers
        model.add(layers.Conv1D(filters=self.num_units, kernel_size=3, activation='relu', 
                        batch_input_shape=(None, self.input_length, 1),
                        padding="causal") )
        
        for _ in range(self.num_conv - 1): 
            model.add(layers.Conv1D(filters=self.num_units, kernel_size=3, 
                                    activation='relu', padding="causal") )
        
        if self.dropout:
            model.add(layers.Dropout(self.dropout))

        #Adding the LSTM layer
        model.add(layers.LSTM(self.num_units)) 
        
        #adding dense layers
        for _ in range(self.num_dense - 1): 
            model.add(layers.Dense(16, activation='relu'))
        
        #choosing activation and loss functions for binary or multiclass task
        if self.classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer=self.optimizer, 
                          metrics=['accuracy', keras.metrics.AUC()])
        else:
            model.add(layers.Dense(self.classes, activation='softmax'))
            model.compile(loss=losses.sparse_categorical_crossentropy, 
                          optimizer=self.optimizer, metrics=['accuracy'])
            
        K.set_value(model.optimizer.learning_rate, self.lr)
        
        return model
