from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras import backend as K
from keras import losses, Model
import keras

from . import BaseRNN

"""
The basic LSTM model replaces the RNN cell in the vanilla model with an LSTM 
cell, all other aspects of the model remain the same.
"""
class VanillaLSTM(BaseRNN):
    
    def model_builder(self) -> Model:
        model = Sequential()
        
        #choosing how many sequential LSTM cells to have
        if self.num_cells == 1:
            model.add(layers.LSTM(self.num_units, batch_input_shape=(None, self.input_length, 1)))
        else:
            model.add(layers.LSTM(self.num_units, return_sequences=True,
                                  batch_input_shape=(None, self.input_length, 1)))
            
            #adding intermediary RNN cells
            for _ in range(self.num_cells-2):
                model.add(layers.LSTM(self.num_units, return_sequences=True))
                
            #adding last RNN cell
            model.add(layers.LSTM(self.num_units))
            
        #adding a self.dropout layer if required
        if self.dropout:
            model.add(layers.dropout(self.dropout))

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
