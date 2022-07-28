from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras import backend as K
from keras import losses, Model
import keras

from . import BaseRNN

"""
The Bidirectional LSTM based model aims to improve on the LSTM model by 
usinga Bidirectional verison of the cell, the final selected model also 
has one more hidden layer to increase capacity (refer to final rnn models notebook)
"""
class BiDirLSTM(BaseRNN):
    

    def model_builder(self) -> Model:
        
        model = Sequential()

        #choosing how many sequential Bidirectional LSTM cells to have
        if self.num_cells == 1:
            model.add(layers.Bidirectional(layers.LSTM(self.num_units),
                                          batch_input_shape=(None, self.input_length, 1)))
        else:
            model.add(layers.Bidirectional(layers.LSTM(self.num_units,return_sequences=True),
                                          batch_input_shape=(None, self.input_length, 1)))
            
            #adding intermediary Bidirectional LSTM cells
            for _ in range(self.num_cells-2):
                model.add(layers.Bidirectional(layers.LSTM(self.num_units,return_sequences=True)))
                
            #adding last RNN cell
            model.add(layers.Bidirectional(layers.LSTM(self.num_units)))
            
        #adding a dropout layer if required
        if self.dropout:
            model.add(layers.Dropout(self.dropout))
            
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
