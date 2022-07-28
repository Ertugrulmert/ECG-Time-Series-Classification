from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras import backend as K
from keras import losses, Model
import keras

from . import BaseRNN

"""
The basic Vanilla RNN model for Task 1 incorporates a standard RNN cell with
variable hidden state size and a fully connected layer. Number of RNN cells,
hidden state size and number of fully connected layers can be changed.
"""
class VanillaRNN(BaseRNN):

    def model_builder(self) -> Model:
        
        model = Sequential()
        
        #choosing how many sequential RNN cells to have
        if self.num_cells == 1:
            model.add(layers.SimpleRNN(self.num_units, batch_input_shape=(None, self.input_length, 1)))
        else:
            model.add(layers.SimpleRNN(self.num_units, return_sequences=True,
                                      batch_input_shape=(None, self.input_length, 1)))
            #adding intermediary RNN cells
            for _ in range(self.num_cells-2):
                model.add(layers.SimpleRNN(self.num_units, return_sequences=True))
            #adding last RNN cell
            model.add(layers.SimpleRNN(self.num_units))
            
        #adding a dropout layer if required
        if self.dropout:
            model.add(layers.Dropout(self.dropout))
        
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
