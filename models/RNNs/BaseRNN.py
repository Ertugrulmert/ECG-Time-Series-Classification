from .. import BaseClassifier

class BaseRNN(BaseClassifier):

    def __init__(self, 
        input_length, 
        num_units, 
        classes=2,  
        num_cells = 1, 
        num_dense = 1,
        dropout=0, 
        optimizer="adam", 
        lr=0.0001,
        *args, **kwargs
    ) -> None:
        self.input_length = input_length
        self.num_units = num_units
        self.classes = classes
        self.num_cells = num_cells
        self.num_dense = num_dense
        self.dropout = dropout
        self.optimizer = optimizer
        self.lr = lr
        super().__init__(*args, **kwargs)
