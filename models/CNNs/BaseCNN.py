from .. import BaseClassifier
from tensorflow.keras import optimizers

class BaseCNN(BaseClassifier):

    def __init__(
        self,
        classes: int, 
        *args, **kwargs
    ) -> None:
        self.classes = classes
        super().__init__(*args, **kwargs)
