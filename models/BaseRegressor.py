from scikeras.wrappers import KerasRegressor
from keras.models import Model
from abc import ABC, abstractmethod
from pathlib import Path

class BaseRegressor(KerasRegressor, ABC):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(model = lambda: self.model_builder(), *args, **kwargs)
        self.set_params(**kwargs)

    def load_weights(self, path : Path):
        self.model_.load_weights(path)

    def set_params(self, **kwargs):
        
        if hasattr(self, "kwargs"):
            self.kwargs.update(kwargs)
        
        else:
            self.kwargs = kwargs

        for key, value in kwargs.items():
            setattr(self, key, value)
        
        return self

    def get_params(self, deep=False):
        return self.kwargs

    @abstractmethod
    def model_builder(self) -> Model:
        pass
