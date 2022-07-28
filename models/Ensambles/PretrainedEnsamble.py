from models import BaseClassifier
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, RMSprop

class PretrainedEnsamble(ABC):

    def __init__(self, pretrained_models):
        self.models = [m[0] for m in pretrained_models]
        self.paths = [m[1] for m in pretrained_models]
    
    def fit(self, X, Y):
        
        for i in range(len(self.models)):
            self.models[i].initialize(X, Y)
            self.models[i].load_weights(self.paths[i])

        self._fit(X, Y)

    @abstractmethod
    def _fit(self, X, Y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
