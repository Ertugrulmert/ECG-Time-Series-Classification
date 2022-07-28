
from scipy import stats
import numpy as np
from .PretrainedEnsamble import PretrainedEnsamble

class VotingEnsamble(PretrainedEnsamble):

    def _fit(self, X, Y):
        pass
    
    def predict(self, X):
        all_pred = np.stack([m.predict(X) for m in self.models], axis=-1)
        modes = stats.mode(all_pred, axis=1).mode[:, 0]
        return modes
