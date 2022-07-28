import numpy as np
from . PretrainedEnsamble import PretrainedEnsamble

class MeanEnsamble(PretrainedEnsamble):

    def _fit(self, X, Y):
        pass
    
    def predict(self, X):
        all_probs = np.stack([m.predict_proba(X) for m in self.models], axis=-1)
        mean_probs = np.mean(all_probs, axis=-1)
        return np.argmax(mean_probs, axis=-1)
