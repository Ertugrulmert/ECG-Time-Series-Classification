import numpy as np
from . PretrainedEnsamble import PretrainedEnsamble
from sklearn.linear_model import LogisticRegression

class LogRegEnsamble(PretrainedEnsamble):

    def __init__(self, pretrained_models, **kwargs):
        self.logreg = LogisticRegression(**kwargs)
        super().__init__(pretrained_models)

    def _fit(self, X, Y):
        all_probs = np.concatenate([m.predict_proba(X) for m in self.models], axis=1)
        return self.logreg.fit(all_probs, Y)
    
    def predict(self, X):
        all_probs = np.concatenate([m.predict_proba(X) for m in self.models], axis=1)
        return self.logreg.predict(all_probs)
