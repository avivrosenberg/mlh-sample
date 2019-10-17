import os
from pathlib import Path

import numpy as np
import pickle
from sklearn.base import ClassifierMixin, BaseEstimator


class MLHChallengeModel(BaseEstimator, ClassifierMixin):

    def __init__(self, foo=1, bar=2, **kw):
        # TODO:
        #  Initialize your model/parameters here.
        #  You can add arguments as necessary for configuring your model.
        #  You can use any existing sklean estimators, pytorch models,
        #  or just implement everything from scratch.
        self.w_ = None

    def fit(self, X, y, **kwargs):
        # TODO:
        #  Implement your training code here.

        # As a demo, our "trained" model will just have random weights...
        n, d = X.shape
        self.w_ = np.random.randn(d) * 0.001

        return self

    def predict(self, X, **kwargs):
        # TODO: Implement your inference code here

        # As a demo, we'll just use our random weights to calculate a
        # probability and use a fixed threshold for classification.
        z = np.dot(X, self.w_)
        z -= np.max(z)
        z_exp = np.exp(z)
        p = z_exp / np.sum(z_exp)
        y_pred = p > 0.01

        return y_pred

    def fit_predict(self, X, y, **kwargs):
        return self.fit(X, y).predict(X)

    def save_state(self, filepath):
        filepath = Path(filepath)  # in case it's a str
        os.makedirs(filepath.parent, exist_ok=True)
        if filepath.suffix.lower() != '.pkl':
            filepath = Path(f'{filepath}.pkl')

        # TODO:
        #  Save any trained parameters you need to load later for inference.
        #  If using pytorch, note that modules have (load_)state_dict
        #  methods that you can use.
        state_dict = {'w_': self.w_}

        with open(filepath, mode='wb') as f:
            pickle.dump(state_dict, f)

    def load_state(self, filepath):
        with open(filepath, mode='rb') as f:
            state_dict = pickle.load(f)

        # TODO:
        #  Load your saved state into the model.
        self.w_ = state_dict['w_']
