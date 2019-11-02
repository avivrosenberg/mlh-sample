import os
from pathlib import Path
import logging

import numpy as np
import pickle
from sklearn.base import ClassifierMixin, BaseEstimator

from sklearn.linear_model.logistic import LogisticRegression

logger = logging.getLogger(__name__)


class MLHChallengeModel(BaseEstimator, ClassifierMixin):
    """
    Model class.

    TODO:
     You can change whatever you want here and/or completely replace
     this class. If you do, just make sure the functions in run.py can still
     work with it.
    """

    def __init__(self, foo=1, bar=2, **kw):
        # TODO:
        #  Initialize your model/parameters here.
        #  You can add arguments as necessary for configuring your model.
        #  You can use and combine any existing sklean estimators, pytorch
        #  models, etc, or just implement everything from scratch.

        # As a demo, we'll use a simple logistic regression model
        self.lr_model = LogisticRegression(
            C=.5, penalty='l1', solver='liblinear'
        )

    def fit(self, X, y, **kwargs):
        # TODO:
        #  Implement your training code here.

        # As a demo, we just fit our logistic regression model
        # and print best the features with some fixed threshold
        self.lr_model.fit(X, y)

        feat_names = kwargs['feat_names']
        best_features = self.lr_model.coef_[0, :] > 0.1
        logger.info(f'best features: {list(feat_names[best_features])}')

        return self

    def predict_proba(self, X, **kwargs):
        # TODO: Implement probability calculation for positive class (1).
        y_proba = self.lr_model.predict_proba(X)
        return y_proba[:, 1]  # probability of class 1

    def predict(self, X, **kwargs):
        # TODO: Implement your inference code here. Return binary labels.
        y_pred = self.lr_model.predict(X)
        return y_pred

    def fit_predict(self, X, y, **kwargs):
        return self.fit(X, y, **kwargs).predict(X, **kwargs)

    def save_state(self, filepath):
        filepath = Path(filepath)  # in case it's a str
        os.makedirs(filepath.parent, exist_ok=True)
        if filepath.suffix.lower() != '.pkl':
            filepath = Path(f'{filepath}.pkl')

        # TODO:
        #  Save any trained parameters you need to load later for inference.
        #  If using pytorch, note that modules have (load_)state_dict
        #  methods that you can use.
        state_dict = {'lr_model': self.lr_model}

        with open(filepath, mode='wb') as f:
            pickle.dump(state_dict, f)

    def load_state(self, filepath):
        with open(filepath, mode='rb') as f:
            state_dict = pickle.load(f)

        # TODO:
        #  Load your saved state into the model.
        self.lr_model = state_dict['lr_model']
