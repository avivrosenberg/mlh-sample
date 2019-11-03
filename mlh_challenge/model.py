import os
from pathlib import Path
import logging

import numpy as np
import pickle
from sklearn.base import ClassifierMixin, BaseEstimator

from sklearn.linear_model.logistic import LogisticRegression
from sklearn.utils.validation import check_is_fitted

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
        pass

    def fit(self, X, y, **kwargs):
        """
        Fits the model parameters to the data.
        :param X: 2d tensor of data samples (N,D)
        :param y: 1d tensor of binary targets labels (N,)
        :param kwargs: any extra args that you need.
        :return: self.
        """

        # TODO:
        #  Implement your training code here.

        # As a demo, we'll fit a simple logistic regression model
        # and print best the features with some fixed threshold
        self.lr_ = LogisticRegression(
            C=.5, penalty='l1', solver='liblinear'
        )
        self.lr_.fit(X, y)

        feat_names = kwargs['feat_names']
        best_features = self.lr_.coef_[0, :] > 0.1
        logger.info(f'best features: {list(feat_names[best_features])}')

        return self

    def predict_proba(self, X, **kwargs):
        """
        Calculates a probability estimate for a positive prediction (1).
        :param X: 2d tensor of data samples (N,D)
        :param kwargs: any extra args that you need.
        :return: 1d tensor of probabilities (N,)
        """
        # TODO: Implement probability calculation for positive class (1).
        check_is_fitted(self, ['lr_'])
        y_proba = self.lr_.predict_proba(X)
        return y_proba[:, 1]  # probability of class 1

    def predict(self, X, **kwargs):
        """
        Predicts the class label for the given samples.
        :param X: 2d tensor of data samples (N,D)
        :param kwargs: any extra args that you need.
        :return: 1d tensor of binary class labels (N,)
        """
        # TODO: Implement your inference code here. Return binary labels.
        check_is_fitted(self, ['lr_'])
        y_pred = self.lr_.predict(X)
        return y_pred.astype(np.int)

    def fit_predict(self, X, y, **kwargs):
        return self.fit(X, y, **kwargs).predict(X, **kwargs)

    def save_state(self):
        # TODO:
        #  Save any trained parameters you need to load later for inference.
        #  If using pytorch, note that modules have (load_)state_dict
        #  methods that you can use.
        state_dict = {
            'lr_': self.lr_
        }

        return state_dict

    def load_state(self, state_dict):
        self.__dict__.update(**state_dict)
