import numpy as np
import logging
from pathlib import Path

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

K_PHYS = 'phys_feat'
K_PHYS_NAMES = 'phys_feat_names'
K_POL = 'pol_feat'
K_POL_NAMES = 'pol_feat_names'
K_TARGET = 'labels'

ALL_KEYS = [K_PHYS, K_PHYS_NAMES, K_POL, K_POL_NAMES, K_TARGET]


def load_raw(filename):
    """
    Loads raw data in the challenge format.
    :param filename: Path to the data file (should be .npz).
    :return: A tuple with the following elements:
        Xb: 2d tensor (N,D1): N samples of D1 physiological features
        Xp: 3d tensor (N,K,D2): N samples of K consecutive days of
            D2 pollution features
        y: 1d tensor (N,): Binary labels representing severity based
           on length of stay (label 1 is severe).
        Xb_names: 1d tensor (D1,): Names of physiological features
        Xp_names: 1d tensor (D2,): Names of pollution features
    """
    filepath = Path(filename).absolute()
    logger.info(f'Loading data file {filepath}...')

    raw_data = np.load(filepath, allow_pickle=True)
    for k in ALL_KEYS:
        if k not in raw_data:
            raise ValueError("Data file does not contain all inputs")

    Xb, Xb_names = raw_data[K_PHYS], raw_data[K_PHYS_NAMES]
    Xp, Xp_names = raw_data[K_POL], raw_data[K_POL_NAMES]
    y = raw_data[K_TARGET]

    return Xb, Xp, y, Xb_names, Xp_names


class MLHChallengeFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, foo=1, bar=2):
        pass

    def fit(self, Xb, Xp, y=None, Xb_names=None, Xp_names=None, **kw):
        """
        Fit's the transformer's parameters to training data.
        :param Xb: Physiological features
        :param Xp: Pollution features
        :param y: Targets
        :param Xb_names: Physiological feature names
        :param Xp_names: Pollution feature names
        :param kw: extra args
        :return: self
        """
        assert Xb.shape[0] == Xp.shape[0]
        n_samples, _ = Xb.shape

        # TODO:
        #  Calculate any necessary metrics about the training data and save
        #  them as members in this class.
        #  These should be used to transform the train and test features in
        #  the same way. For example, use these for feature scaling and
        #  imputation of missing values.

        # As a demo, here we'll just use all basic and pollution features as
        # they are, without any processing.

        X = np.hstack((Xb, Xp.reshape(n_samples, -1)))

        # Example 1: Manually removing specific features
        manually_removed_features = ['x', 'y']
        manually_removed_idx = []
        if Xb_names is not None:
            for name in manually_removed_features:
                manually_removed_idx.append(np.where(Xb_names == name)[0])

        # Save all removed features as a member
        self.removed_features_ = np.concatenate(manually_removed_idx, axis=0)

        # Example 2: Calculate feature means and save as a member
        self.feature_means_ = np.nanmean(X, axis=0)

        return self

    def transform(self, Xb, Xp, Xb_names=None, Xp_names=None, **kw):
        """
        Apply the fitted transformation to data.
        :param Xb: Physiological features
        :param Xp: Pollution features
        :param Xb_names: Physiological feature names
        :param Xp_names: Pollution feature names
        :param kw: extra args
        :return: self
        """
        logger.info(f'Generating features...')

        n_samples, _ = Xb.shape
        assert Xb.shape[0] == Xp.shape[0]

        # Make sure fit was called and stats were saved
        check_is_fitted(self, ['removed_features_', 'feature_means_'])

        # TODO:
        #  Use your saved stats to transform the input. Below are some
        #  examples using the previously saved stats.

        # Example: treat all pollution sample as separate features
        X = np.hstack((Xb, Xp.reshape(n_samples, -1)))

        # Example: handle missing samples by replacing them with saved means
        i_nan, j_nan = np.where(np.isnan(X))
        X[i_nan, j_nan] = np.take(self.feature_means_, j_nan)

        # Example: Remove features marked for removal
        X = np.delete(X, self.removed_features_, axis=1)

        # Example: Generate names for pollution features and remove the
        # names of removed features.
        feat_names = []
        if Xb_names is not None and Xp_names is not None:
            Xp_names_flat = [
                Xp_names[j] + f'_{i + 1:02d}'
                for i in range(Xp.shape[1]) for j in range(Xp.shape[2])
            ]
            feat_names = np.hstack((Xb_names, Xp_names_flat))
            feat_names = np.delete(feat_names, self.removed_features_, axis=0)

        logger.info(f'Processed data shape: {X.shape}')
        return X, feat_names

    def fit_transform(self, Xb, Xp, **kw):
        return self.fit(Xb, Xp, **kw).transform(Xb, Xp, **kw)

    def save_state(self):
        # TODO:
        #  Save members that you need to write to file for later inference.
        state_dict = {
            'removed_features_': self.removed_features_,
            'feature_means_': self.feature_means_
        }

        return state_dict

    def load_state(self, state_dict):
        self.__dict__.update(**state_dict)
