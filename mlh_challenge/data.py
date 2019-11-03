import numpy as np
import logging
from pathlib import Path

from sklearn.impute import SimpleImputer

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
    :return: A dict-like object with the following keys and values:
        K_PHYS: 2d tensor (N,D1): N samples of D1 physiological features
        K_PHYS_NAMES: 1d tensor (D1,): Names of physiological features
        K_POL: 3d tensor (N,K,D2): N samples of K consecutive days of
            D2 pollution features
        K_POL_NAMES: 1d tensor (D2,): Names of pollution features
        K_TARGET: 1d tensor (N,): Binary labels representing severity based
            on length of stay (label 1 is severe).
    """
    filepath = Path(filename).absolute()
    logger.info(f'Loading data file {filepath}...')

    data = np.load(filepath, allow_pickle=True)
    for k in ALL_KEYS:
        if k not in data:
            raise ValueError("Data file does not contain all inputs")

    return dict(**data)


def build_features(raw_data):
    logger.info(f'Processing data...')
    # TODO:
    #  Process the raw data to build your features for training/inference.

    Xb, Xb_names = raw_data[K_PHYS], raw_data[K_PHYS_NAMES]
    Xp, Xp_names = raw_data[K_POL], raw_data[K_POL_NAMES]
    y = raw_data[K_TARGET]

    # As a demo, here we'll just use all basic and pollution features as
    # they are, without any processing.
    n_samples, _ = Xb.shape
    X = np.hstack((Xb, Xp.reshape(n_samples, -1)))
    Xp_names_flat = [Xp_names[j] + f'_{i + 1:02d}'
                     for i in range(Xp.shape[1]) for j in range(Xp.shape[2])]
    feat_names = np.hstack((Xb_names, Xp_names_flat))

    # Example: Remove missing features (all-nan columns)
    missing_features = np.where(np.all(np.isnan(X), axis=0))
    X = np.delete(X, missing_features, axis=1)
    feat_names = np.delete(feat_names, missing_features, axis=0)

    # Example: handle missing samples by naively replacing them with feature
    # median
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X = imputer.fit_transform(X, )

    logger.info(f'Processed data shape: {X.shape}')
    return X, y, feat_names
