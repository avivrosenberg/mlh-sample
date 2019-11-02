import numpy as np
import logging
from pathlib import Path

from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


def load_raw(filename):
    filepath = Path(filename).absolute()
    logger.info(f'Loading data file {filepath}...')

    data = np.load(filepath, allow_pickle=True)
    return data


def build_features(raw_data):
    logger.info(f'Processing data...')
    # TODO:
    #  Process the raw data to build your features for training/inference.

    Xb, Xb_names = raw_data['phys_feat'], raw_data['phys_feat_names']
    Xp, Xp_names = raw_data['pol_feat'], raw_data['pol_feat_names']

    # As a demo, here we'll just use all basic and pollution features as
    # they are, without any processing.
    n_samples, _ = Xb.shape
    X = np.hstack((Xb, Xp.reshape(n_samples, -1)))
    Xp_names_flat = [Xp_names[j] + f'_{i + 1:02d}'
                     for i in range(Xp.shape[1]) for j in range(Xp.shape[2])]
    feat_names = np.hstack((Xb_names, Xp_names_flat))

    # Handle missing samples: naively replace them with -1 (not a good idea...)
    imputer = SimpleImputer(missing_values=np.nan, strategy='constant',
                            fill_value=-1.)
    X = imputer.fit_transform(X,)

    logger.info(f'Processed data shape: {X.shape}')
    return X, feat_names
