import numpy as np
import logging
from pathlib import Path

import mlh_challenge

logger = logging.getLogger(__name__)


def load_raw(filename):
    filepath = Path(filename).absolute()
    logger.info(f'Loading data file {filepath}...')

    data = np.load(filepath)
    return data


def build_features(raw_data):
    logger.info(f'Processing data...')
    # TODO:
    #  Process the raw data to build your features for training/inference.

    Xb, Xb_names = raw_data['feat_basic'], raw_data['feat_basic_names']
    Xp, Xp_names = raw_data['feat_pollution'], raw_data['feat_pollution_names']

    # As a demo, here we'll just combine all basic and pollution features as
    # they are, without any processing.
    n_patients, _ = Xb.shape
    X = np.hstack((Xb, Xp.reshape(n_patients, -1)))
    feat_names = list(np.hstack((Xb_names, Xp_names)))

    logger.info(f'Processed data shape: {X.shape}')
    return X, feat_names
