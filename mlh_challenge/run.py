import logging
import math

import numpy as np

import mlh_challenge.data as data
import mlh_challenge.utils as utils
from mlh_challenge.data import MLHChallengeFeatureTransformer
from mlh_challenge.model import MLHChallengeModel

logger = logging.getLogger(__name__)


def training(data_file, save_model_file, **kw):
    """
    Runs training of the challenge Model on a given dataset and write the
    trained model to a file.
    :param data_file: Path to data file in the challenge format.
    :param save_model_file: Path to output model file. Empty or None will skip
        saving.
    :param kw: Extra args.
    """

    # Load and process data
    Xb, Xp, y, Xb_names, Xp_names = data.load_raw(data_file)
    n_samples, _ = Xb.shape

    # Example: Split into train and validation sets.
    # TODO: Modify to fit your needs.
    test_ratio = .2
    idx = np.random.permutation(np.arange(n_samples))
    train_idx = idx[0:math.floor(n_samples * (1 - test_ratio))]
    valid_idx = idx[math.floor(n_samples * (1 - test_ratio)):]

    ft = MLHChallengeFeatureTransformer()

    y_train = y[train_idx]
    X_train, X_names = ft.fit_transform(
        Xb[train_idx], Xp[train_idx],
        y=y[train_idx], Xb_names=Xb_names, Xp_names=Xp_names
    )

    y_valid = y[valid_idx]
    X_valid, _ = ft.transform(Xb[valid_idx], Xp[valid_idx])

    # TODO:
    #  Initialize your model.
    model = MLHChallengeModel(foo=3, bar=4, **kw)

    logger.info('Training...')
    y_pred_t = model.fit_predict(X_train, y_train, feat_names=X_names)
    y_proba_t = model.predict_proba(X_train)

    y_pred_v = model.predict(X_valid)
    y_proba_v = model.predict_proba(X_valid)

    # Score
    logger.info(f"Train: {utils.Score(y_train, y_pred_t, y_proba_t)}")
    logger.info(f"Validation: {utils.Score(y_valid, y_pred_v, y_proba_v)}")

    if save_model_file:
        state_dict = dict(ft=ft.save_state(), model=model.save_state())
        utils.save_state(save_model_file, state_dict)


def inference(data_file, out_file, load_model_file, **kw):
    """
    Runs inference using a given dataset and a pre-trained model.
    :param data_file: Path to data file in the challenge format.
    :param out_file: Path to output file where the inference results will be
        saved. Can be a .csv/.tsv./.xlsx/.xls file.
    :param load_model_file: Path to pre-trained model file to load.
    :param kw: Extra args.
    """
    # Load trained model and feature transformer
    state_dict = utils.load_state(load_model_file)

    ft = MLHChallengeFeatureTransformer()
    ft.load_state(state_dict['ft'])

    model = MLHChallengeModel()
    model.load_state(state_dict['model'])

    # Load and process data
    Xb, Xp, y_test, Xb_names, Xp_names = data.load_raw(data_file)
    n_samples, _ = Xb.shape
    X_test, X_names = ft.transform(Xb, Xp, Xb_names, Xp_names)

    # Run inference
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # Score
    logger.info(f"Test scores: {utils.Score(y_test, y_pred, y_proba)}")

    if out_file:
        utils.write_output(out_file, y_pred, y_proba)
