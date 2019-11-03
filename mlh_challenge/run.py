import os
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score

import mlh_challenge.data as data
from mlh_challenge.model import MLHChallengeModel

logger = logging.getLogger(__name__)


def training(data_file, save_model, **kw):
    """
    Runs training of the challenge Model on a given dataset and write the
    trained model to a file.
    :param data_file: Path to data file in the challenge format.
    :param save_model: Path to output model file. Empty or None will skip
        saving.
    :param kw: Extra args.
    """

    # Load and process data
    raw_data = data.load_raw(data_file)
    X, y, feat_names = data.build_features(raw_data)

    # TODO:
    #  Split into train and validation sets.
    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.1, shuffle=True)

    # TODO:
    #  Initialize your model.
    model = MLHChallengeModel(foo=3, bar=4, **kw)

    logger.info('Training...')
    y_pred_t = model.fit_predict(X_t, y_t, feat_names=feat_names)
    y_proba_t = model.predict_proba(X_t)

    y_pred_v = model.predict(X_v)
    y_proba_v = model.predict_proba(X_v)

    # Score
    logger.info(f"Train scores: {calc_scores(y_t, y_pred_t, y_proba_t)}")
    logger.info(f"Validation scores: {calc_scores(y_v, y_pred_v, y_proba_v)}")

    if save_model:
        logger.info(f"Saving model to {save_model}...")
        model.save_state(filepath=save_model)


def inference(data_file, out_file, load_model, **kw):
    """
    Runs inference using a given dataset and a pre-trained model.
    :param data_file: Path to data file in the challenge format.
    :param out_file: Path to output file where the inference results will be
        saved. Can be a .csv/.tsv./.xlsx/.xls file.
    :param load_model: Path to pre-trained model file to load.
    :param kw: Extra args.
    """
    # Load and process data
    raw_data = data.load_raw(data_file)
    X_test, y_test, _ = data.build_features(raw_data)

    # Load trained model
    model = MLHChallengeModel()
    model.load_state(filepath=load_model)

    # Run inference
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # Score
    logger.info(f"Test scores: {calc_scores(y_test, y_pred, y_proba)}")

    if out_file:
        write_output(out_file, y_pred, y_proba)


def calc_scores(y, y_pred, y_proba):
    return dict(
        precision=precision_score(y, y_pred),
        recall=recall_score(y, y_pred),
        f1=f1_score(y, y_pred),
        roc_auc=roc_auc_score(y, y_proba)
    )


def write_output(out_file, y_pred, y_proba, **writer_kw):
    df = pd.DataFrame(data=dict(y_pred=y_pred, y_proba=y_proba))

    out_path = Path(out_file)
    os.makedirs(out_path.parent, exist_ok=True)

    fmt = out_path.suffix

    if fmt in ('.csv', '.tsv'):
        writer_kw.setdefault('sep', '\t' if fmt == '.tsv' else ',')
        df.to_csv(out_path, **writer_kw)

    elif fmt in ('.xls', '.xlsx'):
        df.to_excel(out_path, **writer_kw)

    else:
        raise ValueError("Unknown output file format")

    logger.info(f"Wrote output file {out_file}")
