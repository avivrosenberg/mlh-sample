import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score

import mlh_challenge.data as data
from mlh_challenge.model import MLHChallengeModel

logger = logging.getLogger(__name__)


def training(data_file, save_model, **kw):
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


def inference(data_file, load_model, **kw):
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


def calc_scores(y, y_pred, y_proba):
    return dict(
        precision=precision_score(y, y_pred),
        recall=recall_score(y, y_pred),
        f1=f1_score(y, y_pred),
        roc_auc=roc_auc_score(y, y_proba)
    )
