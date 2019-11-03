import logging

import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class Score(object):
    def __init__(self, y, y_pred, y_proba):
        self.precision = precision_score(y, y_pred)
        self.recall = recall_score(y, y_pred)
        self.f1 = f1_score(y, y_pred)
        self.roc_auc = roc_auc_score(y, y_proba)

    def __repr__(self):
        return f'precision={self.precision:.3f}, recall={self.recall:.3f}, ' \
               f'f1={self.f1:.3f}, roc_auc={self.roc_auc:.3f}'


def save_state(filepath, state_dict):
    filepath = Path(filepath)  # in case it's a str
    os.makedirs(filepath.parent, exist_ok=True)
    if filepath.suffix.lower() != '.pkl':
        filepath = Path(f'{filepath}.pkl')

    logger.info(f"Saving model to {filepath}...")
    with open(filepath, mode='wb') as f:
        pickle.dump(state_dict, f)


def load_state(filepath):
    logger.info(f"Loading model from {filepath}...")
    with open(filepath, mode='rb') as f:
        state_dict = pickle.load(f)

    return state_dict


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
