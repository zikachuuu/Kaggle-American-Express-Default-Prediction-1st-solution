import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc, os, random
import time, datetime
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from utils import *


"""
S3_series_feature.py
---------------------
This script builds and trains a LightGBM model using series features. It's a
top-level experiment script (mirrors the original project structure) and does
the following at a high level:

- Loads the train/test feather files from `./input/`.
- Defines a helper `one_hot_encoding` used in preprocessing.
- Loads labels and merges them into `train`.
- Assembles LGB configuration in `lgb_config` and calls
  `Lgb_train_and_predict` from `utils`.

Notes
- This module uses variables from `utils` (imported with `from utils import *`)
  such as `args`, `id_name`, `label_name`, and the function
  `Lgb_train_and_predict`. To run this script interactively, ensure those are
  available in the environment or call the relevant functions from another
  script.
"""

# CLI / args values (provided by utils.args in the original workflow)
root = args.root
seed = args.seed


# Load raw data (feather format) — same as original script
train = pd.read_feather(f'./input/train.feather')
test = pd.read_feather(f'./input/test.feather')


def one_hot_encoding(df, cols, is_drop=True):
    """
    One-hot encode specified categorical columns in-place and return the
    augmented DataFrame.

    Parameters
    - df: pandas.DataFrame
        Input DataFrame that contains the columns listed in `cols`.
    - cols: list[str]
        Column names to one-hot encode.
    - is_drop: bool (default True)
        When True, drop the original categorical columns after encoding.

    Returns
    - pandas.DataFrame: same DataFrame with new one-hot columns appended.

    Example
    >>> import pandas as pd
    >>> df = pd.DataFrame({'customer_ID': ['A','B'], 'cat': ['x','y']})
    >>> out = one_hot_encoding(df.copy(), ['cat'], is_drop=False)
    >>> sorted([c for c in out.columns if c.startswith('oneHot_cat_')])
    ['oneHot_cat_x', 'oneHot_cat_y']

    Notes
    - This function mirrors the behavior used in other modules of the
      repository; it uses `pd.get_dummies` and concatenates the result.
    """
    for col in cols:
        print('one hot encoding:', col)
        dummies = pd.get_dummies(pd.Series(df[col]), prefix=f'oneHot_{col}')
        df = pd.concat([df, dummies], axis=1)
    if is_drop:
        df.drop(cols, axis=1, inplace=True)
    return df


# List of categorical features used elsewhere in the pipeline
cat_features = [
    "B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126",
    "D_63", "D_64", "D_66", "D_68",
]

# Small epsilon used for numeric stability in some operations
eps = 1e-3


# Load train labels and merge into train DataFrame
train_y = pd.read_csv(f'{root}/train_labels.csv')
train = train.merge(train_y, how='left', on=id_name)

print(train.shape, test.shape)


# LightGBM configuration used for training. This dictionary is passed to the
# helper `Lgb_train_and_predict` from `utils` which expects these keys.
lgb_config = {
    'lgb_params': {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'dart',
        'max_depth': -1,
        'num_leaves': 64,
        'learning_rate': 0.035,
        'bagging_freq': 5,
        'bagging_fraction': 0.7,
        'feature_fraction': 0.7,
        'min_data_in_leaf': 256,
        'max_bin': 63,
        'min_data_in_bin': 256,
        # 'min_sum_heassian_in_leaf': 10,
        'tree_learner': 'serial',
        'boost_from_average': 'false',
        'lambda_l1': 0.1,
        'lambda_l2': 30,
        'num_threads': 24,
        'verbosity': 1,
    },
    'feature_name': [col for col in train.columns if col not in [id_name, label_name, 'S_2']],
    'rounds': 4500,
    'early_stopping_rounds': 100,
    'verbose_eval': 50,
    'folds': 5,
    'seed': seed,
}


# Train and predict. `Lgb_train_and_predict` is imported from utils. The call
# is kept identical to the original script to preserve behavior.
Lgb_train_and_predict(train, test, lgb_config, gkf=True, aug=None, run_id='LGB_with_series_feature')
