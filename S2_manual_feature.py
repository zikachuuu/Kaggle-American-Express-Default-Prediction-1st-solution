"""
S2_manual_feature.py
---------------------
Feature engineering helpers used in the Kaggle American Express default predictionsolution. 

This module contains functions for one-hot encoding categorical columns and aggregating features by `customer_ID` 
so that each customer is represented by a single row of aggregated statistics. 

The original script uses global variables such as `lastk`, `num_features`, and `cat_features` 
which are set in the procedural section at the bottom of the file; 

the functions here operate on a DataFrame fragment (typically one partition handled by a process pool) and return aggregated DataFrames.

Notes / assumptions
- Input DataFrame must contain a `customer_ID` column.
- `S_2` is treated specially for ordering (it's used to compute 'rank').
- The functions rely on some global names (e.g. `num_features`, `cat_features`, and `lastk`) 
  which are set in the calling scope in the original script. 
  This keeps compatibility with the original code; you can also set these variables
  in your own scope before calling the functions.

The module exports the following functions:
- one_hot_encoding(df, cols, is_drop=True)
- cat_feature(df) 
- num_feature(df)
- diff_feature(df)

Each function includes a usage example in the docstring that shows a small
input and the expected schema/shape of the output.
"""

import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc,os,random
import time,datetime
from tqdm import tqdm
from multiprocessing import Pool as ThreadPool


def one_hot_encoding(df, cols, is_drop=True):
    """
    Perform one-hot encoding on categorical columns and attach the new dummy columns to the provided DataFrame. 
    New columns are prefixed with 'oneHot_<original_col_name>_'. 
    The original columns are optionally dropped.

    Parameters
    - df: pandas.DataFrame
        Input DataFrame containing the categorical columns and a `customer_ID`.
    - cols: list[str]
        List of column names in `df` to one-hot encode.
    - is_drop: bool, default True
        If True, drop the original categorical columns after creating dummies.

    Returns
    - pandas.DataFrame
        The input DataFrame with new one-hot columns appended (and optionally
        original categorical columns removed).
    """
    for col in cols:
        print('one hot encoding:', col)
        # Use pd.Series to ensure we preserve index alignment
        dummies = pd.get_dummies(pd.Series(df[col]), prefix=f'oneHot_{col}')
        df = pd.concat([df, dummies], axis=1)
    if is_drop:
        df.drop(cols, axis=1, inplace=True)
    return df


def cat_feature(df):
    """
    Aggregate categorical features (including one-hot encoded columns) by `customer_ID`.

    Behavior
    - Finds columns containing 'oneHot' and aggregates them with ['mean','std', 'sum', 'last'] when global `lastk` is None (full history) 
      or without 'last' when `lastk` is set (using a limited tail of records).
    - Aggregates original categorical features (defined in the global `cat_features` list) with ['last', 'nunique'] when `lastk` is None, 
      or ['nunique'] when `lastk` is set.
    - Adds a count of records per `customer_ID` using the `S_2` column.

    Parameters
    - df: pandas.DataFrame
        A DataFrame slice containing rows for many customers. Must contain columns: 'customer_ID', 'S_2', 
        one-hot columns (prefixed with 'oneHot_'), and the columns listed in `cat_features`.

    Returns
    - pandas.DataFrame
        Aggregated DataFrame indexed by `customer_ID` (as a regular column),
        where column names are produced by joining the original column name and
        aggregation function with an underscore (e.g. 'oneHot_cat_mean').
    """
    # collect one-hot encoded columns
    one_hot_features = [col for col in df.columns if 'oneHot' in col]

    # Aggregate one-hot columns. If lastk is None (full history) include the
    # 'last' aggregation to capture the most recent value per customer.
    if lastk is None:
        num_agg_df = df.groupby("customer_ID", sort=False)[one_hot_features].agg(['mean', 'std', 'sum', 'last'])
    else:
        num_agg_df = df.groupby("customer_ID", sort=False)[one_hot_features].agg(['mean', 'std', 'sum'])
    # flatten multiindex columns produced by agg
    num_agg_df.columns = ['_'.join(x) for x in num_agg_df.columns]

    # Aggregate original categorical columns. 'last' is only meaningful when
    # we keep full history (lastk is None).
    if lastk is None:
        cat_agg_df = df.groupby("customer_ID", sort=False)[cat_features].agg(['last', 'nunique'])
    else:
        cat_agg_df = df.groupby("customer_ID", sort=False)[cat_features].agg(['nunique'])
    cat_agg_df.columns = ['_'.join(x) for x in cat_agg_df.columns]

    # Add a record count per customer using S_2 (dates). The count is useful
    # as a feature representing how many records (rows) each customer has.
    count_agg_df = df.groupby("customer_ID", sort=False)[['S_2']].agg(['count'])
    count_agg_df.columns = ['_'.join(x) for x in count_agg_df.columns]

    # Concatenate all aggregated pieces and reset index so customer_ID is a
    # regular column (matching the original script's output format).
    df_out = pd.concat([num_agg_df, cat_agg_df, count_agg_df], axis=1).reset_index()
    print('cat feature shape after engineering', df_out.shape)

    return df_out

def num_feature(df):
    """
    Aggregate numerical features by `customer_ID`.

    Behavior
    - If the first entry in the `num_features` global list starts with
      'rank_', this function only computes the 'last' aggregation (the latest
      rank). This supports the transformed features created by ranking
      operations.
    - Otherwise, compute standard aggregations: mean, std, min, max, sum, and
      optionally 'last' when `lastk` is None (full history) or omit 'last' for
      truncated histories.
    - After aggregation (except for rank_ features), the function scales values
      by flooring division with 0.01: `val = val // 0.01`. This appears to be
      intended as a coarse quantization to reduce cardinality / memory.

    Parameters
    - df: pandas.DataFrame
        Input slice with columns matching the global `num_features` list and a
        `customer_ID` column.

    Returns
    - pandas.DataFrame
        Aggregated numeric features per customer with flattened column names.

    Example
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'customer_ID': ['A','A','B'],
    ...    'num1': [1.0, 3.0, 2.0],
    ...    'S_2': ['2020-01-01','2020-02-01','2020-01-01']
    ... })
    >>> global num_features, lastk
    >>> num_features = ['num1']
    >>> lastk = None
    >>> out = num_feature(df)
    >>> # one row per customer
    >>> out.shape[0]
    2

    """
    # When the features are already rank_* style we only keep the last value
    if num_features[0][:5] == 'rank_':
        num_agg_df = df.groupby("customer_ID", sort=False)[num_features].agg(['last'])
    else:
        if lastk is None:
            num_agg_df = df.groupby("customer_ID", sort=False)[num_features].agg(['mean', 'std', 'min', 'max', 'sum', 'last'])
        else:
            num_agg_df = df.groupby("customer_ID", sort=False)[num_features].agg(['mean', 'std', 'min', 'max', 'sum'])
    num_agg_df.columns = ['_'.join(x) for x in num_agg_df.columns]

    # Quantize numeric aggregated columns to reduce cardinality and memory. This
    # mirrors the original script behaviour: integer division by 0.01.
    if num_features[0][:5] != 'rank_':
        for col in num_agg_df.columns:
            # using floor division to reproduce original behaviour
            num_agg_df[col] = num_agg_df[col] // 0.01

    df_out = num_agg_df.reset_index()
    print('num feature shape after engineering', df_out.shape)

    return df_out

def diff_feature(df):
    """
    Create features based on the difference between consecutive rows for each
    customer's time series and aggregate those differences by customer.

    Behavior
    - Computes group-wise difference: for each `customer_ID`, the difference of
      each numeric column relative to the previous row is computed: df.groupby(...).diff()
    - The resulting columns are prefixed with 'diff_'. These difference
      columns are then aggregated per customer with ['mean','std','min','max','sum']
      and optionally 'last' when `lastk` is None.
    - After aggregation the values are quantized by floor-division with 0.01
      to match the original script behaviour.

    Parameters
    - df: pandas.DataFrame
        Input slice containing `customer_ID` and the numeric columns listed in
        the global `num_features`.

    Returns
    - pandas.DataFrame
        Aggregated difference features per customer, with flattened column
        names like 'diff_num1_mean', 'diff_num1_last', etc.

    Example
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'customer_ID': ['A','A','B'],
    ...    'num1': [1.0, 3.0, 2.0],
    ...    'S_2': ['2020-01-01','2020-02-01','2020-01-01']
    ... })
    >>> global num_features, lastk
    >>> num_features = ['num1']
    >>> lastk = None
    >>> out = diff_feature(df)
    >>> out.shape[0]
    2

    """
    # names for the diffed columns (used after computing groupwise differences)
    diff_num_features = [f'diff_{col}' for col in num_features]

    # Keep the original index of customer_IDs in the same order as input. We
    # compute the groupwise diff and then re-insert customer_ID for grouping.
    cids = df['customer_ID'].values
    df_diff = df.groupby('customer_ID')[num_features].diff().add_prefix('diff_')
    df_diff.insert(0, 'customer_ID', cids)

    # Aggregate the diff features per customer. Include 'last' only when
    # lastk is None to match the historical/full aggregation behaviour.
    if lastk is None:
        num_agg_df = df_diff.groupby("customer_ID", sort=False)[diff_num_features].agg(['mean', 'std', 'min', 'max', 'sum', 'last'])
    else:
        num_agg_df = df_diff.groupby("customer_ID", sort=False)[diff_num_features].agg(['mean', 'std', 'min', 'max', 'sum'])
    num_agg_df.columns = ['_'.join(x) for x in num_agg_df.columns]

    # Quantize as in num_feature
    for col in num_agg_df.columns:
        num_agg_df[col] = num_agg_df[col] // 0.01

    df_out = num_agg_df.reset_index()
    print('diff feature shape after engineering', df_out.shape)

    return df_out

n_cpu = 16
transform = [['','rank_','ym_rank_'],[''],['']]

for li, lastk in enumerate([None,3,6]):
    for prefix in transform[li]:
        df = pd.read_feather(f'./input/train.feather').append(pd.read_feather(f'./input/test.feather')).reset_index(drop=True)
        all_cols = [c for c in list(df.columns) if c not in ['customer_ID','S_2']]
        cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
        num_features = [col for col in all_cols if col not in cat_features]
        for col in [col for col in df.columns if 'S_' in col or 'P_' in col]:
            if col != 'S_2':
                df[col] = df[col].fillna(0)

        if lastk is not None:
            prefix = f'last{lastk}_' + prefix
            print('all df shape',df.shape)
            df['rank'] = df.groupby('customer_ID')['S_2'].rank(ascending=False)
            df = df.loc[df['rank']<=lastk].reset_index(drop=True)
            df = df.drop(['rank'],axis=1)
            print(f'last {lastk} shape',df.shape)

        if prefix == 'rank_':
            cids = df['customer_ID'].values
            df = df.groupby('customer_ID')[num_features].rank(pct=True).add_prefix('rank_')
            df.insert(0,'customer_ID',cids)
            num_features = [f'rank_{col}' for col in num_features]

        if prefix == 'ym_rank_':
            cids = df['customer_ID'].values
            df['ym'] = df['S_2'].apply(lambda x:x[:7])
            df = df.groupby('ym')[num_features].rank(pct=True).add_prefix('ym_rank_')
            num_features = [f'ym_rank_{col}' for col in num_features]
            df.insert(0,'customer_ID',cids)

        if prefix in ['','last3_']:
            df = one_hot_encoding(df,cat_features,False)

        vc = df['customer_ID'].value_counts(sort=False).cumsum()
        batch_size = int(np.ceil(len(vc) / n_cpu))
        dfs = []
        start = 0
        for i in range(min(n_cpu,int(np.ceil(len(vc) / batch_size)))):
            vc_ = vc[i*batch_size:(i+1)*batch_size]
            dfs.append(df[start:vc_[-1]])
            start = vc_[-1]

        pool = ThreadPool(n_cpu)

        if prefix in ['','last3_']:
            cat_feature_df = pd.concat(pool.map(cat_feature,tqdm(dfs,desc='cat_feature'))).reset_index(drop=True)

            cat_feature_df.to_feather(f'./input/{prefix}cat_feature.feather')

        if prefix in ['','last3_','last6_','rank_','ym_rank_']:
            num_feature_df = pd.concat(pool.map(num_feature,tqdm(dfs,desc='num_feature'))).reset_index(drop=True)
            num_feature_df.to_feather(f'./input/{prefix}num_feature.feather')

        if prefix in ['','last3_']:
            diff_feature_df = pd.concat(pool.map(diff_feature,tqdm(dfs,desc='diff_feature'))).reset_index(drop=True)
            diff_feature_df.to_feather(f'./input/{prefix}diff_feature.feather')

        pool.close()
