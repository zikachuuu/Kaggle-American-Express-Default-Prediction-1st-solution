"""
This script performs manual feature engineering on the American Express Default Prediction dataset.

It considers 3 different transformation types and 3 different temporal windows
Transformation types:
    - No transform (Raw features)
    - rank transform (Within customer ranking): for each feature of one customer, rank each value among his own history 
                                                (as percentile, with min as 0 and max as 1)
    - ym_rank transform (Cross customer monthly ranking): for each feature, within each month, rank each customer's value among all customers that month 
                                                          (as percentile, with min as 0 and max as 1)

Temporal windows:
    - Full history
    - last 3 months
    - last 6 months
 
It creates 5 different feature sets based on these combinations:
    - Full history  , no transform
    - Full history  , rank transform
    - Full history  , ym_rank transform
    - Last 3 months , no transform
    - Last 6 months , no transform

It generates output 9 feather files:
    - cat_feature           :                       categorical features                    aggregated by customer_ID
    - diff_feature          :                       difference features                     aggregated by customer_ID
    - last3_cat_feature     :                       categorical features from last 3 months aggregated by customer_ID
    - last3_diff_feature    :                       difference features  from last 3 months aggregated by customer_ID
    - last3_num_feature     :                       numerical features   from last 3 months aggregated by customer_ID
    - last6_num_feature     :                       numerical features   from last 6 months aggregated by customer_ID
    - num_feature           :                       numerical features                      aggregated by customer_ID
    - rank_num_feature      : rank transformed      numerical features                      aggregated by customer_ID
    - ym_rank_num_feature   : ym_rank transformed   numerical features                      aggregated by customer_ID
"""

import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc,os,random
import time,datetime
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

import gc

def one_hot_encoding(df, cat_features, drop=False):
    """
    Purpose: Convert categorical variables into binary (0/1) columns.

    What it does:
        - Takes categorical columns like B_30 which might have values ['A', 'B', 'C']
        - Creates new columns: oneHot_B_30_A, oneHot_B_30_B, oneHot_B_30_C
        - Each row gets a 1 in the column matching its category, 0 elsewhere
        - Processes in chunks (1M rows at a time) to avoid memory overflow
    
    Key trick: Collects ALL unique categories first, then ensures every chunk has the same columns 
                (even if some categories don't appear in that chunk)
    
    If drop is True, the original categorical columns are dropped after encoding.
    """
    # First pass: get all unique categories across the entire dataset
    print("Collecting unique categories...")
    all_categories = {}
    for col in cat_features:
        all_categories[col] = df[col].unique()
        print(f"  {col}: {len(all_categories[col])} unique values")
    
    # Process in chunks to avoid memory overflow
    chunk_size = 1_000_000  # Process 1M rows at a time
    n_chunks = int(np.ceil(len(df) / chunk_size))
    
    result_chunks = []
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(df))
        
        print(f"Processing chunk {chunk_idx + 1}/{n_chunks} (rows {start_idx:,} to {end_idx:,})")
        
        df_chunk = df.iloc[start_idx:end_idx].copy()
        
        for col in cat_features:
            print(f"  one hot encoding: {col}")
            
            # Get dummies with ALL categories (not just ones in this chunk)
            dummies = pd.get_dummies(
                df_chunk[col], 
                prefix=f'oneHot_{col}',     
            )
            
            # Ensure all expected columns exist (fill missing with 0)
            expected_cols = [f'oneHot_{col}_{cat}' for cat in all_categories[col]]
            for expected_col in expected_cols:
                if expected_col not in dummies.columns:
                    dummies[expected_col] = 0
            
            # Keep only expected columns in correct order
            dummies = dummies[expected_cols]
            
            # Concatenate
            df_chunk = pd.concat([df_chunk, dummies], axis=1)
            
            if drop:
                df_chunk.drop(col, axis=1, inplace=True)
            
            del dummies
            gc.collect()
        
        result_chunks.append(df_chunk)
        
        # Clean up
        del df_chunk
        gc.collect()
    
    # Combine all chunks
    print("Combining chunks...")
    df = pd.concat(result_chunks, ignore_index=True)
    
    # Clean up
    del result_chunks
    gc.collect()
    
    return df


def cat_feature(df):
    """
    Purpose: Collapse multiple rows per customer into a single row with summary statistics.

    What it does:

    For one-hot encoded features (columns with oneHot_ prefix):
        - Calculates: mean, std, sum, last (if full history)
        - Example: If customer has 5 statements where oneHot_B_30_A = [1, 0, 1, 1, 0]:
            - oneHot_B_30_A_mean = 0.6 (60% of statements had category A)
            - oneHot_B_30_A_sum = 3 (category A appeared 3 times)
            - oneHot_B_30_A_last = 0 (most recent statement was NOT category A)
    For original categorical features:
        - Calculates: last, nunique (number of unique values)
        - Example: If customer's B_30 = ['A', 'B', 'A', 'A', 'C']:
            - B_30_last = 'C' (most recent category)
            - B_30_nunique = 3 (used 3 different categories over time)
    Record count:
        - S_2_count: How many monthly statements this customer has

    Result: One row per customer with aggregated categorical behavior.
    """
    # collect one-hot encoded columns
    one_hot_features = [col for col in df.columns if 'oneHot' in col]

    # Aggregate one-hot columns.
    if lastk is None:
        num_agg_df = df.groupby("customer_ID", sort=False)[one_hot_features].agg(['mean', 'std', 'sum', 'last'])
    else:
        num_agg_df = df.groupby("customer_ID", sort=False)[one_hot_features].agg(['mean', 'std', 'sum'])
    # Flatten multi-level columns 
    # eg ('oneHot_feature1', 'mean') -> 'oneHot_feature1_mean'
    num_agg_df.columns = ['_'.join(x) for x in num_agg_df.columns]

    # Aggregate original categorical columns. 
    if lastk is None:
        cat_agg_df = df.groupby("customer_ID", sort=False)[cat_features].agg(['last', 'nunique'])
    else:
        cat_agg_df = df.groupby("customer_ID", sort=False)[cat_features].agg(['nunique'])
    cat_agg_df.columns = ['_'.join(x) for x in cat_agg_df.columns]


    # Add a record count per customer using S_2 (dates)
    count_agg_df = df.groupby("customer_ID", sort=False)[['S_2']].agg(['count'])
    count_agg_df.columns = ['_'.join(x) for x in count_agg_df.columns]

    # Concatenate all aggregated pieces and reset index so customer_ID is a
    # regular column (matching the original script's output format).
    df_out = pd.concat([num_agg_df, cat_agg_df, count_agg_df], axis=1).reset_index()
    print('cat feature shape after engineering', df_out.shape)

    return df_out


def num_feature(df):
    """
    Purpose: Summarize numerical columns per customer.

    What it does:

    Normal case (raw features like S_3, P_2):
        - Calculates: mean, std, min, max, sum, last
        - Example: If customer's S_3 = [100, 150, 200, 180, 220]:
            - S_3_mean = 170
            - S_3_max = 220
            - S_3_last = 220 (most recent value)
        - Quantization: Divides all values by 0.01 and floors them (// 0.01)
            - This converts 170.456 → 17045 (reduces precision, saves memory)

    Rank transformed features (eg rank_S_3):
        - Only keeps last (the most recent percentile rank)
        - No quantization applied (ranks are already 0-1 scaled)

    Result: One row per customer with statistical summaries of their numerical behavior.
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
    Purpose: Capture trends and changes over time.

    What it does:

        - Computes month-to-month differences for each numerical column
        - Example: If customer's S_3 = [100, 150, 200, 180, 220]:
            - diff_S_3 = [NaN, 50, 50, -20, 40] (changes between consecutive months)
        - Then aggregates these differences: mean, std, min, max, sum, last
            - diff_S_3_mean = 30 (average monthly change)
            - diff_S_3_min = -20 (biggest drop)
            - diff_S_3_last = 40 (most recent change)

    Why useful: Detects deteriorating payment behavior (increasing debt) vs improving behavior (paying down balances).

    Quantization: Same as num_feature (floor division by 0.01).
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

# transform   = [['','rank_','ym_rank_'],[''],['']]   # feature transformations to apply
# lastks      = [None,3,6]                            # temporal windowing (None = full history, else last k months)

transform   = [['ym_rank_']]   # feature transformations to apply
lastks      = [None]                            # temporal windowing (None = full history, else last k months)

# 5 different feature sets:
#   1. All data            (lastk=None), no transform      (prefix='')
#   2. All data            (lastk=None), rank transform    (prefix='rank_')
#   3. All data            (lastk=None), ym_rank transform (prefix='ym_rank_')
#   4. Last 3 months data  (lastk=3)   , no transform      (prefix='last3_')
#   5. Last 6 months data  (lastk=6)   , no transform      (prefix='last6_')

for li, lastk in enumerate(lastks):
    for prefix in transform[li]:
        # We iterate over each of the 5 feature sets
        print ('----------------------------------------------------------------')
        print (f'Processing feature set: prefix={prefix}, lastk={lastk}')
        print ('----------------------------------------------------------------')

        # Combine train and test data for feature engineering (both sets use same features)
        df = pd.concat([
            pd.read_feather(f"S:/ML_Project/new_data/train_data.feather"),
            pd.read_feather(f"S:/ML_Project/new_data/test_data.feather")
        ], ignore_index=True)

        print ('all df shape',df.shape)
        
        # Get the list of all columns, categorical features, and numerical feature names
        all_cols        = [c for c in list(df.columns) if c not in ['customer_ID','S_2']]
        cat_features    = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
        num_features    = [col for col in all_cols if col not in cat_features]

        print (f'all columns except ID and date: {len(all_cols)}')
        print (f'categorical features: {len(cat_features)}')
        print (f'numerical features: {len(num_features)}')

        # Fill NaNs in 'S_' and 'P_' columns (except 'S_2' ie date) with 0
        # S represent Spend variables, P represent Payment variables
        # We can fill NaNs with 0 as no spend / no payment
        for col in [col for col in df.columns if 'S_' in col or 'P_' in col]:
            if col != 'S_2':
                df[col] = df[col].fillna(0)

        print ('NaNs in S_ and P_ columns filled')


        unique_customers    = df['customer_ID'].unique()
        n_customers         = len(unique_customers)
        chunk_size          = 100_000  # Adjust based on your memory
        n_chunks            = int(np.ceil(n_customers / chunk_size))

        print(f"Total customers: {n_customers:,}")
        print(f"Chunk size: {chunk_size:,}")
        print(f"Total chunks: {n_chunks}")

        # Temporal windowing (set 4, 5)
        # Keep only the last `lastk` records per customer
        # eg if lastk=3, keep only the most recent 3 months of data per customer
        if lastk is not None:
            print (f'Temporal windowing: keeping last {lastk} records per customer')
            prefix      = f'last{lastk}_' + prefix
            df['rank']  = df.groupby('customer_ID')['S_2'].rank(ascending=False)    # rank the dates per customer (1 = most recent)
            df          = df.loc[df['rank']<=lastk].reset_index(drop=True)          # keep only lastk records
            df          = df.drop(['rank'],axis=1)                                  # drop the temporary rank column
            print(f'last {lastk} shape',df.shape)


        # Rank transformations (Set 2)
        # For each customer, convert all numeric features to their rank (percentile) within that customer's history
        # eg a customer's S_1 values [10, 20, 15] would be transformed to [0.33, 1.0, 0.67]
        # This captures relative standing of each value within the customer's timeline
        if prefix == 'rank_':
            print('performing rank transformation per customer')
                        
            print(f"Total customers: {n_customers:,}")
            print(f"Processing in {n_chunks} chunks of {chunk_size:,} customers each")
            
            rank_chunks = []
            
            for i in range(0, n_customers, chunk_size):
                chunk_idx = i // chunk_size + 1
                chunk_customers = unique_customers[i:i+chunk_size]
                df_chunk = df[df['customer_ID'].isin(chunk_customers)].copy()
                
                print(f"Ranking chunk {chunk_idx}/{n_chunks}: {len(chunk_customers):,} customers, {len(df_chunk):,} rows")
                
                # Perform rank transformation on this chunk
                ranked_chunk = df_chunk.groupby('customer_ID')[num_features].rank(pct=True).add_prefix('rank_')
                # add back customer_ID
                ranked_chunk.insert (0, 'customer_ID', df_chunk['customer_ID'].values)
                rank_chunks.append(ranked_chunk)
                
                del df_chunk, ranked_chunk
                gc.collect()
            
            print("Concatenating ranked chunks...")
            df = pd.concat(rank_chunks, ignore_index=False)
            del rank_chunks
            gc.collect()

            num_features = [f'rank_{col}' for col in num_features]  # update num_features to the new ranked columns
            
            print(f"Rank transformation completed: shape {df.shape}")


        # Year-month Rank Transformations (Set 3)
        # For each month, convert all numeric features to the rank (pecentile) across all customers within the same month
        # eg for month 2020-01, if customer A has S_1=10, B has S_1=20, C has S_1=15
        # then their ym_rank_S_1 values would be [0.33, 1.0, 0.67] respectively
        # Captures how a customer compares to others in that specific month
        if prefix == 'ym_rank_':
            print('performing rank transformation per customer per year-month')
                        
            print(f"Total customers: {n_customers:,}")
            print(f"Processing in {n_chunks} chunks of {chunk_size:,} customers each")
            
            rank_chunks = []
            
            for i in range(0, n_customers, chunk_size):
                chunk_idx = i // chunk_size + 1
                chunk_customers = unique_customers[i:i+chunk_size]
                df_chunk = df[df['customer_ID'].isin(chunk_customers)].copy()
                
                print(f"Ranking chunk {chunk_idx}/{n_chunks}: {len(chunk_customers):,} customers, {len(df_chunk):,} rows")
                
                # Perform rank transformation on this chunk
                df_chunk['ym'] = df_chunk['S_2'].apply (lambda x:x[:7]) # extract year-month from date
                ranked_chunk = df_chunk.groupby('ym')[num_features].rank(pct=True).add_prefix('ym_rank_')
                # add back customer_ID
                ranked_chunk.insert (0, 'customer_ID', df_chunk['customer_ID'].values)
                rank_chunks.append(ranked_chunk)
                
                del df_chunk, ranked_chunk
                gc.collect()
            
            print("Concatenating ranked chunks...")
            df = pd.concat(rank_chunks, ignore_index=False)
            del rank_chunks
            gc.collect()

            num_features = [f'ym_rank_{col}' for col in num_features]  # update num_features to the new ym_ranked columns
            
            print(f"Rank transformation completed: shape {df.shape}")


        # One hot encoding (Set 1, 4)
        # We are encoding all categorical features for these 2 sets
        # The original categorical features are not dropped (we have both the original and one-hot columns)
        if prefix in ['','last3_']:
            print ('performing one-hot encoding for categorical features')
            df = one_hot_encoding(df,cat_features,False)


        # Process in chunks to manage memory
        print("Processing in chunks...")

        # Set 1, 4 - Categorical Features
        if prefix in ['','last3_']:
            print('creating categorical features')
            cat_results = []
            
            for i in range(0, n_customers, chunk_size):
                chunk_idx = i // chunk_size + 1
                chunk_customers = unique_customers[i:i+chunk_size]
                df_chunk = df[df['customer_ID'].isin(chunk_customers)].copy()
                
                print(f"Processing cat_feature chunk {chunk_idx}/{n_chunks}: {len(chunk_customers):,} customers, {len(df_chunk):,} rows")
                result = cat_feature(df_chunk)
                cat_results.append(result)
                
                del df_chunk, result
                gc.collect()
            
            print("Concatenating categorical feature results...")
            cat_feature_df = pd.concat(cat_results, ignore_index=True)
            del cat_results
            gc.collect()
            
            cat_feature_df.to_feather(f'S:/ML_Project/new_data/input/{prefix}cat_feature.feather')
            print(f'categorical features saved: shape {cat_feature_df.shape}')
            del cat_feature_df
            gc.collect()

        # Set 1, 2, 3, 4, 5 - Numerical Features
        if prefix in ['','last3_','last6_','rank_','ym_rank_']:
            print('creating numerical features')
            num_results = []
            
            for i in range(0, n_customers, chunk_size):
                chunk_idx = i // chunk_size + 1
                chunk_customers = unique_customers[i:i+chunk_size]
                df_chunk = df[df['customer_ID'].isin(chunk_customers)].copy()
                
                print(f"Processing num_feature chunk {chunk_idx}/{n_chunks}: {len(chunk_customers):,} customers, {len(df_chunk):,} rows")
                result = num_feature(df_chunk)
                num_results.append(result)
                
                del df_chunk, result
                gc.collect()
            
            print("Concatenating numerical feature results...")
            num_feature_df = pd.concat(num_results, ignore_index=True)
            del num_results
            gc.collect()
            
            num_feature_df.to_feather(f'S:/ML_Project/new_data/input/{prefix}num_feature.feather')
            print(f'numerical features saved: shape {num_feature_df.shape}')
            del num_feature_df
            gc.collect()

        # Set 1, 4 - Difference Features
        if prefix in ['','last3_']:
            print('creating difference features')
            diff_results = []
            
            for i in range(0, n_customers, chunk_size):
                chunk_idx = i // chunk_size + 1
                chunk_customers = unique_customers[i:i+chunk_size]
                df_chunk = df[df['customer_ID'].isin(chunk_customers)].copy()
                
                print(f"Processing diff_feature chunk {chunk_idx}/{n_chunks}: {len(chunk_customers):,} customers, {len(df_chunk):,} rows")
                result = diff_feature(df_chunk)
                diff_results.append(result)
                
                del df_chunk, result
                gc.collect()
            
            print("Concatenating difference feature results...")
            diff_feature_df = pd.concat(diff_results, ignore_index=True)
            del diff_results
            gc.collect()
            
            diff_feature_df.to_feather(f'S:/ML_Project/new_data/input/{prefix}diff_feature.feather')
            print(f'difference features saved: shape {diff_feature_df.shape}')
            del diff_feature_df
            gc.collect()

        # Clean up main dataframe at the end of each feature set iteration
        del df
        gc.collect()

        print(f'feature set with prefix={prefix}, lastk={lastk} completed')

