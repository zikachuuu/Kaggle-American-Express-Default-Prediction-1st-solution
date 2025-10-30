import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def denoise(df):
    # convert categorical to numerical
    # stored as int8 to save memory
    df['D_63'] = df['D_63'].apply(lambda t: {'CR':0, 'XZ':1, 'XM':2, 'CO':3, 'CL':4, 'XL':5}[t]).astype(np.int8)
    df['D_64'] = df['D_64'].apply(lambda t: {np.nan:-1, 'O':0, '-1':1, 'R':2, 'U':3}[t]).astype(np.int8)

    # For all columns except customer_ID, S_2 (date), D_63, and D64, multiply by 100 and floor the values
    # This reduces noise by removing decimal precision
    # and saves memory by converting to int16
    for col in tqdm(df.columns):
        if col not in ['customer_ID','S_2','D_63','D_64']:
            df[col] = np.floor(df[col]*100)
    return df

train = pd.read_csv('./input/train_data.csv')
train = denoise(train)
train.to_feather('./input/train.feather')

del train

test = pd.read_csv('./input/test_data.csv')
test = denoise(test)
test.to_feather('./input/test.feather')
