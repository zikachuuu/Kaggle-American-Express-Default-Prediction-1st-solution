"""
Denoising and converting split CSV files to feather format

1. Convert categorical columns with string values to integer codes
2. For all other numerical columns (except customer_ID and S_2), multiply by 100 and floor the values to reduce noise and save memory
3. Combine all split files into single feather files for train and test datasets
"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import os

def denoise_chunk(df):
    # convert categorical columns with string values to integer codes
    df['D_63'] = df['D_63'].map({'CR':0, 'XZ':1, 'XM':2, 'CO':3, 'CL':4, 'XL':5}).astype(np.int8)
    df['D_64'] = df['D_64'].fillna(-1).map({-1:-1, 'O':0, '-1':1, 'R':2, 'U':3}).astype(np.int8)
    
    # For all columns except customer_ID, S_2 (date), D_63, and D64, multiply by 100 and floor the values
    # This reduces noise by removing decimal precision
    # and saves memory by converting to int16
    for col in df.columns:
        if col not in ['customer_ID','S_2','D_63','D_64']:
            df[col] = np.floor(df[col]*100).astype(np.int16)
    return df

def process_split_files(input_folder, output_file):
    """
    Process multiple split CSV files and combine into single feather file
    """
    # Get all CSV files in the folder
    csv_files = sorted(glob.glob(os.path.join(input_folder, '*.csv')))
    print(f"Found {len(csv_files)} files to process")
    
    chunks = []

    for file in tqdm(csv_files, desc="Processing files"):
        # Read file
        df = pd.read_csv(file)
        # Process
        df = denoise_chunk(df)
        chunks.append(df)
        print(f"Processed {file}, shape: {df.shape} (total chunks: {len(chunks)})")
        
        # Optional: periodically save and clear memory if still running out
        # if len(chunks) >= 5:
        #     temp_df = pd.concat(chunks, ignore_index=True)
        #     chunks = [temp_df]
    
    # Concatenate all chunks
    print("Concatenating all data...")
    final_df = pd.concat(chunks, ignore_index=True)
    
    # Save as feather
    print(f"Saving to {output_file}...")
    final_df.to_feather(output_file)
    
    del chunks, final_df
    print("Done!")

# Process train data
print("Processing train data...")
process_split_files(r"/S:/ML_Project/amex-default-prediction/train_data_splitted/", r"/S:/ML_Project/new_data/train_data.feather")

# Process test data
print("\nProcessing test data...")
process_split_files(r"/S:/ML_Project/amex-default-prediction/test_data_splitted/", r"/S:/ML_Project/new_data/test_data.feather")