import pandas as pd
import glob
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
# Path to your raw data folder
DATA_DIR = r"C:\Users\rachn\OneDrive\Desktop\companiesresume\projects\NIDS-CICIDS2017\data\raw"

# Find all CSV files in that folder
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(" -", os.path.basename(f))

# Load and combine all files into one DataFrame
df_list = []
for file in csv_files:
    print(f"Loading {os.path.basename(file)}...")
    temp_df = pd.read_csv(file, encoding='latin1')
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)
print("\nCombined shape:", df.shape)

# Fix the leading/trailing space issue in column names
df.columns = df.columns.str.strip()
print("\nColumns after stripping whitespace:")
print(df.columns.tolist())

# Full label distribution across ALL days combined
print("\nFull label distribution across all files:")
print(df['Label'].value_counts())
# Step 4: Map raw labels into 4 target classes
label_map = {
    'BENIGN': 'Normal',

    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'DDoS': 'DoS',

    'PortScan': 'PortScan',

    'FTP-Patator': 'BruteForce',
    'SSH-Patator': 'BruteForce',
}

# Keep only rows whose label is in our map; drop out-of-scope attacks
before_count = len(df)
df = df[df['Label'].isin(label_map.keys())].copy()
df['Label'] = df['Label'].map(label_map)
after_count = len(df)

print(f"\nDropped {before_count - after_count} out-of-scope rows.")
print(f"Remaining rows: {after_count}")
print("\nFinal class distribution:")
print(df['Label'].value_counts())

import numpy as np

# Step 5: Handle infinite and missing values
print("\nChecking for infinite values...")
inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
print(inf_counts[inf_counts > 0])

# Replace infinities with NaN so they can be handled uniformly
df.replace([np.inf, -np.inf], np.nan, inplace=True)

print("\nChecking for missing values (NaN)...")
nan_counts = df.isna().sum()
print(nan_counts[nan_counts > 0])

# Drop rows with any NaN (simplest, safe approach given how few rows are typically affected)
before_drop = len(df)
df.dropna(inplace=True)
after_drop = len(df)
print(f"\nDropped {before_drop - after_drop} rows containing NaN/inf values.")
print(f"Final clean dataset shape: {df.shape}")

print("\nFinal class distribution after cleaning:")
print(df['Label'].value_counts())

# Step 6a: Save cleaned dataset so we don't need to reprocess raw files every time
OUTPUT_PATH = r"C:\Users\rachn\OneDrive\Desktop\companiesresume\projects\NIDS-CICIDS2017\data\cleaned_data.csv"
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved cleaned dataset to {OUTPUT_PATH}")
print(f"Shape: {df.shape}")