# Stratified sampling

import pandas as pd
import numpy as np

# Loading dataframe
df = pd.read_csv("Data/output_table.csv")
print("Loaded:", len(df), "rows")

# Dropping the rows where the text is missing or empty
before_text = len(df)
df = df[df['text'].notna() & (df['text'].str.strip() != "")]
after_text = len(df)
print(f"Removed {before_text - after_text:,} rows with empty or missing text")
print(f"Remaining rows with valid text: {after_text:,}")

# Dropping duplicate hospital admissions 
before = len(df)
df = df.drop_duplicates(subset=['hadm_id'], keep='first')
after = len(df)
print(f"Removed {before - after:,} duplicate rows based on hadm_id")
print(f"Remaining unique admissions: {after:,}")

# Computing class counts
hf_only_df = df[(df['heartfailure_label']==1) & (df['diabetes_label']==0)]
dm_only_df = df[(df['diabetes_label']==1) & (df['heartfailure_label']==0)]
both_df    = df[(df['heartfailure_label']==1) & (df['diabetes_label']==1)]
neither_df = df[(df['heartfailure_label']==0) & (df['diabetes_label']==0)]

print("\nClass counts:")
print(f"  HF only: {len(hf_only_df):,}")
print(f"  DM only: {len(dm_only_df):,}")
print(f"  Both HF+DM: {len(both_df):,}")
print(f"  Neither: {len(neither_df):,}")

# Sampling equal number of negatives for a positive subgroup
def sample_negatives(pos_count, neg_pool):
    if pos_count > len(neg_pool):
        raise ValueError("Not enough negatives to sample from")
    return neg_pool.sample(n=pos_count, random_state=42)

# Sampling negatives for each positive subgroup
neg_remaining = neither_df.copy()

neg_hf = sample_negatives(len(hf_only_df), neg_remaining)
neg_remaining = neg_remaining.drop(neg_hf.index)

neg_dm = sample_negatives(len(dm_only_df), neg_remaining)
neg_remaining = neg_remaining.drop(neg_dm.index)

neg_both = sample_negatives(len(both_df), neg_remaining)

# Combining positives and their matched negatives
sampled_df = pd.concat([hf_only_df, neg_hf, dm_only_df, neg_dm, both_df, neg_both], axis=0)

# Shuffling the final dataset
sampled_df = sampled_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Printing statistics
print("\nSampled dataset size:", len(sampled_df))
print("Label distribution (sum of 1s):")
print(sampled_df[['heartfailure_label','diabetes_label']].sum())

# Saving to csv
sampled_df.to_csv("Data/balanced_subset_stratified.csv", index=False)

