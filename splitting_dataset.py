import pandas as pd
from sklearn.model_selection import train_test_split

# Load stratified balanced dataset
df = pd.read_csv("Data/final_result_preprocessed.csv")
print("Loaded", len(df), "rows")

# Combine labels for stratification
df["label_combo"] = (
    df["heartfailure_label"].astype(str) + "_" + df["diabetes_label"].astype(str)
)

# Split into train (70%), val (15%), test (15%)
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["label_combo"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label_combo"], random_state=42
)

# Drop helper column
for split in [train_df, val_df, test_df]:
    split.drop(columns=["label_combo"], inplace=True)

# Save splits
train_df.to_csv("Data/train.csv", index=False)
val_df.to_csv("Data/val.csv", index=False)
test_df.to_csv("Data/test.csv", index=False)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
