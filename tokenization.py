import os
import torch
import pandas as pd
from transformers import AutoTokenizer

# Paths
model_dir = "Model/BioClinical_ModernBERT_base"
data_dir = "Data/"
save_dir = "Data/"
os.makedirs(save_dir, exist_ok=True)

# Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

splits = ["train", "val", "test"]

for split in splits:
    path = os.path.join(data_dir, f"{split}.csv")
    print(f"\n Processing {split} split from {path}")

    df = pd.read_csv(path)
    df = df.dropna(subset=["text"])  
    texts = df["text"].astype(str).tolist()

    # Converting labels to torch tensor (multi-label)
    labels = torch.tensor(
        df[["heartfailure_label", "diabetes_label"]].values, dtype=torch.float
    )

    # Tokenizing in batches to prevent memory overflow
    all_input_ids, all_attention_masks = [], []
    batch_size = 256  
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=4096,
            return_tensors="pt"
        )
        all_input_ids.append(encodings["input_ids"])
        all_attention_masks.append(encodings["attention_mask"])

    # Concatenating all batches
    input_ids = torch.cat(all_input_ids)
    attention_mask = torch.cat(all_attention_masks)

    # Saving tokenized tensors
    save_path = os.path.join(save_dir, f"{split}_tokenized.pt")
    torch.save({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }, save_path)
