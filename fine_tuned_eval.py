# Evaluation of fine-tuned BioClinical MordernBERT on test dataset

import torch 
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

data = torch.load("../Data_NLPinHealth/tokenized_phenotyping/test_tokenized.pt")

input_ids = data["input_ids"]
attention_mask = data["attention_mask"]
labels = data["labels"]

from torch.utils.data import Dataset

class MultiLabelDataset(Dataset):
    def __init__(self, data):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels = data["labels"]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

    def __len__(self):
        return len(self.input_ids)

test_dataset = MultiLabelDataset(data)

model = AutoModelForSequenceClassification.from_pretrained(
    "../Models/Finetuned_BioClinical_ModernBERT_base",
    num_labels=2,
    problem_type="multi_label_classification"
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=32
)

trainer = Trainer(
    model=model,
    args=training_args
)

predictions = trainer.predict(test_dataset)

import numpy as np
from sklearn.metrics import classification_report

# Converting logits to probabilities
probs = 1 / (1 + np.exp(-predictions.predictions))
y_pred = (probs > 0.5).astype(int)

# True labels
y_true = data["labels"].numpy()

# Evaluate
print(classification_report(y_true, y_pred, target_names=["Heart_Failure", "Diabetes"]))