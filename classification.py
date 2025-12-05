import torch
from datasets import Dataset
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

train_data = torch.load("../Data_NLPinHealth/tokenized_phenotyping/train_tokenized.pt")
val_data   = torch.load("../Data_NLPinHealth/tokenized_phenotyping/val_tokenized.pt")

train_dataset = Dataset.from_dict({
    "input_ids": train_data["input_ids"],
    "attention_mask": train_data["attention_mask"],
    "labels": train_data["labels"]
})

val_dataset = Dataset.from_dict({
    "input_ids": val_data["input_ids"],
    "attention_mask": val_data["attention_mask"],
    "labels": val_data["labels"]
})

model = AutoModelForSequenceClassification.from_pretrained(
    "../Models/BioClinical_ModernBERT_base",
    num_labels=2,  # Diabetes and Heart Failure
    problem_type="multi_label_classification"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = labels.astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir="../outputs",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=112,
    per_device_eval_batch_size=112,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="../logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    bf16=True,
    optim="adamw_torch_fused",
    gradient_checkpointing=True,
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    max_grad_norm=1.0,
    report_to="none",
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("../Models/Finetuned_BioClinical_MordernBERT_base")
