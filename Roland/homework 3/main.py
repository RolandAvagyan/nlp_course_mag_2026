import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

print(sum(p.numel() for p in model.parameters()))

dataset = load_dataset("sst2")

def tokenize(batch):
    return tokenizer(
        batch["sentence"], padding=True,
        truncation=True, max_length=128
    )

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format(
    "torch", columns=["input_ids", "attention_mask", "label"]
)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    warmup_ratio=0.1,
    weight_decay=0.01,
    report_to="none",
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
)
trainer.train()


