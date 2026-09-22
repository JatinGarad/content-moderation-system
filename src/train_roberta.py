"""
train_roberta.py
-----------------
Fine-tunes RoBERTa on the Jigsaw Toxic Comment Classification dataset
to produce the checkpoint used by profanity_detection.RobertaDetector.

Usage:
    python train_roberta.py --data path/to/jigsaw/train.csv --output models/roberta-profanity

Dataset format expected (matches the Kaggle Jigsaw Toxic Comment Classification
Challenge CSV): a 'comment_text' column plus binary label columns
(toxic, severe_toxic, obscene, threat, insult, identity_hate).
We collapse these into a single binary label: toxic if ANY label == 1.
"""

import argparse
import logging

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def load_jigsaw_dataset(csv_path: str, val_size: float = 0.1, seed: int = 42):
    df = pd.read_csv(csv_path)
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected Jigsaw label columns: {missing}")

    df["label"] = (df[LABEL_COLUMNS].sum(axis=1) > 0).astype(int)
    df = df[["comment_text", "label"]].dropna()

    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=seed, stratify=df["label"]
    )
    logger.info("Train: %d rows (%.1f%% toxic) | Val: %d rows (%.1f%% toxic)",
                len(train_df), 100 * train_df["label"].mean(),
                len(val_df), 100 * val_df["label"].mean())

    return Dataset.from_pandas(train_df.reset_index(drop=True)), \
           Dataset.from_pandas(val_df.reset_index(drop=True))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa on the Jigsaw Toxic Comment dataset.")
    parser.add_argument("--data", required=True, help="Path to Jigsaw train.csv")
    parser.add_argument("--output", default="models/roberta-profanity")
    parser.add_argument("--base-model", default="roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    train_ds, val_ds = load_jigsaw_dataset(args.data)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch["comment_text"], truncation=True, max_length=args.max_length, padding="max_length")

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)

    training_args = TrainingArguments(
        output_dir="checkpoints/roberta-profanity",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    logger.info("Final validation metrics: %s", metrics)

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info("Fine-tuned model saved to %s", args.output)
    logger.info(
        "Target from spec: >90%% F1 on Jigsaw test set. Achieved F1: %.3f",
        metrics.get("eval_f1", float("nan")),
    )


if __name__ == "__main__":
    main()
