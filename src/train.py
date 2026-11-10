import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from labels import LABELS, LABEL2ID, ID2LABEL


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if len(item["tokens"]) != len(item["ner_tags"]):
                raise ValueError(f"{path}:{line_no}: tokens and ner_tags lengths differ")
            unknown = set(item["ner_tags"]) - set(LABELS)
            if unknown:
                raise ValueError(f"{path}:{line_no}: unknown labels {sorted(unknown)}")
            rows.append(item)
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def align_labels_with_tokens(tokenizer, examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
    )
    aligned = []
    for batch_index, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        previous_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(LABEL2ID[labels[word_id]])
            else:
                # Only the first subword is scored/trained. This avoids
                # accidentally duplicating entity labels across subwords.
                label_ids.append(-100)
            previous_word_id = word_id
        aligned.append(label_ids)
    tokenized["labels"] = aligned
    return tokenized


def compute_metrics(eval_prediction):
    logits, label_ids = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    true_predictions, true_labels = [], []
    for prediction_row, label_row in zip(predictions, label_ids):
        pred_seq, label_seq = [], []
        for pred_id, label_id in zip(prediction_row, label_row):
            if label_id == -100:
                continue
            pred_seq.append(ID2LABEL[int(pred_id)])
            label_seq.append(ID2LABEL[int(label_id)])
        true_predictions.append(pred_seq)
        true_labels.append(label_seq)
    return {
        "precision": precision_score(true_labels, true_predictions, zero_division=0),
        "recall": recall_score(true_labels, true_predictions, zero_division=0),
        "f1": f1_score(true_labels, true_predictions, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--validation", default="data/validation.jsonl")
    parser.add_argument("--output", default="models/phi-distilbert")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--test", default="data/test.jsonl")
    args = parser.parse_args()

    train_rows = read_jsonl(args.train)
    val_rows = read_jsonl(args.validation)
    test_rows = read_jsonl(args.test)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for word_ids() alignment.")

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)
    test_ds = Dataset.from_list(test_rows)
    train_ds = train_ds.map(
        lambda batch: align_labels_with_tokens(tokenizer, batch),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        lambda batch: align_labels_with_tokens(tokenizer, batch),
        batched=True,
        remove_columns=val_ds.column_names,
    )
    test_ds = test_ds.map(
    lambda batch: align_labels_with_tokens(tokenizer, batch),
    batched=True,
    remove_columns=test_ds.column_names,)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}")
    print(f"Exact parameter count: {parameter_count:,}")
    print(f"Under 1B parameters: {parameter_count <= 1_000_000_000}")

    training_args = TrainingArguments(
        output_dir="outputs/training",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print("Validation metrics:", trainer.evaluate())
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved fine-tuned model to: {args.output}")

    test_metrics = trainer.evaluate(
    eval_dataset=test_ds,
    metric_key_prefix="test")
    print("Independent test metrics:", test_metrics)


if __name__ == "__main__":
    main()
