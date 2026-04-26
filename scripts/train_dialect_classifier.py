"""
scripts/train_dialect_classifier.py

Fine-tunes DeBERTa-v3-base on deduplicated unswnlporg/BESSTIE as a 3-class
dialect classifier (en-AU / en-IN / en-UK).

Deduplication: texts appearing under multiple varieties are dropped entirely;
texts appearing under the same variety but multiple tasks are deduplicated to
one instance. No length filtering is applied. The cleaned dataset is re-split
80/10/10 with stratification.

Outputs:
  - Best checkpoint saved to --output_dir
  - label2id / id2label saved alongside
  - Final test-set metrics printed and saved to metrics.json
  - Model pushed to HF Hub as jordanpainter/diallm-dialect-classifier

Usage:
    python3 scripts/train_dialect_classifier.py
"""

import json
import os
import random
import argparse
import numpy as np
from collections import defaultdict, Counter
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import classification_report, accuracy_score
import torch


LABELS = ["en-AU", "en-IN", "en-UK"]
MODEL_NAME = "microsoft/deberta-v3-base"
HF_REPO = "jordanpainter/diallm-dialect-classifier"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="runs/dialect_classifier")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--push_to_hub", action="store_true", default=True)
    return p.parse_args()


def load_and_deduplicate(seed=42):
    """
    Load unswnlporg/BESSTIE, deduplicate by text, and return a clean Dataset
    with only 'text' and 'variety' columns.

    - Texts appearing under more than one variety are dropped entirely.
    - Texts appearing under one variety but multiple tasks are collapsed to
      a single instance.
    """
    print("Loading unswnlporg/BESSTIE...")
    raw = load_dataset("unswnlporg/BESSTIE")
    all_data = concatenate_datasets(list(raw.values()))
    print(f"Raw rows: {len(all_data)}")

    # Group rows by text
    text_to_rows = defaultdict(list)
    for i in range(len(all_data)):
        row = all_data[i]
        text_to_rows[row["text"]].append(row)

    clean = []
    dropped_cross_variety = 0
    for text, rows in text_to_rows.items():
        varieties = set(r["variety"] for r in rows)
        if len(varieties) > 1:
            # Ambiguous — text assigned to multiple varieties, drop entirely
            dropped_cross_variety += 1
            continue
        clean.append({"text": text, "variety": rows[0]["variety"]})

    print(f"Dropped (cross-variety conflicts): {dropped_cross_variety}")
    print(f"Clean unique texts: {len(clean)}")
    print("Variety distribution:", Counter(r["variety"] for r in clean))

    random.seed(seed)
    random.shuffle(clean)
    return Dataset.from_list(clean)


def stratified_split(dataset, label_col, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Re-split dataset with stratification by label."""
    random.seed(seed)

    label_to_indices = defaultdict(list)
    for i, label in enumerate(dataset[label_col]):
        label_to_indices[label].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in label_to_indices.items():
        random.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    return (
        dataset.select(train_idx),
        dataset.select(val_idx),
        dataset.select(test_idx),
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for l, i in label2id.items()}

    # --- Load and deduplicate ---
    all_data = load_and_deduplicate(seed=args.seed)

    # Cast variety to int label
    all_data = all_data.map(
        lambda x: {"label": label2id[x["variety"]]},
        remove_columns=["variety"],
    )

    # --- Stratified split ---
    train_ds, val_ds, test_ds = stratified_split(
        all_data, "label", train_ratio=0.8, val_ratio=0.1, seed=args.seed
    )
    print(f"\nSplit sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    for split_name, split_ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        counts = Counter(split_ds["label"])
        print(f"  {split_name}: { {id2label[k]: v for k, v in sorted(counts.items())} }")

    # --- Tokenise ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds   = val_ds.map(tokenize, batched=True)
    test_ds  = test_ds.map(tokenize, batched=True)

    for ds in [train_ds, val_ds, test_ds]:
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # --- Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True,
    )

    # --- Metrics ---
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    # --- Training ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=20,
        seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=HF_REPO if args.push_to_hub else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\nTraining...")
    trainer.train()

    # --- Test evaluation ---
    print("\nEvaluating on test set...")
    preds_output = trainer.predict(test_ds)
    logits = preds_output.predictions
    labels = preds_output.label_ids
    probs  = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds  = np.argmax(logits, axis=-1)

    report = classification_report(
        labels, preds,
        target_names=LABELS,
        digits=4,
        output_dict=True,
    )
    print(classification_report(labels, preds, target_names=LABELS, digits=4))

    metrics = {
        "test_accuracy": float(accuracy_score(labels, preds)),
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall":    report[label]["recall"],
                "f1":        report[label]["f1-score"],
                "support":   report[label]["support"],
            }
            for label in LABELS
        },
        "mean_confidence": float(probs.max(axis=-1).mean()),
    }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    label_map_path = os.path.join(args.output_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    if args.push_to_hub:
        print(f"\nPushing to {HF_REPO}...")
        trainer.push_to_hub()
        tokenizer.push_to_hub(HF_REPO)
        print("Done.")


if __name__ == "__main__":
    main()
