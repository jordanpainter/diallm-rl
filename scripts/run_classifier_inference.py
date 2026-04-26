"""
scripts/run_classifier_inference.py

Runs jordanpainter/diallm-dialect-classifier over every response in
annotation_responses.jsonl, appending classifier probabilities for
en-AU, en-IN, and en-UK to each record.

Output is a new JSONL with the same fields plus:
  classifier_en-AU, classifier_en-IN, classifier_en-UK  (float, sum to 1)
  classifier_pred  (string, argmax label)

Usage:
    python3 scripts/run_classifier_inference.py \
        --input  annotation_responses.jsonl \
        --output annotation_responses_classified.jsonl
"""

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CLASSIFIER_REPO = "jordanpainter/diallm-dialect-classifier"
LABELS = ["en-AU", "en-IN", "en-UK"]
BATCH_SIZE = 64
MAX_LENGTH = 512


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="annotation_responses.jsonl")
    p.add_argument("--output", default="annotation_responses_classified.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading classifier from {CLASSIFIER_REPO}...")
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_REPO,
        dtype=torch.float16 if device == "cuda:0" else torch.float32,
    ).to(device)
    model.eval()
    print("Loaded.")

    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"Records to classify: {len(records)}")

    results = []
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        texts = [r["response"] for r in batch]

        enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()

        for record, prob in zip(batch, probs):
            out = dict(record)
            for label, p in zip(LABELS, prob):
                out[f"classifier_{label}"] = round(p, 6)
            out["classifier_pred"] = LABELS[prob.index(max(prob))]
            results.append(out)

        if (i // BATCH_SIZE) % 5 == 0:
            print(f"  {i + len(batch)}/{len(records)}")

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. Written to {args.output}")


if __name__ == "__main__":
    main()
