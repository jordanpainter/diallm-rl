"""
scripts/run_feature_inference.py

Runs the dialect feature identifier (srirag/feature-identifier) over every
response in annotation_responses_classified.jsonl. For dialect-specific
variants (aus/brit/ind), only the relevant feature subset is recorded.
For 'all' and 'base' variants, all 135 features are recorded.

Output adds to each record:
  feature_indices  — list of int, the feature indices scored
  feature_probs    — list of float (sigmoid), same order as feature_indices
  feature_mean     — float, mean activation across active features

Usage:
    python3 scripts/run_feature_inference.py \
        --input  annotation_responses_classified.jsonl \
        --output annotation_responses_features.jsonl
"""

import argparse
import json
import sys
import torch
from rewards.dialect_feature_model import MultiheadDialectFeatureModel
from transformers import AutoTokenizer

FEATURE_IDENTIFIER = "srirag/feature-identifier"
BATCH_SIZE = 64
MAX_LENGTH = 256

AUS_FEATURES  = [2, 13, 14, 18, 42, 50, 54, 55, 73, 74, 82, 92, 106, 111, 124, 125, 126, 128, 132, 134]
BRIT_FEATURES = [13, 14, 18, 19, 27, 31, 34, 36, 37, 38, 48, 54, 72, 73, 74, 82, 99, 101, 102, 104, 107, 110, 115, 116, 124, 125, 126, 130, 133, 134]
IND_FEATURES  = [18, 20, 21, 22, 23, 24, 25, 26, 27, 31, 34, 36, 37, 38, 40, 41, 43, 48, 50, 51, 54, 55, 67, 88, 92, 107, 110, 111, 112, 118, 120, 123, 127, 129, 130, 131, 132, 134]
ALL_FEATURES  = list(range(135))

VARIANT_FEATURES = {
    "aus":  AUS_FEATURES,
    "brit": BRIT_FEATURES,
    "ind":  IND_FEATURES,
    "all":  ALL_FEATURES,
    "base": ALL_FEATURES,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="annotation_responses_classified.jsonl")
    p.add_argument("--output", default="annotation_responses_features.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading feature identifier from {FEATURE_IDENTIFIER}...")
    tokenizer = AutoTokenizer.from_pretrained(FEATURE_IDENTIFIER)
    model = MultiheadDialectFeatureModel.from_pretrained(FEATURE_IDENTIFIER)
    model.to(device)
    model.eval()
    print("Loaded.")

    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"Records: {len(records)}")

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
            logits = model(**enc).logits          # (batch, 135)
        probs = torch.sigmoid(logits).cpu()       # (batch, 135)

        for record, prob_row in zip(batch, probs.tolist()):
            variant  = record["variant"]
            indices  = VARIANT_FEATURES.get(variant, ALL_FEATURES)
            sub_probs = [round(prob_row[idx], 6) for idx in indices]

            out = dict(record)
            out["feature_indices"] = indices
            out["feature_probs"]   = sub_probs
            out["feature_mean"]    = round(sum(sub_probs) / len(sub_probs), 6)
            results.append(out)

        if (i // BATCH_SIZE) % 5 == 0:
            print(f"  {i + len(batch)}/{len(records)}")

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. Written to {args.output}")


if __name__ == "__main__":
    main()
