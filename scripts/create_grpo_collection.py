"""
scripts/create_grpo_collection.py

Creates a HuggingFace collection for the DialLM GRPO models and adds them to it.

Usage (run from repo root):
    HF_TOKEN=<token> python scripts/create_grpo_collection.py

Requires: HF_TOKEN env var set.
"""

import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("ERROR: HF_TOKEN is not set.")

NAMESPACE = "jordanpainter"
MODELS = [
    "jordanpainter/diallm-gemma-grpo-all",
    "jordanpainter/diallm-gemma-grpo-aus",
    "jordanpainter/diallm-gemma-grpo-brit",
    "jordanpainter/diallm-gemma-grpo-ind",
    "jordanpainter/diallm-llama-grpo-all",
    "jordanpainter/diallm-llama-grpo-aus",
    "jordanpainter/diallm-llama-grpo-brit",
    "jordanpainter/diallm-llama-grpo-ind",
    "jordanpainter/diallm-qwen-grpo-all",
    "jordanpainter/diallm-qwen-grpo-aus",
    "jordanpainter/diallm-qwen-grpo-brit",
    "jordanpainter/diallm-qwen-grpo-ind",
]

api = HfApi(token=HF_TOKEN)

print("Creating collection...")
collection = api.create_collection(
    namespace=NAMESPACE,
    title="DialLM GRPO",
    description="Group Relative Policy Optimization fine-tunes for DialLM across Gemma, Llama, and Qwen models, covering all dialect variants.",
    private=False,
)
print(f"Created: {collection.slug}")

for model_id in MODELS:
    api.add_collection_item(
        collection_slug=collection.slug,
        item_id=model_id,
        item_type="model",
    )
    print(f"  Added {model_id}")

print(f"\nDone! https://huggingface.co/collections/{collection.slug}")
