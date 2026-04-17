"""
scripts/create_gspo_collection.py

Creates a HuggingFace collection for the DialLM GSPO models and adds them to it.

Usage (run from anywhere on the cluster):
    python3 scripts/create_gspo_collection.py

Requires: HF_TOKEN env var set.
"""

import os
import requests
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("ERROR: HF_TOKEN is not set.")

NAMESPACE = "jordanpainter"
MODELS = [
    "jordanpainter/diallm-gemma-gspo-all",
    "jordanpainter/diallm-gemma-gspo-aus",
    "jordanpainter/diallm-gemma-gspo-brit",
    "jordanpainter/diallm-gemma-gspo-ind",
    "jordanpainter/diallm-llama-gspo-all",
    "jordanpainter/diallm-llama-gspo-aus",
    "jordanpainter/diallm-llama-gspo-brit",
    "jordanpainter/diallm-llama-gspo-ind",
    "jordanpainter/diallm-qwen-gspo-ind",
    # Add when ready:
    # "jordanpainter/diallm-qwen-gspo-all",
    # "jordanpainter/diallm-qwen-gspo-aus",
    # "jordanpainter/diallm-qwen-gspo-brit",
]

api = HfApi(token=HF_TOKEN)

print("Creating collection...")
collection = api.create_collection(
    namespace=NAMESPACE,
    title="DialLM GSPO",
    description="Group Supervised Policy Optimization fine-tunes for DialLM across Gemma, Llama, and Qwen models, covering all dialect variants.",
    private=False,
)
print(f"Created: {collection.slug}")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

for model_id in MODELS:
    r = requests.post(
        f"https://huggingface.co/api/collections/{collection.slug}/items",
        headers=HEADERS,
        json={"item": {"id": model_id, "type": "model"}},
    )
    r.raise_for_status()
    print(f"  Added {model_id}")

print(f"\nDone! https://huggingface.co/collections/{collection.slug}")
