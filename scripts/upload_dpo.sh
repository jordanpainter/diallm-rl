#!/bin/bash
# Upload all DPO runs to HuggingFace.
# - Removes checkpoint* directories from each run first.
# - Derives the HF repo name from the config (replaces 'sft' with 'dpo' in model_id).
#
# Usage (run from repo root on the cluster):
#   bash scripts/upload_dpo.sh
#
# Requires: HF_TOKEN env var set, hf (huggingface_hub CLI) available.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIGS_DIR="$REPO_ROOT/configs/dpo"
SCRATCH="${SCRATCH:-/mnt/fast/nobackup/scratch4weeks/$USER}"
RUNS_BASE="$SCRATCH/repos/diallm-rl/runs"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it before running this script."
    exit 1
fi

hf auth login --token "$HF_TOKEN" 2>/dev/null || true

echo "=== DPO Upload Script ==="
echo "Runs base: $RUNS_BASE"
echo ""

for config in "$CONFIGS_DIR"/*.json; do
    name="$(basename "$config" .json)"

    # Parse output_dir and model_id from config
    output_dir="$(python3 -c "import json; d=json.load(open('$config')); print(d['trainer']['output_dir'])")"
    model_id="$(python3 -c "import json; d=json.load(open('$config')); print(d['model']['model_id'])")"

    # Derive HF repo name: replace 'sft' with 'dpo' in the model_id repo part
    hf_repo="${model_id/sft/dpo}"

    run_dir="$RUNS_BASE/$(basename "$output_dir")"

    echo "--- $name ---"
    echo "  Run dir : $run_dir"
    echo "  HF repo : $hf_repo"

    if [[ ! -d "$run_dir" ]]; then
        echo "  SKIP: run dir not found"
        echo ""
        continue
    fi

    # Remove checkpoint* directories/files
    checkpoint_count=0
    while IFS= read -r -d '' item; do
        echo "  Removing: $item"
        rm -rf "$item"
        ((checkpoint_count++)) || true
    done < <(find "$run_dir" -maxdepth 1 -name "checkpoint*" -print0)

    if [[ $checkpoint_count -eq 0 ]]; then
        echo "  (no checkpoints to remove)"
    else
        echo "  Removed $checkpoint_count checkpoint(s)"
    fi

    # Upload to HuggingFace
    echo "  Uploading to $hf_repo ..."
    HF_TOKEN="$HF_TOKEN" hf upload "$hf_repo" "$run_dir" . --repo-type model

    echo "  Done: $hf_repo"
    echo ""
done

echo "=== All DPO runs processed ==="
