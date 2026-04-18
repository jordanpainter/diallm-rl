#!/bin/bash
# Upload all GRPO runs and gspo_qwen_all to HuggingFace.
# - Removes checkpoint* directories from each run first.
# - GRPO repos derived by replacing 'sft' with 'grpo' in model_id.
# - gspo_qwen_all uploaded explicitly to diallm-qwen-gspo-all.
#
# Usage (run from repo root on the cluster):
#   bash scripts/upload_grpo.sh
#
# Requires: HF_TOKEN env var set, hf (huggingface_hub CLI) available.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SCRATCH:-/mnt/fast/nobackup/scratch4weeks/$USER}"
RUNS_BASE="$SCRATCH/repos/diallm-rl/runs"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it before running this script."
    exit 1
fi

hf auth login --token "$HF_TOKEN" 2>/dev/null || true

upload_run() {
    local run_dir="$1"
    local hf_repo="$2"

    echo "  Run dir : $run_dir"
    echo "  HF repo : $hf_repo"

    if [[ ! -d "$run_dir" ]]; then
        echo "  SKIP: run dir not found"
        echo ""
        return
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

    echo "  Uploading to $hf_repo ..."
    HF_TOKEN="$HF_TOKEN" hf upload "$hf_repo" "$run_dir" . --repo-type model

    echo "  Done: $hf_repo"
    echo ""
}

# --- GRPO runs (all 12) ---
echo "=== GRPO Upload ==="
echo "Runs base: $RUNS_BASE"
echo ""

CONFIGS_DIR="$REPO_ROOT/configs/grpo"

for config in "$CONFIGS_DIR"/*.json; do
    name="$(basename "$config" .json)"
    output_dir="$(python3 -c "import json; d=json.load(open('$config')); print(d['trainer']['output_dir'])")"
    model_id="$(python3 -c "import json; d=json.load(open('$config')); print(d['model']['model_id'])")"
    hf_repo="${model_id/sft/grpo}"
    run_dir="$RUNS_BASE/$(basename "$output_dir")"

    echo "--- grpo/$name ---"
    upload_run "$run_dir" "$hf_repo"
done

echo "=== GRPO runs processed ==="
echo ""

# --- gspo_qwen_all ---
echo "=== gspo_qwen_all Upload ==="

GSPO_QWEN_CONFIG="$REPO_ROOT/configs/gspo/qwen.json"
output_dir="$(python3 -c "import json; d=json.load(open('$GSPO_QWEN_CONFIG')); print(d['trainer']['output_dir'])")"
run_dir="$RUNS_BASE/$(basename "$output_dir")"

echo "--- gspo/qwen ---"
upload_run "$run_dir" "jordanpainter/diallm-qwen-gspo-all"

echo "=== All runs processed ==="
