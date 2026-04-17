#!/bin/bash
# Upload ready GSPO runs to HuggingFace.
# - Removes checkpoint* directories from each run first.
# - Skips qwen_all, qwen_aus, qwen_brit (not yet ready).
#
# Usage (run from repo root on the cluster):
#   bash scripts/upload_gspo.sh
#
# Requires: HF_TOKEN env var set, hf (huggingface_hub CLI) available.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIGS_DIR="$REPO_ROOT/configs/gspo"
SCRATCH="${SCRATCH:-/mnt/fast/nobackup/scratch4weeks/$USER}"
RUNS_BASE="$SCRATCH/repos/diallm-rl/runs"
NAMESPACE="jordanpainter"

# Runs not yet ready — skip these
SKIP=("qwen" "qwen_aus" "qwen_brit")

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it before running this script."
    exit 1
fi

hf auth login --token "$HF_TOKEN" 2>/dev/null || true

echo "=== GSPO Upload Script ==="
echo "Runs base: $RUNS_BASE"
echo ""

for config in "$CONFIGS_DIR"/*.json; do
    name="$(basename "$config" .json)"

    # Skip unready runs
    skip=false
    for s in "${SKIP[@]}"; do
        if [[ "$name" == "$s" ]]; then
            skip=true
            break
        fi
    done
    if $skip; then
        echo "--- $name --- SKIPPED (not ready)"
        echo ""
        continue
    fi

    # Derive model and dialect from config name (e.g. gemma_aus -> gemma, aus)
    model="$(echo "$name" | cut -d_ -f1)"
    dialect_suffix="$(echo "$name" | cut -s -d_ -f2-)"
    dialect="${dialect_suffix:-all}"

    hf_repo="$NAMESPACE/diallm-${model}-gspo-${dialect}"

    output_dir="$(python3 -c "import json; d=json.load(open('$config')); print(d['trainer']['output_dir'])")"
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

echo "=== All ready GSPO runs processed ==="
