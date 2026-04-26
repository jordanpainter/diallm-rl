#!/bin/bash
#SBATCH --job-name=classifier_inference
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/fast/nobackup/scratch4weeks/%u/logs/%x.%N.%j.out
#SBATCH --error=/mnt/fast/nobackup/scratch4weeks/%u/logs/%x.%N.%j.err

set -euo pipefail

SCRATCH=/mnt/fast/nobackup/scratch4weeks/$USER
SIF="$SCRATCH/dialect-grpo-qwen-cu121.sif"
REPO="$SCRATCH/repos/diallm-rl"

mkdir -p "$SCRATCH/logs"

export HF_HOME="$SCRATCH/hf"
export HUGGINGFACE_HUB_CACHE="$SCRATCH/hf/hub"
export HF_DATASETS_CACHE="$SCRATCH/hf/datasets"
export TRANSFORMERS_CACHE="$SCRATCH/hf/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_TOKEN="${HF_TOKEN:-}"
export TMPDIR="$SCRATCH/tmp"
export TEMP="$SCRATCH/tmp"
export TMP="$SCRATCH/tmp"
export TOKENIZERS_PARALLELISM=false

echo "HOST=$(hostname)"
echo "REPO=$REPO"
nvidia-smi

apptainer exec --nv "$SIF" bash -lc "
    set -euo pipefail
    export HF_HOME='$HF_HOME'
    export HUGGINGFACE_HUB_CACHE='$HUGGINGFACE_HUB_CACHE'
    export HF_DATASETS_CACHE='$HF_DATASETS_CACHE'
    export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
    export HF_HUB_ENABLE_HF_TRANSFER='$HF_HUB_ENABLE_HF_TRANSFER'
    export HF_TOKEN='$HF_TOKEN'
    export TMPDIR='$TMPDIR'
    export TEMP='$TEMP'
    export TMP='$TMP'
    export TOKENIZERS_PARALLELISM='$TOKENIZERS_PARALLELISM'
    export PYTHONPATH='$REPO':\${PYTHONPATH:-}
    cd '$REPO'
    echo 'PWD=' \$(pwd)
    python3 --version
    python3 -c \"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\"
    python3 scripts/run_classifier_inference.py \
        --input  '$REPO/annotation_responses.jsonl' \
        --output '$REPO/annotation_responses_classified.jsonl'
"
