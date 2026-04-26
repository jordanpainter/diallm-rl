#!/bin/bash
#SBATCH --job-name=dialect_classifier
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/fast/nobackup/scratch4weeks/%u/logs/%x.%N.%j.out
#SBATCH --error=/mnt/fast/nobackup/scratch4weeks/%u/logs/%x.%N.%j.err

set -euo pipefail

SCRATCH=/mnt/fast/nobackup/scratch4weeks/$USER
SIF="$SCRATCH/dialect-grpo-qwen-cu121.sif"
REPO="$SCRATCH/repos/diallm-rl"

mkdir -p \
  "$SCRATCH/logs" \
  "$SCRATCH/tmp" \
  "$SCRATCH/hf/hub" \
  "$SCRATCH/hf/datasets" \
  "$SCRATCH/hf/transformers" \
  "$SCRATCH/models/dialect_classifier"

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
echo "SIF=$SIF"
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
    python3 scripts/train_dialect_classifier.py \
        --output_dir '$SCRATCH/models/dialect_classifier' \
        --epochs 5 \
        --batch_size 16 \
        --lr 1e-5 \
        --max_length 512 \
        --push_to_hub
"
