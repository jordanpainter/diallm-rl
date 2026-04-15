"""
scripts/gen_sub_files.py

Generates all SLURM .sub files for the Surrey AI cluster.
Run once from the repo root: python scripts/gen_sub_files.py
"""

import os

EXPERIMENTS = [
    "gemma", "llama", "qwen",
    "gemma_aus", "llama_aus", "qwen_aus",
    "gemma_ind", "llama_ind", "qwen_ind",
    "gemma_brit", "llama_brit", "qwen_brit",
]

ALGORITHMS = ["gspo", "grpo", "dpo"]

WANDB_PROJECTS = {
    # (algorithm, suffix)
    ("gspo", "all"):    "gspo-all",
    ("gspo", "narrow"): "gspo-narrow",
    ("grpo", "all"):    "grpo-all",
    ("grpo", "narrow"): "grpo-narrow",
    ("dpo",  "all"):    "dpo-all",
    ("dpo",  "narrow"): "dpo-narrow",
}

BROAD_EXPERIMENTS = {"gemma", "llama", "qwen"}


def get_wandb_project(algorithm, exp_name):
    suffix = "all" if exp_name in BROAD_EXPERIMENTS else "narrow"
    return WANDB_PROJECTS[(algorithm, suffix)]


def get_run_module(algorithm):
    return "src.train" if algorithm in ("gspo", "grpo") else "src.dpo"


def make_sub(algorithm, exp_name):
    job_name = f"{algorithm}_{exp_name}"
    wandb_project = get_wandb_project(algorithm, exp_name)
    wandb_name = f"{algorithm} {exp_name.replace('_', ' ')}"
    config_path = f"$REPO/configs/{algorithm}/{exp_name}.json"
    run_module = get_run_module(algorithm)

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=3090
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/fast/nobackup/scratch4weeks/%u/repos/diallm-rl/logs/%x.%N.%j.out
#SBATCH --error=/mnt/fast/nobackup/scratch4weeks/%u/repos/diallm-rl/logs/%x.%N.%j.err

set -euo pipefail

SCRATCH=/mnt/fast/nobackup/scratch4weeks/$USER
SIF="$SCRATCH/dialect-grpo-qwen-cu121.sif"
REPO="$SCRATCH/repos/diallm-rl"
CONFIG="{config_path}"

mkdir -p \\
  "$REPO/logs" \\
  "$SCRATCH/tmp" \\
  "$SCRATCH/wandb" \\
  "$SCRATCH/hf/hub" \\
  "$SCRATCH/hf/datasets" \\
  "$SCRATCH/hf/transformers" \\
  "$SCRATCH/runs"

export HF_HOME="$SCRATCH/hf"
export HUGGINGFACE_HUB_CACHE="$SCRATCH/hf/hub"
export HF_DATASETS_CACHE="$SCRATCH/hf/datasets"
export TRANSFORMERS_CACHE="$SCRATCH/hf/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=0

export WANDB_DIR="$SCRATCH/wandb"
export WANDB_ENTITY="jordanpainter"
export WANDB_PROJECT="{wandb_project}"
export WANDB_NAME="{wandb_name}"
export WANDB_MODE="online"

export WANDB_API_KEY="${{WANDB_API_KEY:-}}"
export HF_TOKEN="${{HF_TOKEN:-}}"

export TMPDIR="$SCRATCH/tmp"
export TEMP="$SCRATCH/tmp"
export TMP="$SCRATCH/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "HOST=$(hostname)"
echo "SIF=$SIF"
echo "REPO=$REPO"
echo "CONFIG=$CONFIG"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "WANDB_NAME=$WANDB_NAME"

nvidia-smi

apptainer exec --nv "$SIF" bash -lc "
    set -euo pipefail

    export HF_HOME='$HF_HOME'
    export HUGGINGFACE_HUB_CACHE='$HUGGINGFACE_HUB_CACHE'
    export HF_DATASETS_CACHE='$HF_DATASETS_CACHE'
    export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
    export HF_HUB_ENABLE_HF_TRANSFER='$HF_HUB_ENABLE_HF_TRANSFER'

    export WANDB_DIR='$WANDB_DIR'
    export WANDB_ENTITY='$WANDB_ENTITY'
    export WANDB_PROJECT='$WANDB_PROJECT'
    export WANDB_NAME='$WANDB_NAME'
    export WANDB_MODE='$WANDB_MODE'
    export WANDB_API_KEY='$WANDB_API_KEY'
    export HF_TOKEN='$HF_TOKEN'

    export TMPDIR='$TMPDIR'
    export TEMP='$TEMP'
    export TMP='$TMP'
    export TOKENIZERS_PARALLELISM='$TOKENIZERS_PARALLELISM'

    export PYTHONPATH='$REPO':\\${{PYTHONPATH:-}}

    cd '$REPO'

    echo 'PWD=' $(pwd)
    echo 'PYTHONPATH=' $PYTHONPATH
    python3 --version
    python3 -c \\"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count())\\"
    python3 -c \\"import os; print('CONFIG exists:', os.path.exists('$CONFIG'))\\"

    accelerate launch --num_processes 1 -m {run_module} -c '$CONFIG'
"
"""


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(repo_root, "sub")

    print("Generating sub files...")
    count = 0
    for algorithm in ALGORITHMS:
        for exp in EXPERIMENTS:
            out_path = os.path.join(sub_dir, algorithm, f"{exp}.sub")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", newline="\n") as f:
                f.write(make_sub(algorithm, exp))
            print(f"  wrote sub/{algorithm}/{exp}.sub")
            count += 1

    print(f"\nDone: {count} sub files written.")


if __name__ == "__main__":
    main()
