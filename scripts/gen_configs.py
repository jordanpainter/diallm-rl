"""
scripts/gen_configs.py

Generates all 36 training configs (12 GSPO + 12 GRPO + 12 DPO).
Run once from the repo root: python scripts/gen_configs.py
"""

import copy
import json
import os

WANDB_ENTITY = "jordanpainter"

AUS_FEATURES  = [2, 13, 14, 18, 42, 50, 54, 55, 73, 74, 82, 92, 106, 111, 124, 125, 126, 128, 132, 134]
BRIT_FEATURES = [13, 14, 18, 19, 27, 31, 34, 36, 37, 38, 48, 54, 72, 73, 74, 82, 99, 101, 102, 104, 107, 110, 115, 116, 124, 125, 126, 130, 133, 134]
IND_FEATURES  = [18, 20, 21, 22, 23, 24, 25, 26, 27, 31, 34, 36, 37, 38, 40, 41, 43, 48, 50, 51, 54, 55, 67, 88, 92, 107, 110, 111, 112, 118, 120, 123, 127, 129, 130, 131, 132, 134]

COMMON_PEFT = {
    "enabled": False,
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

COMMON_NORMALIZATION = {
    "method": "running_zscore",
    "clip_z": 5.0,
    "eps": 1e-6,
    "beta": 0.99,
    "warmup_steps": 50,
}

COMMON_COMET = {
    "comet_model_name": "Unbabel/wmt22-comet-da",
    "comet_batch_size": 2,
    "comet_force_cpu": True,
    "sim_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "verbose_examples": 2,
}

COMMON_TRAINER_BASE = {
    "ddp_find_unused_parameters": False,
    "length_safety_margin": 16,
    "report_to": "wandb",
    "max_steps": 5000,
    "warmup_ratio": 0.05,
    "weight_decay": 0.0,
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "steps_per_generation": 4,
    "logging_steps": 1,
    "num_generations": 4,
    "temperature": 0.7,
    "top_p": 0.95,
    "generation_kwargs": {"do_sample": True},
    "beta": 0.02,
    "epsilon": 0.0003,
    "epsilon_high": 0.0004,
    "loss_type": "grpo",
    "scale_rewards": False,
    "max_grad_norm": 1.0,
}

COMMON_TRAINER_GSPO = {
    **COMMON_TRAINER_BASE,
    "importance_sampling_level": "sequence",
}

COMMON_DPO_TRAINER_BASE = {
    "ddp_find_unused_parameters": False,
    "report_to": "wandb",
    "max_steps": 5000,
    "learning_rate": None,  # filled per model
    "warmup_ratio": 0.05,
    "weight_decay": 0.0,
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "logging_steps": 1,
    "beta": 0.1,
    "loss_type": "sigmoid",
    "max_length": 2048,
    "max_prompt_length": 1024,
    "max_grad_norm": 1.0,
    "precompute_ref_log_probs": True,
}

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # --- Broad / Implicit (Thread 1) ---
    {
        "name": "gemma",
        "wandb_project_suffix": "all",
        "model_id": "jordanpainter/DialLM-Gemma-sft-all",
        "tokenizer_id": "google/gemma-3-4b-it",
        "dataset_id": "jordanpainter/dialect-gemma-base-all",
        "dpo_dataset_id": "jordanpainter/dialect-preferences",
        "feature_indices": None,
        "reward_weights": {"dialect": 0.8, "comet": 0.1, "cosine": 0.1},
        "lr": 5e-6,
        "max_completion_length": 256,
        "max_prompt_length": 2048,
        "optim": None,
    },
    {
        "name": "llama",
        "wandb_project_suffix": "all",
        "model_id": "jordanpainter/DialLM-Llama-sft-all",
        "tokenizer_id": "meta-llama/Llama-3.1-8B",
        "dataset_id": "jordanpainter/dialect-llama-base-all",
        "dpo_dataset_id": "jordanpainter/dialect-preferences",
        "feature_indices": None,
        "reward_weights": {"dialect": 0.8, "comet": 0.1, "cosine": 0.1},
        "lr": 2e-6,
        "max_completion_length": 256,
        "max_prompt_length": 2048,
        "optim": "adamw_bnb_8bit",
    },
    {
        "name": "qwen",
        "wandb_project_suffix": "all",
        "model_id": "jordanpainter/DialLM-Qwen-sft-all",
        "tokenizer_id": "Qwen/Qwen3-8B",
        "dataset_id": "jordanpainter/dialect-qwen-base-all",
        "dpo_dataset_id": "jordanpainter/dialect-preferences",
        "feature_indices": None,
        "reward_weights": {"dialect": 0.8, "comet": 0.1, "cosine": 0.1},
        "lr": 2e-6,
        "max_completion_length": 192,
        "max_prompt_length": 1024,
        "optim": "adamw_bnb_8bit",
    },
    # --- AusE / Explicit ---
    {
        "name": "gemma_aus",
        "wandb_project_suffix": "narrow",
        "model_id": "jordanpainter/diallm-gemma-sft-aus",
        "tokenizer_id": "google/gemma-3-4b-it",
        "dataset_id": "jordanpainter/alignment-australian-final",
        "feature_indices": AUS_FEATURES,
        "reward_weights": {"dialect": 0.8, "comet": 0.1, "cosine": 0.1},
        "lr": 5e-6,
        "max_completion_length": 256,
        "max_prompt_length": 2048,
        "optim": None,
    },
    {
        "name": "llama_aus",
        "wandb_project_suffix": "narrow",
        "model_id": "jordanpainter/diallm-llama-sft-aus",
        "tokenizer_id": "meta-llama/Llama-3.1-8B",
        "dataset_id": "jordanpainter/alignment-australian-final",
        "feature_indices": AUS_FEATURES,
        "reward_weights": {"dialect": 0.8, "comet": 0.1, "cosine": 0.1},
        "lr": 2e-6,
        "max_completion_length": 256,
        "max_prompt_length": 2048,
        "optim": "adamw_bnb_8bit",
    },
    {
        "name": "qwen_aus",
        "wandb_project_suffix": "narrow",
        "model_id": "jordanpainter/diallm-qwen-sft-aus",
        "tokenizer_id": "Qwen/Qwen3-8B",
        "dataset_id": "jordanpainter/alignment-australian-final",
        "feature_indices": AUS_FEATURES,
        "reward_weights": {"dialect": 0.8, "comet": 0.1, "cosine": 0.1},
        "lr": 2e-6,
        "max_completion_length": 192,
        "max_prompt_length": 1024,
        "optim": "adamw_bnb_8bit",
    },
    # --- IndE / Explicit ---
    {
        "name": "gemma_ind",
        "wandb_project_suffix": "narrow",
        "model_id": "jordanpainter/diallm-gemma-sft-ind",
        "tokenizer_id": "google/gemma-3-4b-it",
        "dataset_id": "jordanpainter/alignment-indian-final",
        "feature_indices": IND_FEATURES,
        "reward_weights": {"dialect": 0.8, "comet": 0.1, "cosine": 0.1},
        "lr": 5e-6,
        "max_completion_length": 256,
        "max_prompt_length": 2048,
        "optim": None,
    },
    {
        "name": "llama_ind",
        "wandb_project_suffix": "narrow",
        "model_id": "jordanpainter/diallm-llama-sft-ind",
        "tokenizer_id": "meta-llama/Llama-3.1-8B",
        "dataset_id": "jordanpainter/alignment-indian-final",
        "feature_indices": IND_FEATURES,
        "reward_weights": {"dialect": 0.8, "comet": 0.1, "cosine": 0.1},
        "lr": 2e-6,
        "max_completion_length": 256,
        "max_prompt_length": 2048,
        "optim": "adamw_bnb_8bit",
    },
    {
        "name": "qwen_ind",
        "wandb_project_suffix": "narrow",
        "model_id": "jordanpainter/diallm-qwen-sft-ind",
        "tokenizer_id": "Qwen/Qwen3-8B",
        "dataset_id": "jordanpainter/alignment-indian-final",
        "feature_indices": IND_FEATURES,
        "reward_weights": {"dialect": 0.8, "comet": 0.1, "cosine": 0.1},
        "lr": 2e-6,
        "max_completion_length": 192,
        "max_prompt_length": 1024,
        "optim": "adamw_bnb_8bit",
    },
    # --- NorthE (Brit) / Explicit ---
    {
        "name": "gemma_brit",
        "wandb_project_suffix": "narrow",
        "model_id": "jordanpainter/diallm-gemma-sft-brit",
        "tokenizer_id": "google/gemma-3-4b-it",
        "dataset_id": "jordanpainter/alignment-british-final",
        "feature_indices": BRIT_FEATURES,
        "reward_weights": {"dialect": 0.8, "comet": 0.1, "cosine": 0.1},
        "lr": 5e-6,
        "max_completion_length": 256,
        "max_prompt_length": 2048,
        "optim": None,
    },
    {
        "name": "llama_brit",
        "wandb_project_suffix": "narrow",
        "model_id": "jordanpainter/diallm-llama-sft-brit",
        "tokenizer_id": "meta-llama/Llama-3.1-8B",
        "dataset_id": "jordanpainter/alignment-british-final",
        "feature_indices": BRIT_FEATURES,
        "reward_weights": {"dialect": 0.8, "comet": 0.1, "cosine": 0.1},
        "lr": 2e-6,
        "max_completion_length": 256,
        "max_prompt_length": 2048,
        "optim": "adamw_bnb_8bit",
    },
    {
        "name": "qwen_brit",
        "wandb_project_suffix": "narrow",
        "model_id": "jordanpainter/diallm-qwen-sft-brit",
        "tokenizer_id": "Qwen/Qwen3-8B",
        "dataset_id": "jordanpainter/alignment-british-final",
        "feature_indices": BRIT_FEATURES,
        "reward_weights": {"dialect": 0.8, "comet": 0.1, "cosine": 0.1},
        "lr": 2e-6,
        "max_completion_length": 192,
        "max_prompt_length": 1024,
        "optim": "adamw_bnb_8bit",
    },
]


def build_rl_config(exp, algorithm):
    """Build a GSPO or GRPO config dict."""
    suffix = exp["wandb_project_suffix"]
    wandb_project = f"{algorithm}-{suffix}"
    display_name = exp["name"].replace("_", " ")

    rewards = {
        "weights": exp["reward_weights"],
        "normalization": COMMON_NORMALIZATION,
        **COMMON_COMET,
    }
    if exp["feature_indices"] is not None:
        rewards["dialect_feature_indices"] = exp["feature_indices"]

    trainer = copy.deepcopy(COMMON_TRAINER_GSPO if algorithm == "gspo" else COMMON_TRAINER_BASE)
    trainer["output_dir"] = f"runs/{algorithm}_{exp['name']}"
    trainer["learning_rate"] = exp["lr"]
    trainer["max_completion_length"] = exp["max_completion_length"]
    trainer["max_prompt_length"] = exp["max_prompt_length"]
    if exp["optim"]:
        trainer["optim"] = exp["optim"]

    cfg = {
        "algorithm": algorithm,
        "wandb": {
            "project": wandb_project,
            "entity": WANDB_ENTITY,
        },
        "model": {
            "model_id": exp["model_id"],
            "tokenizer_id": exp["tokenizer_id"],
            "load_in_4bit": False,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        },
        "peft": COMMON_PEFT,
        "data": {
            "dataset_id": exp["dataset_id"],
            "dataset_split": "train",
            "seed": 42,
            "test_size": 0.02,
            "system_prompt": "You are a helpful assistant.",
        },
        "rewards": rewards,
        "trainer": trainer,
    }
    return cfg


def build_dpo_config(exp):
    """Build a DPO config dict."""
    suffix = exp["wandb_project_suffix"]
    wandb_project = f"dpo-{suffix}"

    trainer = copy.deepcopy(COMMON_DPO_TRAINER_BASE)
    trainer["output_dir"] = f"runs/dpo_{exp['name']}"
    trainer["learning_rate"] = exp["lr"]
    trainer["max_prompt_length"] = exp["max_prompt_length"]
    trainer["max_length"] = exp["max_prompt_length"] + exp["max_completion_length"]
    if exp["optim"]:
        trainer["optim"] = exp["optim"]

    cfg = {
        "algorithm": "dpo",
        "wandb": {
            "project": wandb_project,
            "entity": WANDB_ENTITY,
        },
        "model": {
            "model_id": exp["model_id"],
            "tokenizer_id": exp["tokenizer_id"],
            "load_in_4bit": False,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        },
        "peft": COMMON_PEFT,
        "data": {
            "dataset_id": exp.get("dpo_dataset_id", exp["dataset_id"]),
            "dataset_split": "train",
            "seed": 42,
            "test_size": 0.02,
            "system_prompt": "You are a helpful assistant.",
        },
        "trainer": trainer,
    }
    return cfg


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {path}")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configs_dir = os.path.join(repo_root, "configs")

    print("Generating configs...")
    for exp in EXPERIMENTS:
        name = exp["name"]
        write_json(os.path.join(configs_dir, "gspo", f"{name}.json"), build_rl_config(exp, "gspo"))
        write_json(os.path.join(configs_dir, "grpo", f"{name}.json"), build_rl_config(exp, "grpo"))
        write_json(os.path.join(configs_dir, "dpo",  f"{name}.json"), build_dpo_config(exp))

    print(f"\nDone: {len(EXPERIMENTS) * 3} configs written.")


if __name__ == "__main__":
    main()
