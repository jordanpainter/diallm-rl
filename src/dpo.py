"""
src/dpo.py

DPO training script for dialectal English adaptation.

Uses the same config structure as train.py (model, peft, data, trainer sections).
The dataset must have columns: prompt, chosen, rejected.

Run:
    accelerate launch --num_processes=1 -m src.dpo -c configs/dpo/gemma.json
    accelerate launch --num_processes=1 -m src.dpo -c configs/dpo/gemma_brit.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset as hf_load_dataset
from datasets import load_from_disk
from huggingface_hub import login as hf_login, snapshot_download
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import DPOConfig, DPOTrainer

from src.formatting import build_chat_prompt


# =============================================================================
# Logging and utilities
# =============================================================================


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("diallm.dpo")


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def hard_trim_completion(text: str, stop_strings) -> str:
    if not text:
        return text
    cut: Optional[int] = None
    for s in stop_strings:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            cut = idx if cut is None else min(cut, idx)
    return text[:cut].rstrip() if cut is not None else text


def truncate_prompt_to_max_tokens(tokenizer, text: str, max_tokens: int) -> str:
    if not text:
        return text
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        truncation=True,
        max_length=max_tokens,
    )
    return tokenizer.decode(enc["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)


# =============================================================================
# Model + tokenizer loading
# =============================================================================


def _build_quant_config(mcfg: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    if not mcfg.get("load_in_4bit", False):
        return None
    compute_dtype = torch.bfloat16 if mcfg.get("bnb_4bit_compute_dtype", "bfloat16") == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=mcfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(mcfg.get("bnb_4bit_use_double_quant", True)),
    )


def load_policy_and_tokenizer(cfg: Dict[str, Any], logger: logging.Logger) -> Tuple[torch.nn.Module, Any, Any]:
    mcfg = cfg["model"]
    model_id = mcfg["model_id"]
    tok_id = mcfg.get("tokenizer_id", model_id)

    # Try AutoProcessor first — multimodal models (e.g. Gemma 3) need a Processor
    # as processing_class for DPOTrainer, not a bare tokenizer.
    try:
        processor = AutoProcessor.from_pretrained(tok_id, use_fast=True)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        logger.info("Loaded AutoProcessor for %s", tok_id)
    except Exception:
        processor = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
        tokenizer = processor
        logger.info("Loaded AutoTokenizer for %s", tok_id)

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    ensure_pad_token(tokenizer)

    quant_cfg = _build_quant_config(mcfg)
    gpu_count = torch.cuda.device_count()
    device_map = "auto" if gpu_count > 1 else {"": get_local_rank()}
    logger.info("cuda_device_count=%s | device_map=%s", gpu_count, device_map)

    dtype = torch.bfloat16 if cfg.get("trainer", {}).get("bf16", False) else None
    from_pretrained_kwargs: Dict[str, Any] = dict(
        low_cpu_mem_usage=True,
        quantization_config=quant_cfg,
        device_map=device_map,
        return_dict=True,
    )
    if dtype is not None:
        from_pretrained_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **from_pretrained_kwargs)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = not bool(cfg.get("trainer", {}).get("gradient_checkpointing", False))

    return model, tokenizer, processor


# =============================================================================
# Dataset loading
# =============================================================================


def load_dataset(cfg_data: Dict[str, Any], logger: logging.Logger):
    dataset_path = cfg_data.get("dataset_path")
    dataset_id = cfg_data.get("dataset_id")
    split = cfg_data.get("dataset_split", "train")

    if dataset_path:
        logger.info("Loading dataset from disk: %s", dataset_path)
        ds_any = load_from_disk(dataset_path)
    elif dataset_id:
        logger.info("Downloading dataset snapshot from Hub: %s", dataset_id)
        local_path = snapshot_download(repo_id=dataset_id, repo_type="dataset")
        logger.info("Downloaded to %s", local_path)
        try:
            ds_any = load_from_disk(local_path)
        except Exception:
            logger.info("Not a save_to_disk snapshot, falling back to load_dataset")
            ds_any = hf_load_dataset(dataset_id, split=split)
    else:
        raise ValueError("Expected either data.dataset_path or data.dataset_id in config.")

    from datasets import Dataset, DatasetDict
    if isinstance(ds_any, Dataset):
        logger.info("Loaded Dataset with columns: %s", ds_any.column_names)
        return ds_any

    if isinstance(ds_any, DatasetDict):
        ds = ds_any[split] if split in ds_any else ds_any[list(ds_any.keys())[0]]
        logger.info("Loaded split '%s' with columns: %s", split, ds.column_names)
        return ds

    raise TypeError(f"Unsupported dataset type: {type(ds_any)}")


def build_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return build_chat_prompt(tokenizer, system_prompt, user_prompt)


# =============================================================================
# DPOConfig builder
# =============================================================================


def build_dpo_config(cfg: Dict[str, Any], logger: logging.Logger, tokenizer) -> DPOConfig:
    import inspect
    raw_args = dict(cfg.get("trainer", {}))

    raw_args.setdefault("beta", 0.1)
    raw_args.setdefault("loss_type", "sigmoid")

    sig = inspect.signature(DPOConfig.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    filtered_args = {k: v for k, v in raw_args.items() if k in allowed}
    dropped = sorted(set(raw_args.keys()) - set(filtered_args.keys()))
    if dropped:
        logger.warning("Dropping unsupported DPOConfig args: %s", dropped)

    logger.info(
        "DPO config | beta=%s loss_type=%s output_dir=%s",
        filtered_args.get("beta"),
        filtered_args.get("loss_type"),
        filtered_args.get("output_dir"),
    )

    return DPOConfig(**filtered_args)


# =============================================================================
# Main entrypoint
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    logger = setup_logging()
    torch.set_float32_matmul_precision("high")

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = json.load(f)

    logger.info("Config: %s", args.config)

    hf_token = cfg.get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        hf_login(token=hf_token)
        logger.info("Logged into Hugging Face Hub")

    seed = int(cfg.get("data", {}).get("seed", 42))
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model, tokenizer, processor = load_policy_and_tokenizer(cfg, logger)

    dcfg = cfg["data"]
    ds = load_dataset(dcfg, logger)

    for col in ["prompt", "chosen", "rejected"]:
        if col not in ds.column_names:
            raise ValueError(f"Dataset missing required column '{col}'. Found: {ds.column_names}")

    ds = ds.train_test_split(test_size=float(dcfg.get("test_size", 0.02)), seed=seed)
    train_ds, eval_ds = ds["train"], ds["test"]
    logger.info("Train/Eval: %d / %d", len(train_ds), len(eval_ds))

    n_tr = int(dcfg.get("smoke_subset_train", 0) or 0)
    n_ev = int(dcfg.get("smoke_subset_eval", 0) or 0)
    if n_tr > 0:
        train_ds = train_ds.select(range(min(n_tr, len(train_ds))))
    if n_ev > 0:
        eval_ds = eval_ds.select(range(min(n_ev, len(eval_ds))))

    system_prompt = dcfg.get("system_prompt", "") or ""
    max_prompt_len = max(32, int(cfg.get("trainer", {}).get("max_prompt_length", 2048)))
    tokenizer.model_max_length = max_prompt_len

    stop_strings = ["\nUser:", "\nAssistant:", "\n### User:", "\n### Assistant:", "\n<|user|>", "\n<|assistant|>"]

    def map_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        ex["prompt"] = build_prompt(tokenizer, system_prompt, ex["prompt"])
        ex["prompt"] = truncate_prompt_to_max_tokens(tokenizer, ex["prompt"], max_prompt_len)
        ex["chosen"] = hard_trim_completion(ex["chosen"], stop_strings)
        ex["rejected"] = hard_trim_completion(ex["rejected"], stop_strings)
        return ex

    train_ds = train_ds.map(map_fn)
    eval_ds = eval_ds.map(map_fn)

    # Keep only the columns DPO needs
    keep_cols = {"prompt", "chosen", "rejected"}
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c not in keep_cols])

    # Gemma 3 (and other vision models): TRL's DPOTrainer detects the Gemma3Processor
    # and expects an "images" column in the dataset. Add a None column so it falls
    # through to text-only processing.
    if hasattr(processor, "tokenizer"):
        train_ds = train_ds.add_column("images", [None] * len(train_ds))
        eval_ds = eval_ds.add_column("images", [None] * len(eval_ds))
        logger.info("Added dummy images column for vision processor (%s)", type(processor).__name__)

    peft_cfg = cfg.get("peft", {})
    lora_cfg = None
    if bool(peft_cfg.get("enabled", True)):
        lora_cfg = LoraConfig(
            r=int(peft_cfg.get("r", 8)),
            lora_alpha=int(peft_cfg.get("lora_alpha", 16)),
            lora_dropout=float(peft_cfg.get("lora_dropout", 0.05)),
            target_modules=list(peft_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
            bias=str(peft_cfg.get("bias", "none")),
            task_type=str(peft_cfg.get("task_type", "CAUSAL_LM")),
        )

    dpo_args = build_dpo_config(cfg, logger, tokenizer)

    logger.info("model device=%s dtype=%s", next(model.parameters()).device, next(model.parameters()).dtype)

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        peft_config=lora_cfg,
    )

    trainer.train()

    out_dir = trainer.args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info("Saved to %s", out_dir)


if __name__ == "__main__":
    main()
