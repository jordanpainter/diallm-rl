"""
scripts/run_annotation_inference.py

Runs all DialLM models (base, CPT, SFT, DPO, GRPO, GSPO) against the 25
annotation prompts using greedy decoding. Writes one JSONL record per
model+prompt, with a live log to stdout and a dedicated log file.

Usage:
    python3 scripts/run_annotation_inference.py \
        --prompts annotation_prompt_candidates.json \
        --output  annotation_responses.jsonl \
        --log     annotation_inference.log
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Model registry
# Each entry: model_id, stage, family, variant, template_source (if needed)
# template_source: HF repo to pull the chat template from when the model's
#   own tokenizer doesn't have one (base models, CPT).
# ---------------------------------------------------------------------------
INSTRUCT_TEMPLATES = {
    "llama":  "meta-llama/Llama-3.1-8B-Instruct",
    "qwen":   "Qwen/Qwen3-8B",          # Qwen3-8B already has a template
    "gemma":  "google/gemma-3-4b-it",
}

MODELS = [
    # --- Base ---
    {"model_id": "meta-llama/Llama-3.1-8B",  "stage": "base",  "family": "llama", "variant": "base", "template_source": "meta-llama/Llama-3.1-8B-Instruct"},
    {"model_id": "Qwen/Qwen3-8B",             "stage": "base",  "family": "qwen",  "variant": "base", "template_source": None},
    {"model_id": "google/gemma-3-4b-it",      "stage": "base",  "family": "gemma", "variant": "base", "template_source": None},

    # --- CPT ---
    {"model_id": "jordanpainter/diallm-gemma-cpt", "stage": "cpt", "family": "gemma", "variant": "all", "template_source": "google/gemma-3-4b-it"},
    {"model_id": "jordanpainter/diallm-llama-cpt", "stage": "cpt", "family": "llama", "variant": "all", "template_source": "meta-llama/Llama-3.1-8B-Instruct"},
    {"model_id": "jordanpainter/diallm-qwen-cpt",  "stage": "cpt", "family": "qwen",  "variant": "all", "template_source": "Qwen/Qwen3-8B"},

    # --- SFT ---
    {"model_id": "jordanpainter/diallm-gemma-sft-all",  "stage": "sft", "family": "gemma", "variant": "all"},
    {"model_id": "jordanpainter/diallm-gemma-sft-aus",  "stage": "sft", "family": "gemma", "variant": "aus"},
    {"model_id": "jordanpainter/diallm-gemma-sft-brit", "stage": "sft", "family": "gemma", "variant": "brit"},
    {"model_id": "jordanpainter/diallm-gemma-sft-ind",  "stage": "sft", "family": "gemma", "variant": "ind"},
    {"model_id": "jordanpainter/diallm-llama-sft-all",  "stage": "sft", "family": "llama", "variant": "all"},
    {"model_id": "jordanpainter/diallm-llama-sft-aus",  "stage": "sft", "family": "llama", "variant": "aus"},
    {"model_id": "jordanpainter/diallm-llama-sft-brit", "stage": "sft", "family": "llama", "variant": "brit"},
    {"model_id": "jordanpainter/diallm-llama-sft-ind",  "stage": "sft", "family": "llama", "variant": "ind"},
    {"model_id": "jordanpainter/diallm-qwen-sft-all",   "stage": "sft", "family": "qwen",  "variant": "all"},
    {"model_id": "jordanpainter/diallm-qwen-sft-aus",   "stage": "sft", "family": "qwen",  "variant": "aus"},
    {"model_id": "jordanpainter/diallm-qwen-sft-brit",  "stage": "sft", "family": "qwen",  "variant": "brit"},
    {"model_id": "jordanpainter/diallm-qwen-sft-ind",   "stage": "sft", "family": "qwen",  "variant": "ind"},

    # --- DPO ---
    {"model_id": "jordanpainter/diallm-gemma-dpo-all",  "stage": "dpo", "family": "gemma", "variant": "all"},
    {"model_id": "jordanpainter/diallm-gemma-dpo-aus",  "stage": "dpo", "family": "gemma", "variant": "aus"},
    {"model_id": "jordanpainter/diallm-gemma-dpo-brit", "stage": "dpo", "family": "gemma", "variant": "brit"},
    {"model_id": "jordanpainter/diallm-gemma-dpo-ind",  "stage": "dpo", "family": "gemma", "variant": "ind"},
    {"model_id": "jordanpainter/diallm-llama-dpo-all",  "stage": "dpo", "family": "llama", "variant": "all",  "template_source": "meta-llama/Llama-3.1-8B-Instruct"},
    {"model_id": "jordanpainter/diallm-llama-dpo-aus",  "stage": "dpo", "family": "llama", "variant": "aus",  "template_source": "meta-llama/Llama-3.1-8B-Instruct"},
    {"model_id": "jordanpainter/diallm-llama-dpo-brit", "stage": "dpo", "family": "llama", "variant": "brit", "template_source": "meta-llama/Llama-3.1-8B-Instruct"},
    {"model_id": "jordanpainter/diallm-llama-dpo-ind",  "stage": "dpo", "family": "llama", "variant": "ind",  "template_source": "meta-llama/Llama-3.1-8B-Instruct"},
    {"model_id": "jordanpainter/diallm-qwen-dpo-all",   "stage": "dpo", "family": "qwen",  "variant": "all"},
    {"model_id": "jordanpainter/diallm-qwen-dpo-aus",   "stage": "dpo", "family": "qwen",  "variant": "aus"},
    {"model_id": "jordanpainter/diallm-qwen-dpo-brit",  "stage": "dpo", "family": "qwen",  "variant": "brit"},
    {"model_id": "jordanpainter/diallm-qwen-dpo-ind",   "stage": "dpo", "family": "qwen",  "variant": "ind"},

    # --- GRPO ---
    {"model_id": "jordanpainter/diallm-gemma-grpo-all",  "stage": "grpo", "family": "gemma", "variant": "all"},
    {"model_id": "jordanpainter/diallm-gemma-grpo-aus",  "stage": "grpo", "family": "gemma", "variant": "aus"},
    {"model_id": "jordanpainter/diallm-gemma-grpo-brit", "stage": "grpo", "family": "gemma", "variant": "brit"},
    {"model_id": "jordanpainter/diallm-gemma-grpo-ind",  "stage": "grpo", "family": "gemma", "variant": "ind"},
    {"model_id": "jordanpainter/diallm-llama-grpo-all",  "stage": "grpo", "family": "llama", "variant": "all",  "template_source": "meta-llama/Llama-3.1-8B-Instruct"},
    {"model_id": "jordanpainter/diallm-llama-grpo-aus",  "stage": "grpo", "family": "llama", "variant": "aus",  "template_source": "meta-llama/Llama-3.1-8B-Instruct"},
    {"model_id": "jordanpainter/diallm-llama-grpo-brit", "stage": "grpo", "family": "llama", "variant": "brit", "template_source": "meta-llama/Llama-3.1-8B-Instruct"},
    {"model_id": "jordanpainter/diallm-llama-grpo-ind",  "stage": "grpo", "family": "llama", "variant": "ind",  "template_source": "meta-llama/Llama-3.1-8B-Instruct"},
    {"model_id": "jordanpainter/diallm-qwen-grpo-all",   "stage": "grpo", "family": "qwen",  "variant": "all"},
    {"model_id": "jordanpainter/diallm-qwen-grpo-aus",   "stage": "grpo", "family": "qwen",  "variant": "aus"},
    {"model_id": "jordanpainter/diallm-qwen-grpo-brit",  "stage": "grpo", "family": "qwen",  "variant": "brit"},
    {"model_id": "jordanpainter/diallm-qwen-grpo-ind",   "stage": "grpo", "family": "qwen",  "variant": "ind"},

    # --- GSPO ---
    {"model_id": "jordanpainter/diallm-gemma-gspo-all",  "stage": "gspo", "family": "gemma", "variant": "all"},
    {"model_id": "jordanpainter/diallm-gemma-gspo-aus",  "stage": "gspo", "family": "gemma", "variant": "aus"},
    {"model_id": "jordanpainter/diallm-gemma-gspo-brit", "stage": "gspo", "family": "gemma", "variant": "brit"},
    {"model_id": "jordanpainter/diallm-gemma-gspo-ind",  "stage": "gspo", "family": "gemma", "variant": "ind"},
    {"model_id": "jordanpainter/diallm-llama-gspo-all",  "stage": "gspo", "family": "llama", "variant": "all",  "template_source": "meta-llama/Llama-3.1-8B-Instruct"},
    {"model_id": "jordanpainter/diallm-llama-gspo-aus",  "stage": "gspo", "family": "llama", "variant": "aus",  "template_source": "meta-llama/Llama-3.1-8B-Instruct"},
    {"model_id": "jordanpainter/diallm-llama-gspo-brit", "stage": "gspo", "family": "llama", "variant": "brit", "template_source": "meta-llama/Llama-3.1-8B-Instruct"},
    {"model_id": "jordanpainter/diallm-llama-gspo-ind",  "stage": "gspo", "family": "llama", "variant": "ind",  "template_source": "meta-llama/Llama-3.1-8B-Instruct"},
    {"model_id": "jordanpainter/diallm-qwen-gspo-all",   "stage": "gspo", "family": "qwen",  "variant": "all"},
    {"model_id": "jordanpainter/diallm-qwen-gspo-aus",   "stage": "gspo", "family": "qwen",  "variant": "aus"},
    {"model_id": "jordanpainter/diallm-qwen-gspo-brit",  "stage": "gspo", "family": "qwen",  "variant": "brit"},
    {"model_id": "jordanpainter/diallm-qwen-gspo-ind",   "stage": "gspo", "family": "qwen",  "variant": "ind"},
]

SYSTEM_PROMPT = "You are a helpful assistant."
MAX_NEW_TOKENS = 512


def setup_logging(log_path: str):
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    for h in handlers:
        h.flush = lambda: None  # force line-buffered below
    logging.root.handlers = handlers


def log(msg: str):
    logging.info(msg)
    for h in logging.root.handlers:
        h.stream.flush() if hasattr(h, "stream") else None


def load_tokenizer(model_info: dict) -> AutoTokenizer:
    # For base/CPT models with a template_source, use the instruct tokenizer
    # directly — vocab is identical since they start from the same base.
    source = model_info.get("template_source") or model_info["model_id"]
    return AutoTokenizer.from_pretrained(source, trust_remote_code=True)


def build_prompt(tokenizer: AutoTokenizer, user_text: str, family: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    # Disable Qwen3 thinking mode for consistent plain responses
    if "qwen3" in (tokenizer.name_or_path or "").lower() or family == "qwen":
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


def run_model(model_info: dict, prompts: list, out_file, already_done: set):
    model_id = model_info["model_id"]
    stage    = model_info["stage"]
    family   = model_info["family"]
    variant  = model_info["variant"]

    log(f"[{stage.upper()}] Loading {model_id}")
    t0 = time.time()

    tokenizer = load_tokenizer(model_info)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    log(f"  Loaded in {time.time() - t0:.1f}s")

    for idx, item in enumerate(prompts):
        key = (model_id, idx)
        if key in already_done:
            log(f"  [{idx+1:02d}/25] SKIP (already done)")
            continue

        prompt_text = build_prompt(tokenizer, item["prompt"], family)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        record = {
            "model_id": model_id,
            "stage":    stage,
            "family":   family,
            "variant":  variant,
            "prompt_id": idx,
            "domain":   item["domain"],
            "prompt":   item["prompt"],
            "response": response,
        }
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_file.flush()

        log(f"  [{idx+1:02d}/25] {item['domain']} — {item['prompt'][:60]}...")

    del model
    torch.cuda.empty_cache()
    log(f"  Done {model_id} ({time.time() - t0:.1f}s total)\n")


def load_done(output_path: str) -> set:
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["model_id"], r["prompt_id"]))
            except Exception:
                pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="annotation_prompt_candidates.json")
    parser.add_argument("--output",  default="annotation_responses.jsonl")
    parser.add_argument("--log",     default="annotation_inference.log")
    parser.add_argument("--stages",  nargs="*", default=None,
                        help="Restrict to specific stages e.g. --stages cpt base")
    args = parser.parse_args()

    setup_logging(args.log)
    log(f"Output : {args.output}")
    log(f"Log    : {args.log}")
    log(f"Models : {len(MODELS)} total")

    with open(args.prompts, encoding="utf-8") as f:
        prompts = json.load(f)
    log(f"Prompts: {len(prompts)}\n")

    models = MODELS
    if args.stages:
        models = [m for m in MODELS if m["stage"] in args.stages]
        log(f"Filtering to stages {args.stages} — {len(models)} models\n")

    already_done = load_done(args.output)
    if already_done:
        log(f"Resuming — {len(already_done)} records already written\n")

    with open(args.output, "a", encoding="utf-8") as out_file:
        for i, model_info in enumerate(models):
            log(f"=== Model {i+1}/{len(models)}: {model_info['model_id']} ===")
            try:
                run_model(model_info, prompts, out_file, already_done)
            except Exception as e:
                log(f"  ERROR on {model_info['model_id']}: {e}")
                torch.cuda.empty_cache()

    log("All done.")


if __name__ == "__main__":
    main()
