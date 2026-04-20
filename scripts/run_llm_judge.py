"""
scripts/run_llm_judge.py

LLM-as-judge evaluation using Phi-4 for DiaLLM annotation tasks.

Two conditions per trial:
  expert  — Phi-4 acts as a dialect expert with no explicit feature guidance
  feature — Phi-4 is given the eWAVE feature list for the target dialect

Matches the 4-task structure of the human annotation app:
  Task 1: instruct vs sft-dialect          (pair)
  Task 2: sft-dialect vs grpo-dialect      (pair)
  Task 3: dpo vs grpo vs gspo (dialect)    (triple)
  Task 4: grpo-all vs grpo-dialect         (pair)

Outputs a JSONL with one record per (task, prompt, condition).
Resumes from existing output (skips completed task/prompt/condition combos).

Usage:
    python3 scripts/run_llm_judge.py \
        --responses  /path/to/annotation_responses.jsonl \
        --output     /path/to/llm_judge_results.jsonl \
        --log        /path/to/llm_judge.log
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "microsoft/phi-4"
FAMILY   = "llama"
SEED     = 42
MAX_NEW_TOKENS = 32  # Only need a single letter + optional brief explanation

DIALECTS = {
    "en-AU": "aus",
    "en-IN": "ind",
    "en-UK": "brit",
}

TASKS = [
    {
        "id":    1,
        "type":  "pair",
        "model_a": {"stage": "instruct", "variant": "base"},
        "model_b": {"stage": "sft",      "variant": "{dialect}"},
        "question": "Which response sounds more like {dialect_name} English?",
    },
    {
        "id":    2,
        "type":  "pair",
        "model_a": {"stage": "sft",  "variant": "{dialect}"},
        "model_b": {"stage": "grpo", "variant": "{dialect}"},
        "question": "Which response sounds more like {dialect_name} English?",
    },
    {
        "id":    3,
        "type":  "triple",
        "models": [
            {"stage": "dpo",  "variant": "{dialect}"},
            {"stage": "grpo", "variant": "{dialect}"},
            {"stage": "gspo", "variant": "{dialect}"},
        ],
        "question": "Which response sounds most like {dialect_name} English?",
    },
    {
        "id":    4,
        "type":  "pair",
        "model_a": {"stage": "grpo", "variant": "all"},
        "model_b": {"stage": "grpo", "variant": "{dialect}"},
        "question": "Which response sounds more like {dialect_name} English?",
    },
]

# eWAVE feature lists per dialect (from Dialect_reward_features_all.xlsx)
FEATURES = {
    "en-UK": [
        ("Object pronoun as possessive (1SG)",              "He's me brother; I've lost me bike"),
        ("Us with singular referent",                        "Show us them boots"),
        ("2PL pronoun other than you",                       "y'all, youse"),
        ("Definite article where StE has indefinite",        "I had the toothache"),
        ("Zero article where StE has definite article",      "Did you get mileage-claim for that trip?"),
        ("Zero article where StE has indefinite article",    "getting girl from India"),
        ("Definite article where StE favours zero",          "the Nestlé Ghana Ltd."),
        ("Double comparatives/superlatives",                 "so much more easier"),
        ("Progressive extended to stative verbs",            "I'm liking this; What are you wanting?"),
        ("Past tense replacing past participle",             "He had went"),
        ("Past participle replacing past tense",             "He gone to Mary"),
        ("Multiple negation / negative concord",             "He won't do no harm"),
        ("Was/were generalisation",                          "You were hungry but he were thirsty"),
        ("Relativizer as",                                   "a chap as got a living"),
        ("Relativizer at",                                   "The man at painted my house"),
        ("Relativizer what",                                 "The man what painted my house"),
        ("Resumptive/shadow pronouns",                       "This is the house which I painted it yesterday"),
        ("Right as intensifier",                             "right poorly; right good"),
        ("Nowt/owt (nothing/anything)",                      "I've got nowt"),
        ("Bare adverbs (adjective = adverb)",                "Come quick, drive slow"),
        ("Sentence-final discourse marker like",             "It were good, like"),
        ("Like as quotative/focuser",                        "She was like, I don't know"),
    ],
    "en-IN": [
        ("Definite article where StE has indefinite",        "I had the toothache"),
        ("Progressive extended to stative verbs",            "I'm liking this; What are you wanting?"),
        ("Mass noun pluralisation",                          "moneys, knowledges, informations"),
        ("Absence of plural marking after quantifiers",      "four pound; five year"),
        ("Present perfect for StE simple past",              "Some of us have been to New York years ago"),
        ("Questions without subject-auxiliary inversion",    "What you are doing? You are coming tomorrow?"),
        ("Non-standard intensifier too (= very)",            "It is too difficult (= very difficult)"),
        ("Object pronoun drop",                              "I like [it]"),
        ("Subject pronoun drop",                             "[I] sold [it] already"),
        ("Insertion of it where StE favours zero",           "As I made it clear before"),
        ("Invariant non-concord tag innit",                  "They had them in their hair, innit?"),
        ("Would for future (contrast to will)",              "I would eat rice tomorrow"),
    ],
    "en-AU": [
        ("Like as quotative/focuser",                        "She was like, I don't know"),
        ("There's/there is with plural subjects",            "There's two men waiting in the hall"),
        ("Bare adverbs (adjective = adverb)",                "Come quick, drive slow"),
        ("Non-standard adverb formation",                    "She runs real fast"),
        ("Progressive extended to stative verbs",            "I'm liking this"),
        ("Past participle replacing past tense",             "He gone to Mary"),
        ("Multiple negation / negative concord",             "He won't do no harm"),
        ("Me-possessive (1SG)",                              "I've lost me bike"),
        ("Us with singular referent",                        "Show us them boots"),
        ("2PL pronoun other than you",                       "y'all, youse"),
        ("Right as intensifier",                             "right poorly; right good"),
    ],
}

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
EXPERT_SYSTEM = (
    "You are a linguistics expert specialising in {dialect_name} English. "
    "You have in-depth knowledge of the phonological, morphological, syntactic, "
    "and lexical features that distinguish this variety from Standard English."
)

FEATURE_SYSTEM = (
    "You are a linguistics expert specialising in {dialect_name} English. "
    "The following are documented features of this variety according to the "
    "Electronic World Atlas of Varieties of English (eWAVE):\n\n"
    "{feature_list}\n\n"
    "Use these features to guide your judgements."
)

PAIR_USER = (
    "A speaker was given the following prompt:\n\"{prompt}\"\n\n"
    "Here are two responses:\n\n"
    "Response A:\n{resp_a}\n\n"
    "Response B:\n{resp_b}\n\n"
    "{question} "
    "Reply with exactly one word: A, B, or Tie."
)

TRIPLE_USER = (
    "A speaker was given the following prompt:\n\"{prompt}\"\n\n"
    "Here are three responses:\n\n"
    "Response A:\n{resp_a}\n\n"
    "Response B:\n{resp_b}\n\n"
    "Response C:\n{resp_c}\n\n"
    "{question} "
    "Reply with exactly one letter: A, B, or C."
)


def build_feature_list(dialect_name):
    feats = FEATURES.get(dialect_name, [])
    lines = [f"- {name}: e.g. \"{example}\"" for name, example in feats]
    return "\n".join(lines)


def build_messages(condition, dialect_name, trial, task):
    if condition == "expert":
        system = EXPERT_SYSTEM.format(dialect_name=dialect_name)
    else:
        system = FEATURE_SYSTEM.format(
            dialect_name=dialect_name,
            feature_list=build_feature_list(dialect_name),
        )

    question = task["question"].format(dialect_name=dialect_name)

    if trial["task_type"] == "pair":
        user = PAIR_USER.format(
            prompt=trial["prompt"],
            resp_a=trial["left_resp"],
            resp_b=trial["right_resp"],
            question=question,
        )
    else:
        opts = trial["options"]
        user = TRIPLE_USER.format(
            prompt=trial["prompt"],
            resp_a=opts[0]["response"],
            resp_b=opts[1]["response"],
            resp_c=opts[2]["response"],
            question=question,
        )

    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def parse_judgement(text, task_type):
    """Extract A/B/C/Tie from model output."""
    text = text.strip()
    # Look for standalone A/B/C/Tie (case-insensitive)
    for pattern in [r'\bTie\b', r'\bA\b', r'\bB\b', r'\bC\b']:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = m.group().capitalize()
            if task_type == "triple" and val == "Tie":
                continue  # no tie for triples
            return val
    return "PARSE_ERROR"


# ---------------------------------------------------------------------------
# Trial building (mirrors annotation app logic)
# ---------------------------------------------------------------------------
def resolve_variant(template, dialect_code):
    return dialect_code if template == "{dialect}" else template


def build_trials(records, dialect_code):
    idx = {}
    for r in records:
        if r["family"] == FAMILY:
            idx[(r["stage"], r["variant"], r["prompt_id"])] = r

    prompt_meta = {}
    for r in records:
        if r["family"] == FAMILY and r["stage"] in ("instruct", "sft") \
                and r["variant"] in ("base", dialect_code):
            if r["prompt_id"] not in prompt_meta:
                prompt_meta[r["prompt_id"]] = {
                    "prompt": r["prompt"],
                    "domain": r.get("domain", ""),
                }
    prompt_ids = sorted(prompt_meta.keys())

    all_trials = []
    for task in TASKS:
        task_rng = random.Random(SEED + task["id"])
        for pid in prompt_ids:
            meta = prompt_meta.get(pid, {"prompt": "", "domain": ""})
            trial = {
                "task_id":   task["id"],
                "task_type": task["type"],
                "prompt_id": pid,
                "domain":    meta["domain"],
                "prompt":    meta["prompt"],
            }
            if task["type"] == "pair":
                stage_a   = task["model_a"]["stage"]
                variant_a = resolve_variant(task["model_a"]["variant"], dialect_code)
                stage_b   = task["model_b"]["stage"]
                variant_b = resolve_variant(task["model_b"]["variant"], dialect_code)
                rec_a = idx.get((stage_a, variant_a, pid))
                rec_b = idx.get((stage_b, variant_b, pid))
                if rec_a is None or rec_b is None:
                    continue
                a_is_left = task_rng.random() < 0.5
                trial.update({
                    "stage_a":    stage_a,   "variant_a":  variant_a,
                    "stage_b":    stage_b,   "variant_b":  variant_b,
                    "a_is_left":  a_is_left,
                    "left_resp":  rec_a["response"] if a_is_left else rec_b["response"],
                    "right_resp": rec_b["response"] if a_is_left else rec_a["response"],
                })
            else:
                resolved = []
                for m in task["models"]:
                    s = m["stage"]
                    v = resolve_variant(m["variant"], dialect_code)
                    rec = idx.get((s, v, pid))
                    if rec:
                        resolved.append({"stage": s, "variant": v, "response": rec["response"]})
                if len(resolved) < 3:
                    continue
                task_rng.shuffle(resolved)
                trial["options"] = resolved
            all_trials.append(trial)
    return all_trials


def winner_from_judgement(trial, judgement):
    if trial["task_type"] == "pair":
        if judgement == "Tie":
            return "tie", "tie"
        # A/B refers to left/right; map back to stage_a/stage_b
        left_wins = (judgement == "A")
        if trial["a_is_left"] == left_wins:
            return trial["stage_a"], trial["variant_a"]
        else:
            return trial["stage_b"], trial["variant_b"]
    else:
        if judgement == "PARSE_ERROR":
            return "error", "error"
        ci = {"A": 0, "B": 1, "C": 2}[judgement]
        opt = trial["options"][ci]
        return opt["stage"], opt["variant"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--log",       default="llm_judge.log")
    parser.add_argument("--dialect",   default=None,
                        help="Restrict to one dialect name e.g. en-UK")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    # Load existing results for resumption
    done = set()
    out_path = Path(args.output)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["dialect"], r["task_id"], r["prompt_id"], r["condition"]))
    log.info(f"Resuming — {len(done)} records already done")

    # Load responses
    records = []
    with open(args.responses, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # Determine dialects to run
    dialects_to_run = {args.dialect: DIALECTS[args.dialect]} if args.dialect else DIALECTS

    # Load model
    log.info(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    log.info("Model loaded")

    conditions = ["expert", "feature"]

    with open(out_path, "a", encoding="utf-8") as f_out:
        for dialect_name, dialect_code in dialects_to_run.items():
            log.info(f"Building trials for {dialect_name}")
            trials = build_trials(records, dialect_code)
            task_map = {t["id"]: t for t in TASKS}

            for trial in trials:
                task = task_map[trial["task_id"]]
                for condition in conditions:
                    key = (dialect_name, trial["task_id"], trial["prompt_id"], condition)
                    if key in done:
                        continue

                    messages = build_messages(condition, dialect_name, trial, task)
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = tokenizer(text, return_tensors="pt").to(model.device)

                    with torch.no_grad():
                        output = model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    raw = tokenizer.decode(
                        output[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    ).strip()

                    judgement = parse_judgement(raw, trial["task_type"])
                    winner_stage, winner_variant = winner_from_judgement(trial, judgement)

                    record = {
                        "dialect":        dialect_name,
                        "task_id":        trial["task_id"],
                        "trial_type":     trial["task_type"],
                        "prompt_id":      trial["prompt_id"],
                        "domain":         trial["domain"],
                        "condition":      condition,
                        "judgement":      judgement,
                        "winner_stage":   winner_stage,
                        "winner_variant": winner_variant,
                        "raw_response":   raw,
                    }
                    if trial["task_type"] == "pair":
                        record["comparison"] = f"{trial['stage_a']}_{trial['variant_a']}_vs_{trial['stage_b']}_{trial['variant_b']}"
                        record["a_is_left"]  = trial["a_is_left"]
                    else:
                        record["comparison"] = "_vs_".join(
                            f"{o['stage']}_{o['variant']}" for o in trial["options"]
                        )

                    f_out.write(json.dumps(record) + "\n")
                    f_out.flush()
                    done.add(key)

                    log.info(
                        f"{dialect_name} | task={trial['task_id']} pid={trial['prompt_id']} "
                        f"cond={condition} → {judgement} ({winner_stage}) | {raw[:60]}"
                    )

    log.info("Done.")


if __name__ == "__main__":
    main()
