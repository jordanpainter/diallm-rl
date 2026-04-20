"""
scripts/analyse_llm_judge.py

Analyse Phi-4 LLM-as-judge results from llm_judge_results.jsonl.

Outputs:
  - Win rates per task × dialect × condition (expert vs feature-informed)
  - Expert vs feature-informed agreement rates
  - Task 3 (triple) method distribution
  - Summary of key findings

Usage:
    python3 scripts/analyse_llm_judge.py [--input llm_judge_results.jsonl]
"""

import json
import argparse
from collections import defaultdict, Counter
from pathlib import Path


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# ---------------------------------------------------------------------------
# Task descriptions
# ---------------------------------------------------------------------------
TASK_META = {
    1: {"label": "Instruct vs SFT_d",       "question": "Does SFT shift toward target variety?",           "win_stage": "sft"},
    2: {"label": "SFT_d vs GRPO_d",          "question": "Does GRPO improve dialectal output over SFT?",    "win_stage": "grpo"},
    3: {"label": "DPO vs GRPO vs GSPO",      "question": "Which alignment method is most dialectal?",       "win_stage": None},
    4: {"label": "GRPO_all vs GRPO_d",       "question": "Does explicit targeting beat broad alignment?",   "win_stage": "grpo"},
}

DIALECTS    = ["en-AU", "en-IN", "en-UK"]
CONDITIONS  = ["expert", "feature"]
TASKS       = [1, 2, 3, 4]


def win_rate(records, task_id, condition, dialect, win_stage):
    """
    For pair tasks: proportion of trials where win_stage wins (excl. ties).
    Returns (wins, ties, losses, tie_excl_rate).
    """
    subset = [r for r in records
              if r["task_id"] == task_id
              and r["condition"] == condition
              and r["dialect"] == dialect]
    if not subset:
        return None

    wins = sum(1 for r in subset if r["winner_stage"] == win_stage)
    ties = sum(1 for r in subset if r["winner_stage"] == "tie")
    total = len(subset)
    losses = total - wins - ties
    tie_excl = wins / (wins + losses) if (wins + losses) > 0 else float("nan")
    return wins, ties, losses, tie_excl, total


def triple_dist(records, task_id, condition, dialect):
    """For task 3: count wins per method."""
    subset = [r for r in records
              if r["task_id"] == task_id
              and r["condition"] == condition
              and r["dialect"] == dialect]
    if not subset:
        return None
    return Counter(r["winner_stage"] for r in subset), len(subset)


def agreement_rate(records, task_id, dialect):
    """
    Proportion of prompt_ids where expert and feature conditions agree on winner.
    For pairs: same winner_stage. For triples: same winner_stage.
    """
    expert_map  = {r["prompt_id"]: r["winner_stage"]
                   for r in records if r["task_id"] == task_id
                   and r["dialect"] == dialect and r["condition"] == "expert"}
    feature_map = {r["prompt_id"]: r["winner_stage"]
                   for r in records if r["task_id"] == task_id
                   and r["dialect"] == dialect and r["condition"] == "feature"}
    shared = set(expert_map) & set(feature_map)
    if not shared:
        return float("nan"), 0
    agree = sum(1 for pid in shared if expert_map[pid] == feature_map[pid])
    return agree / len(shared), len(shared)


def fmt_pct(x):
    return f"{x*100:.1f}%" if x == x else "n/a"


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="llm_judge_results.jsonl")
    args = ap.parse_args()

    records = load(args.input)
    print(f"Loaded {len(records)} records")

    # -----------------------------------------------------------------------
    # Tasks 1, 2, 4 — pair win rates
    # -----------------------------------------------------------------------
    for task_id in [1, 2, 4]:
        meta = TASK_META[task_id]
        print_section(f"Task {task_id}: {meta['label']}")
        print(f"  Q: {meta['question']}")
        print(f"  Win = '{meta['win_stage']}' stage\n")

        header = f"  {'Dialect':<8}  {'Condition':<10}  {'Win':>5}  {'Tie':>5}  {'Loss':>5}  {'Win% (excl tie)':>16}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for dialect in DIALECTS:
            for condition in CONDITIONS:
                result = win_rate(records, task_id, condition, dialect, meta["win_stage"])
                if result is None:
                    print(f"  {dialect:<8}  {condition:<10}  {'no data':>33}")
                    continue
                wins, ties, losses, tie_excl, total = result
                print(f"  {dialect:<8}  {condition:<10}  {wins:>5}  {ties:>5}  {losses:>5}  {fmt_pct(tie_excl):>16}")
            print()

    # -----------------------------------------------------------------------
    # Task 3 — triple distribution
    # -----------------------------------------------------------------------
    print_section("Task 3: DPO vs GRPO vs GSPO (most dialectal, forced choice)")
    print(f"  Q: {TASK_META[3]['question']}\n")

    for dialect in DIALECTS:
        print(f"  {dialect}")
        for condition in CONDITIONS:
            result = triple_dist(records, 3, condition, dialect)
            if result is None:
                print(f"    {condition:<10}  no data")
                continue
            counts, total = result
            dpo  = counts.get("dpo",  0)
            grpo = counts.get("grpo", 0)
            gspo = counts.get("gspo", 0)
            print(f"    {condition:<10}  DPO={dpo}/{total} ({fmt_pct(dpo/total)})  "
                  f"GRPO={grpo}/{total} ({fmt_pct(grpo/total)})  "
                  f"GSPO={gspo}/{total} ({fmt_pct(gspo/total)})")
        print()

    # -----------------------------------------------------------------------
    # Expert vs feature-informed agreement
    # -----------------------------------------------------------------------
    print_section("Expert vs Feature-Informed Agreement")
    print(f"  {'Task':<8}  {'Dialect':<8}  {'Agreement':>10}  {'N':>5}")
    print("  " + "-" * 40)
    for task_id in TASKS:
        for dialect in DIALECTS:
            rate, n = agreement_rate(records, task_id, dialect)
            print(f"  Task {task_id:<4}  {dialect:<8}  {fmt_pct(rate):>10}  {n:>5}")
        print()

    # -----------------------------------------------------------------------
    # Summary: feature-informed win rates for the "positive" stage
    # -----------------------------------------------------------------------
    print_section("Summary — Feature-Informed Condition (tie-excluded win rates)")
    print(f"  {'Task':<30}  {'en-AU':>8}  {'en-IN':>8}  {'en-UK':>8}")
    print("  " + "-" * 60)

    for task_id in [1, 2, 4]:
        meta = TASK_META[task_id]
        row = f"  {meta['label']:<30}"
        for dialect in DIALECTS:
            result = win_rate(records, task_id, "feature", dialect, meta["win_stage"])
            row += f"  {fmt_pct(result[3]) if result else 'n/a':>8}"
        print(row)

    # Task 3 — show GRPO win rate as summary proxy
    row = f"  {'DPO/GRPO/GSPO (GRPO wins)':<30}"
    for dialect in DIALECTS:
        result = triple_dist(records, 3, "feature", dialect)
        if result:
            counts, total = result
            grpo = counts.get("grpo", 0)
            row += f"  {fmt_pct(grpo/total):>8}"
        else:
            row += f"  {'n/a':>8}"
    print(row)

    print()


if __name__ == "__main__":
    main()
