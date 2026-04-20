"""
annotation_app/app.py

4-task dialect preference annotation tool for DiaLLM EMNLP submission.

Tasks (25 trials each, presented in blocks):
  1. Llama-Instruct vs SFT-dialect  — which is more dialectal? (pair)
  2. SFT-dialect vs GRPO-dialect    — which is more dialectal? (pair)
  3. DPO vs GRPO vs GSPO (dialect)  — which is most dialectal? (triple)
  4. GRPO-all vs GRPO-dialect       — which is more dialectal? (pair)

Run:
    streamlit run annotation_app/app.py
"""

import json
import random
import csv
from pathlib import Path
from datetime import datetime

import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESPONSES_FILE = Path(__file__).parent / "data" / "responses.jsonl"
FAMILY = "llama"
SEED   = 42

DIALECTS = {
    "en-AU": "aus",
    "en-IN": "ind",
    "en-UK": "brit",
}

TASKS = [
    {
        "id":    1,
        "type":  "pair",
        "title": "Task 1 of 4",
        "description": (
            "You will see **25 pairs** of responses to the same prompt.\n\n"
            "Select whichever response sounds **more dialectal** for **{dialect} English** "
            "— closer to how a speaker of that variety might naturally respond."
        ),
        "model_a": {"stage": "instruct", "variant": "base"},
        "model_b": {"stage": "sft",      "variant": "{dialect}"},
        "question": "Which response sounds more like {dialect} English?",
    },
    {
        "id":    2,
        "type":  "pair",
        "title": "Task 2 of 4",
        "description": (
            "Another **25 pairs** of responses to the same prompts.\n\n"
            "As before, select whichever response sounds **more dialectal** for **{dialect} English**."
        ),
        "model_a": {"stage": "sft",  "variant": "{dialect}"},
        "model_b": {"stage": "grpo", "variant": "{dialect}"},
        "question": "Which response sounds more like {dialect} English?",
    },
    {
        "id":    3,
        "type":  "triple",
        "title": "Task 3 of 4",
        "description": (
            "You will see **25 sets of three responses** to the same prompt.\n\n"
            "Select whichever response sounds **most dialectal** for **{dialect} English**.\n\n"
            "There is no tie option — pick the best of the three."
        ),
        "models": [
            {"stage": "dpo",  "variant": "{dialect}"},
            {"stage": "grpo", "variant": "{dialect}"},
            {"stage": "gspo", "variant": "{dialect}"},
        ],
        "question": "Which response sounds most like {dialect} English?",
    },
    {
        "id":    4,
        "type":  "pair",
        "title": "Task 4 of 4",
        "description": (
            "Final **25 pairs** of responses.\n\n"
            "Select whichever response sounds **more dialectal** for **{dialect} English**."
        ),
        "model_a": {"stage": "grpo", "variant": "all"},
        "model_b": {"stage": "grpo", "variant": "{dialect}"},
        "question": "Which response sounds more like {dialect} English?",
    },
]

BLOCK_SIZE = 25


# ---------------------------------------------------------------------------
# Data loading & trial construction
# ---------------------------------------------------------------------------
@st.cache_data
def load_responses():
    records = []
    with open(RESPONSES_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def resolve_variant(template, dialect_code):
    return dialect_code if template == "{dialect}" else template


def build_all_trials(records, dialect_code):
    """Build 100 trials in task-block order (25 per task)."""
    # Index by (stage, variant, prompt_id)
    idx = {}
    for r in records:
        if r["family"] == FAMILY:
            idx[(r["stage"], r["variant"], r["prompt_id"])] = r

    # Prompt metadata — prefer instruct records, fall back to sft
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
                    "stage_a":    stage_a,
                    "variant_a":  variant_a,
                    "stage_b":    stage_b,
                    "variant_b":  variant_b,
                    "a_is_left":  a_is_left,
                    "left_resp":  rec_a["response"] if a_is_left else rec_b["response"],
                    "right_resp": rec_b["response"] if a_is_left else rec_a["response"],
                })

            else:  # triple
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
                trial["options"] = resolved  # list of 3: {stage, variant, response}

            all_trials.append(trial)

    return all_trials


# ---------------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------------
def save_results(annotator, dialect, trials, judgements):
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name  = annotator.strip().replace(" ", "_")
    filename   = f"annotations_{dialect}_{safe_name}_{timestamp}.csv"

    fieldnames = [
        "annotator", "dialect", "task_id", "trial_type",
        "prompt_id", "domain", "comparison",
        "a_is_left", "judgement", "winner_stage", "winner_variant",
    ]

    rows = []
    for trial, judgement in zip(trials, judgements):
        row = {
            "annotator":  annotator,
            "dialect":    dialect,
            "task_id":    trial["task_id"],
            "trial_type": trial["task_type"],
            "prompt_id":  trial["prompt_id"],
            "domain":     trial["domain"],
            "judgement":  judgement,
        }
        if trial["task_type"] == "pair":
            row["comparison"]     = f"{trial['stage_a']}_{trial['variant_a']}_vs_{trial['stage_b']}_{trial['variant_b']}"
            row["a_is_left"]      = trial["a_is_left"]
            if judgement == "A":
                row["winner_stage"]   = trial["stage_a"]
                row["winner_variant"] = trial["variant_a"]
            elif judgement == "B":
                row["winner_stage"]   = trial["stage_b"]
                row["winner_variant"] = trial["variant_b"]
            else:
                row["winner_stage"]   = "tie"
                row["winner_variant"] = "tie"
        else:
            opts = trial["options"]
            row["comparison"]     = "_vs_".join(f"{o['stage']}_{o['variant']}" for o in opts)
            row["a_is_left"]      = ""
            ci = {"A": 0, "B": 1, "C": 2}[judgement]
            row["winner_stage"]   = opts[ci]["stage"]
            row["winner_variant"] = opts[ci]["variant"]
        rows.append(row)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return filename


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def task_for_index(i):
    return TASKS[i // BLOCK_SIZE]


def main():
    st.set_page_config(page_title="DiaLLM Annotation", layout="wide")
    st.title("DiaLLM — Dialect Generation Annotation")

    # --- Setup ---
    if "started" not in st.session_state:
        st.markdown("## Setup")
        name    = st.text_input("Your name", placeholder="e.g. Jane Smith")
        dialect = st.selectbox("Your assigned dialect", list(DIALECTS.keys()))
        st.markdown(
            "**Overview:** You will complete **100 judgements** across 4 tasks (25 each).  \n"
            "Each task asks you to read responses to a prompt and judge which sounds "
            "more dialectal for your assigned variety of English.  \n"
            "Full instructions appear at the start of each task."
        )
        if st.button("Start", disabled=not name.strip(), type="primary"):
            records = load_responses()
            st.session_state.started          = True
            st.session_state.annotator        = name.strip()
            st.session_state.dialect          = dialect
            st.session_state.dialect_code     = DIALECTS[dialect]
            st.session_state.trials           = build_all_trials(records, DIALECTS[dialect])
            st.session_state.current          = 0
            st.session_state.judgements       = []
            st.session_state.task_intro_seen  = set()
            st.rerun()
        return

    trials       = st.session_state.trials
    current      = st.session_state.current
    total        = len(trials)
    dialect      = st.session_state.dialect
    annotator    = st.session_state.annotator

    # --- Complete ---
    if current >= total:
        st.success(f"All {total} judgements complete — thank you, {annotator}!")
        filename = save_results(
            annotator, dialect, trials, st.session_state.judgements
        )
        with open(filename, "rb") as f:
            st.download_button(
                "⬇️ Download results CSV", f,
                file_name=filename, mime="text/csv",
            )
        return

    task      = task_for_index(current)
    block_pos = current % BLOCK_SIZE

    # --- Task intro ---
    if task["id"] not in st.session_state.task_intro_seen:
        st.markdown(f"## {task['title']}")
        st.markdown(task["description"].format(dialect=dialect))
        st.markdown("---")
        if task["type"] == "pair":
            st.info("Buttons: **A** · **Tie** · **B**")
        else:
            st.info("Buttons: **A** · **B** · **C** — no tie, pick the best of the three")
        if st.button(f"Begin {task['title']}", type="primary"):
            st.session_state.task_intro_seen.add(task["id"])
            st.rerun()
        return

    # --- Progress ---
    st.caption(
        f"{task['title']} · Trial {block_pos + 1} / {BLOCK_SIZE}"
        f"  |  Overall {current + 1} / {total}"
    )
    st.progress((current + 1) / total)

    # --- Trial ---
    trial = trials[current]
    st.markdown(f"**Prompt:** {trial['prompt']}")
    st.markdown("---")

    question = f"**{task['question'].format(dialect=dialect)}**"

    if trial["task_type"] == "pair":
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Response A")
            st.markdown(trial["left_resp"])
        with col_b:
            st.markdown("### Response B")
            st.markdown(trial["right_resp"])

        st.markdown("---")
        st.markdown(question)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("A", use_container_width=True, key=f"btn_A_{current}"):
                j = "A" if trial["a_is_left"] else "B"
                st.session_state.judgements.append(j)
                st.session_state.current += 1
                st.rerun()
        with c2:
            if st.button("Tie", use_container_width=True, key=f"btn_tie_{current}"):
                st.session_state.judgements.append("Tie")
                st.session_state.current += 1
                st.rerun()
        with c3:
            if st.button("B", use_container_width=True, key=f"btn_B_{current}"):
                j = "B" if trial["a_is_left"] else "A"
                st.session_state.judgements.append(j)
                st.session_state.current += 1
                st.rerun()

    else:  # triple
        opts = trial["options"]
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("### Response A")
            st.markdown(opts[0]["response"])
        with col_b:
            st.markdown("### Response B")
            st.markdown(opts[1]["response"])
        with col_c:
            st.markdown("### Response C")
            st.markdown(opts[2]["response"])

        st.markdown("---")
        st.markdown(question)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("A", use_container_width=True, key=f"btn_A_{current}"):
                st.session_state.judgements.append("A")
                st.session_state.current += 1
                st.rerun()
        with c2:
            if st.button("B", use_container_width=True, key=f"btn_B_{current}"):
                st.session_state.judgements.append("B")
                st.session_state.current += 1
                st.rerun()
        with c3:
            if st.button("C", use_container_width=True, key=f"btn_C_{current}"):
                st.session_state.judgements.append("C")
                st.session_state.current += 1
                st.rerun()


if __name__ == "__main__":
    main()
