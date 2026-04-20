# DialLM-RL Development Log

---

## Dialect Classifier Results on Annotation Responses

**Data:** 54 models × 25 prompts = 1,350 responses  
**Classifier:** `jordanpainter/diallm-dialect-classifier` (DeBERTa-v3-base, 77.2% on BESSTIE-CW-26)  
**Metric:** P(target dialect) and accuracy (classifier_pred == target) per model/variant

### en-UK: Classifier Metric Not Usable

Accuracy is **0–2.7% across all stages** — the classifier almost always predicts en-AU instead of en-UK. This reflects the AU/UK confusion baked into the classifier (shared orthography and spelling conventions). The classifier metric is only interpretable for **en-AU and en-IN**. Must be acknowledged in the paper as a known confound between closely related varieties.

### Base → CPT Distribution Shift

| Stage | en-AU pred | en-IN pred | en-UK pred | Mean P(AU) | Mean P(IN) | Mean P(UK) |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| Base  | 55/75 | 16/75 | 4/75 | 0.507 | 0.258 | 0.235 |
| CPT   | 11/75 | 63/75 | 1/75 | 0.159 | 0.727 | 0.114 |

Neutral base model English is classified as en-AU by default. CPT on ICE dramatically shifts outputs toward en-IN — likely reflecting the composition/register of the ICE corpus. Interesting finding in itself.

### Alignment Progression — en-AU Accuracy

| Family | SFT  | DPO      | GRPO | GSPO     |
|--------|------|----------|------|----------|
| Gemma  | 0.84 | 0.84     | 0.88 | **0.92** |
| Llama  | 0.56 | **0.76** | 0.48 | 0.52     |
| Qwen   | 0.68 | **0.80** | 0.76 | **0.80** |
| **Mean** | **0.69** | **0.80** | **0.71** | **0.75** |

### Alignment Progression — en-IN Accuracy

| Family | SFT  | DPO      | GRPO | GSPO     |
|--------|------|----------|------|----------|
| Gemma  | 0.52 | **0.60** | 0.44 | 0.44     |
| Llama  | 0.72 | 0.76     | 0.80 | **0.84** |
| Qwen   | 0.40 | **0.68** | 0.36 | 0.44     |
| **Mean** | **0.55** | **0.68** | **0.53** | **0.57** |

### Stage-Level Summary (dialect-specific variants, averaged across families)

| Stage | P(AU) | Acc(AU) | P(IN) | Acc(IN) | P(UK) | Acc(UK) |
|-------|-------|---------|-------|---------|-------|---------|
| SFT   | 0.462 | 0.693   | 0.438 | 0.547   | 0.214 | 0.013   |
| DPO   | 0.517 | 0.800   | 0.525 | 0.680   | 0.206 | 0.013   |
| GRPO  | 0.459 | 0.707   | 0.434 | 0.533   | 0.212 | 0.027   |
| GSPO  | 0.472 | 0.747   | 0.459 | 0.573   | 0.218 | 0.027   |

### Key Takeaways

1. **DPO consistently improves over SFT** for both en-AU and en-IN across all three families — the most reliable alignment gain.
2. **GSPO is best for Gemma+AU (0.92) and Llama+IN (0.84)** — the families that adapted most cleanly.
3. **GRPO is inconsistent** — falls below SFT in several cases (Llama/AU: 0.48 vs 0.56, Qwen/IN: 0.36 vs 0.40). Worth reporting as a finding.
4. **en-UK not measurable** via this classifier — exclude from classifier analysis or caveat explicitly.
5. **CPT distribution shift** is worth flagging as a qualitative observation about what CPT does to output register.

### Paper Framing

Present as a **supporting metric** alongside human annotation. The 25-prompt sample means individual numbers are noisy but aggregate trends are interpretable. The classifier's independence from the eWAVE reward makes it a valid cross-check that directly addresses the reviewer circularity concern.

---

## Updated Analysis — Hand-Crafted Annotation Prompts (2026-04-20)

New results from `annotation_responses_features.jsonl` and `annotation_responses_classified.jsonl` using the 25 hand-crafted casual prompts (replacing the earlier ShareGPT set). Llama, Gemma, Qwen × aus/brit/ind × sft/dpo/grpo/gspo = 300 explicit-thread records each.

### Feature Density (eWAVE, explicit thread only)

Mean feature density per alignment method, averaged across all three dialect variants, per family:

| Family | DPO    | GRPO       | GSPO   |
|--------|--------|------------|--------|
| Llama  | 0.1242 | **0.1424** | 0.1359 |
| Gemma  | 0.1281 | **0.1489** | 0.1459 |
| Qwen   | 0.1262 | **0.1422** | 0.1360 |

**GRPO consistently highest across all three families.** GSPO second, DPO lowest. This is robust and consistent — use GRPO as the "best alignment" method for automatic metric comparisons.

**Tension with qualitative observation:** DPO responses read more naturally but score lowest on feature density. GRPO generates more detectable dialectal features but with rougher surface form. This is a genuine finding worth discussing in the paper — the two metrics are not measuring the same thing.

### Dialect Classifier — Updated Accuracy (explicit thread)

#### en-AU accuracy

| Family | SFT  | DPO  | GRPO | GSPO |
|--------|------|------|------|------|
| Llama  | 0.88 | 0.88 | 0.84 | 0.76 |
| Gemma  | 0.92 | 0.84 | 0.80 | 0.80 |
| Qwen   | 0.88 | 0.76 | 0.80 | 0.76 |
| **Mean** | **0.89** | **0.83** | **0.81** | **0.77** |

SFT already achieves very high en-AU accuracy. Alignment does not consistently improve over SFT on the classifier metric for en-AU — but feature density does increase. These measure different things.

#### en-IN accuracy

| Family | SFT  | DPO      | GRPO | GSPO |
|--------|------|----------|------|------|
| Llama  | 0.48 | **0.64** | 0.48 | 0.40 |
| Gemma  | 0.56 | **0.60** | 0.60 | 0.56 |
| Qwen   | 0.52 | **0.76** | 0.36 | 0.36 |
| **Mean** | **0.52** | **0.67** | **0.48** | **0.44** |

DPO is the clear winner on classifier accuracy for en-IN. GRPO and GSPO fall below SFT in several cases (Qwen especially). Again contrasts with feature density where GRPO leads.

#### en-UK classifier — not usable

Of 300 brit-variant records (sft/dpo/grpo/gspo, all families), the classifier predicts:
- **en-AU: 81.7%**
- en-IN: 17.0%
- en-UK: 1.3%

The broad (-all) models show the same distribution (83% en-AU). The classifier cannot distinguish en-UK outputs from en-AU. The eWAVE features rewarded for en-UK (y'all, me-possessive, us-singular) do not match BESSTIE's en-UK feature distribution. **Do not use classifier accuracy as an en-UK metric in the paper.** Use feature density only for en-UK, and acknowledge this limitation explicitly.

### Implications for Paper

- Report classifier accuracy for en-AU and en-IN only; note en-UK limitation
- Report feature density for all three varieties as the primary generation quality metric
- The DPO vs GRPO divergence (DPO wins on classifier, GRPO wins on feature density) is a genuine finding — frame it as showing that the two metrics capture different aspects of dialectal output quality

---

## Session 1 — Infrastructure & Dataset Fixes

### SLURM Sub Files (`scripts/gen_sub_files.py`)
- Fixed `--gres=gpus:1` typo → `--gres=gpu:1` (was causing `sbatch: Invalid generic resource` error)
- Fixed `PYTHONPATH: unbound variable` crash inside apptainer exec: added `export PYTHONPATH="${PYTHONPATH:-}"` to outer script and escaped inner echo to `\\$PYTHONPATH`
- Restructured `GPU_CONFIG` to `(partition, gres, mem)` tuples with per-experiment GPU assignments:
  - Broad (gemma/llama/qwen): `3090` partition, 4× RTX 3090
  - Narrow Gemma: `a100`, 1 GPU (A100-80GB)
  - Narrow Llama: `3090_risk`, 4 GPUs
  - Narrow Qwen: `3090`, 4 GPUs
- Added `DPO_GPU_CONFIG`: all 12 DPO experiments → `a100 / gpu:1 / 64G`

### Dataset Loading (`src/train.py`, `src/dpo.py`)
- Fixed dataset loading for HuggingFace `save_to_disk()` Arrow snapshots
- Old method (`hf_load_dataset`) returned 1 row of internal metadata instead of training data
- Fix: `snapshot_download(repo_id=..., repo_type="dataset")` + `load_from_disk()`, with fallback to `hf_load_dataset` for non-snapshot datasets

### Config Generator (`scripts/gen_configs.py`)
- Added `dpo_dataset_id` field to broad experiments (gemma/llama/qwen): `jordanpainter/dialect-preferences`
- Narrow experiments already used correct dialect-specific datasets (australian/indian/british-final)
- `build_dpo_config` updated to use `exp.get("dpo_dataset_id", exp["dataset_id"])`

### Status
- 12 DPO jobs submitted and queued on `a100` partition, pending resources
- GRPO/GSPO runs paused — reward function under discussion (resolved in Session 2 below)

---

## Session 2 — Dialect Reward Function

### Problem
The original reward used **sigmoid-density**: `sum(sigmoid(logits)) / num_features`, giving a compressed `[0, 1]` signal with very little within-batch spread (~0.026 range on test sentences). This limits the gradient signal GRPO can use to differentiate between completions.

### Reference Implementation
Unpacked `for-jordan.tar.gz` from Srirag. The intended reward formula from `reward_model.py` is:

```
reward = log1p(sum(sigmoid(logits)))
```

Not density (no division), and with `log1p` compression for concavity.

### Changes Made

**`rewards/dialect_reward_model.py`**
- Removed: `score_raw`, `score_density`, `score_details`, `compare_density`
- Added: `score_log1p` — computes `log1p(sum(sigmoid(logits[feature_indices])))`

**`rewards/dialect_reward.py`**
- Removed: `dialect_density`, `dialect_raw_score`, `dialect_density_gain`
- Added: `dialect_log1p` — public wrapper around `score_log1p`

**`src/train.py`**
- Updated import: `dialect_log1p` (was `dialect_density`)
- Updated call sites in `CombinedReward.__call__`
- Updated log line labels: `gen_log1p` / `chosen_log1p`

### Result
On the same 5 test sentences:

| Method | Range |
|---|---|
| Old density `sum(sigmoid) / 135` | 0.026 |
| New `log1p(sum(sigmoid))` | 0.379 |

**15× wider spread** — much denser signal for GRPO to differentiate completions within a batch.

### Narrow Run Masking
Unchanged — for narrow runs, `reset_scorer(feature_indices=...)` is called at startup and `score_log1p` slices to only those feature columns before summing. Theoretical max for narrow runs: `log1p(len(feature_indices))` e.g. `log1p(20) ≈ 3.04` for AusE vs `log1p(135) ≈ 4.91` for broad.

---

## Pending

- GRPO/GSPO runs: ready to submit once reward function confirmed (done above)
- Broad Gemma GRPO: was OOM on RTX 8000 — consider moving to `a100` in `GPU_CONFIG`
- LR correction in paper Appendix G: Llama/Qwen shown as `1×10⁻⁶`, configs use `2×10⁻⁶`
