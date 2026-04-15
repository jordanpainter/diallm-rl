from __future__ import annotations

import os
from typing import List, Optional

import torch

from .dialect_reward_model import DialectDensityScorer

_SCORER: Optional[DialectDensityScorer] = None


def reset_scorer(feature_indices: Optional[List[int]] = None) -> None:
    """
    (Re-)initialize the global scorer, optionally with a dialect-specific feature mask.
    Call this before training starts when running a dialect-specific GSPO variant.
    """
    global _SCORER
    model_path = os.environ.get("DIALECT_REWARD_MODEL", "srirag/feature-identifier")
    device = os.environ.get("DIALECT_REWARD_DEVICE") or _default_device()
    max_length = int(os.environ.get("DIALECT_REWARD_MAX_LENGTH", "256"))
    _SCORER = DialectDensityScorer(
        model_path=model_path,
        device=device,
        max_length=max_length,
        feature_indices=feature_indices,
    )


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def _default_device() -> str:
    if torch.cuda.is_available():
        lr = _get_local_rank()
        torch.cuda.set_device(lr)
        return f"cuda:{lr}"
    return "cpu"


def _get_scorer() -> DialectDensityScorer:
    global _SCORER
    if _SCORER is None:
        model_path = os.environ.get("DIALECT_REWARD_MODEL", "srirag/feature-identifier")
        device = os.environ.get("DIALECT_REWARD_DEVICE")
        if device is None:
            device = _default_device()

        max_length = int(os.environ.get("DIALECT_REWARD_MAX_LENGTH", "256"))
        _SCORER = DialectDensityScorer(
            model_path=model_path,
            device=device,
            max_length=max_length,
        )
    return _SCORER


def dialect_log1p(texts: List[str]) -> List[float]:
    """
    Returns log1p(sum(sigmoid(logits))) over active dialect features for each text.
    Denser signal than density, concave compression via log1p.
    """
    scorer = _get_scorer()
    scores = scorer.score_log1p(texts)

    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().tolist()

    return [float(x) for x in scores]