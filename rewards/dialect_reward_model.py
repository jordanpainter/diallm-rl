from __future__ import annotations

from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

from .dialect_feature_model import MultiheadDialectFeatureModel


class DialectDensityScorer:
    """
    Loads the trained dialect feature classifier and exposes reusable scoring methods.

    For each text, we compute:
    - logits for each dialect feature
    - probabilities via sigmoid
    - raw_score = sum(feature probabilities) over active indices
    - density = raw_score / num_active_features

    density is in [0, 1] and is the cleanest quantity to compare between:
    - generation
    - base output

    If feature_indices is provided, only those logit positions are summed and the
    denominator becomes len(feature_indices), giving density relative to the
    target dialect's own attested feature set rather than all 135 features.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 256,
        feature_indices: Optional[List[int]] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = MultiheadDialectFeatureModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.num_features = int(self.model.num_features)

        if feature_indices is not None:
            self._feature_indices = torch.tensor(feature_indices, dtype=torch.long, device=self.device)
            self.num_active_features = len(feature_indices)
        else:
            self._feature_indices = None
            self.num_active_features = self.num_features

    def to(self, device: str) -> "DialectDensityScorer":
        self.device = torch.device(device)
        self.model.to(self.device)
        if self._feature_indices is not None:
            self._feature_indices = self._feature_indices.to(self.device)
        return self

    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)

    @torch.no_grad()
    def predict_logits(self, texts: List[str]) -> torch.Tensor:
        inputs = self._tokenize(texts)
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        return outputs.logits.detach()

    @torch.no_grad()
    def predict_feature_probabilities(self, texts: List[str]) -> torch.Tensor:
        logits = self.predict_logits(texts)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def score_log1p(self, texts: List[str]) -> torch.Tensor:
        """
        Returns log1p(sum(sigmoid(logits))) over active dialect features per text.
        If feature_indices was set, only those positions are included.
        Range: [0, log1p(num_active_features)] — denser signal than density,
        concave compression discourages runaway reward hacking.
        Shape: (batch,)
        """
        logits = self.predict_logits(texts)
        probs = torch.sigmoid(logits)
        if self._feature_indices is not None:
            probs = probs[:, self._feature_indices]
        return torch.log1p(probs.sum(dim=1))