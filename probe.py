"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


from sklearn.ensemble import RandomForestClassifier

class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=15,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X_np = x.detach().cpu().numpy()
        probs = self.model.predict_proba(X_np)[:, 1]
        return torch.from_numpy(probs).float()

