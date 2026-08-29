"""
Root-cause classification: *why* is this revenue at risk?

One model per surface, all three learned from data. v2 used a trained model
for payment failures and hand-written if/else rules for the other two
surfaces, on the reasoning that abandonment and receivables logic was
already well understood. That was a fair call when there were no labels for
those surfaces — improvements.md #2 lists adding them as the follow-up once
labelled data exists. The data simulator now produces those labels, so all
three surfaces are learned here, which buys two things beyond accuracy:

* **Real confidence numbers.** The hand-written rules returned hardcoded
  confidences (`return "price_sensitivity", 0.70`). That number was a
  guess, and it flowed straight into a guardrail that routes low-confidence
  cases to a human — so the guardrail was effectively gated on a constant.
  A fitted model returns a calibrated posterior, which makes the gate mean
  something.
* **A full distribution, not just an argmax.** The economics layer needs
  P(fraud) specifically, not merely "the top class isn't fraud", to price
  chargeback risk into a retry. See src/economics.py.

The classifier deliberately gets no access to outcome columns; enforced by
`assert_no_leakage` on every feature set.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from . import features as F
from .logistic import (
    LogisticModel, accuracy, expected_calibration_error, log_loss, reliability_table,
)
from ..schemas import CHECKOUT_ABANDONMENT, OVERDUE_RECEIVABLE, PAYMENT_FAILURE


# Which columns feed which surface's classifier.
SURFACE_FEATURES: dict[str, dict[str, list[str]]] = {
    PAYMENT_FAILURE: {
        "numeric": F.PAYMENT_NUMERIC + F.PAYMENT_CUSTOMER_NUMERIC,
        "categorical": F.PAYMENT_CATEGORICAL,
    },
    CHECKOUT_ABANDONMENT: {
        "numeric": F.CHECKOUT_NUMERIC + F.CHECKOUT_CUSTOMER_NUMERIC,
        "categorical": F.CHECKOUT_CATEGORICAL,
    },
    OVERDUE_RECEIVABLE: {
        # `dispute_flagged_in_ar` is a genuine decision-time field from the
        # AR system, not a hidden label. Including it is the whole reason
        # the agent can refuse to chase a disputed invoice.
        "numeric": F.RECEIVABLE_NUMERIC + F.RECEIVABLE_CUSTOMER_NUMERIC + ["dispute_flagged_in_ar_int"],
        "categorical": F.RECEIVABLE_CATEGORICAL,
    },
}

DERIVE: dict[str, Any] = {
    PAYMENT_FAILURE: F.add_payment_features,
    CHECKOUT_ABANDONMENT: F.add_checkout_features,
    OVERDUE_RECEIVABLE: F.add_receivable_features,
}


def prepare(surface: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply the surface's derived features plus the shared customer ones."""
    out = DERIVE[surface](df)
    out = F.add_customer_features(out)
    if surface == OVERDUE_RECEIVABLE:
        out["dispute_flagged_in_ar_int"] = (
            out["dispute_flagged_in_ar"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
        )
    return out


class RootCauseClassifier:
    """Predicts a probability distribution over root causes for one surface."""

    def __init__(self, surface: str):
        self.surface = surface
        spec = SURFACE_FEATURES[surface]
        F.assert_no_leakage(spec["numeric"] + spec["categorical"])
        self.encoder = F.FeatureEncoder(spec["numeric"], spec["categorical"])
        self.model = LogisticModel(l2=2e-3, lr=0.1, max_iter=1500)
        self.classes: list[str] = []
        self.metrics: dict[str, Any] = {}

    # -- training ----------------------------------------------------

    def fit(self, df: pd.DataFrame, label_col: str = "true_root_cause") -> "RootCauseClassifier":
        prepared = prepare(self.surface, df)
        X = self.encoder.fit_transform(prepared)
        self.classes = sorted(prepared[label_col].astype(str).unique())
        index = {c: i for i, c in enumerate(self.classes)}
        y = prepared[label_col].astype(str).map(index).to_numpy()
        self.model.fit(X, y, n_classes=len(self.classes))
        return self

    def evaluate(self, df: pd.DataFrame, label_col: str = "true_root_cause") -> dict[str, Any]:
        prepared = prepare(self.surface, df)
        X = self.encoder.transform(prepared)
        proba = self.model.predict_proba(X)
        index = {c: i for i, c in enumerate(self.classes)}
        y = prepared[label_col].astype(str).map(index).to_numpy()
        pred = proba.argmax(axis=1)

        # Majority-class baseline. Quoting accuracy without it is close to
        # meaningless on an imbalanced label set.
        counts = np.bincount(y, minlength=len(self.classes))
        majority = float(counts.max() / counts.sum())

        # Is the model's own confidence trustworthy? Take the top-class
        # probability as a binary forecast of "was the top class correct"
        # and calibrate that. This is the number the human-review gate
        # actually depends on.
        top_conf = proba.max(axis=1)
        correct = (pred == y).astype(float)

        return {
            "n": int(len(y)),
            "accuracy": accuracy(y, pred),
            "majority_baseline": majority,
            "log_loss": log_loss(y, proba),
            "confidence_calibration_error": expected_calibration_error(correct, top_conf),
            "confidence_reliability": reliability_table(correct, top_conf, n_bins=8),
            "per_class": self._per_class(y, pred),
        }

    def _per_class(self, y: np.ndarray, pred: np.ndarray) -> list[dict]:
        rows = []
        for i, cls in enumerate(self.classes):
            tp = int(np.sum((pred == i) & (y == i)))
            fp = int(np.sum((pred == i) & (y != i)))
            fn = int(np.sum((pred != i) & (y == i)))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append({
                "class": cls, "support": int(np.sum(y == i)),
                "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            })
        return rows

    # -- inference ---------------------------------------------------

    def predict_distribution(self, df: pd.DataFrame) -> list[dict[str, float]]:
        """Full posterior per row. The economics layer needs the whole
        distribution (specifically P(fraud)), not just the top class."""
        prepared = prepare(self.surface, df)
        X = self.encoder.transform(prepared)
        proba = self.model.predict_proba(X)
        return [
            {cls: float(row[i]) for i, cls in enumerate(self.classes)}
            for row in proba
        ]

    def predict_one(self, row: dict) -> tuple[str, float, dict[str, float]]:
        """Returns (top class, its probability, full distribution)."""
        dist = self.predict_distribution(pd.DataFrame([row]))[0]
        top = max(dist, key=dist.get)
        return top, dist[top], dist

    # -- serialisation -----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "classes": self.classes,
            "encoder": self.encoder.to_dict(),
            "model": self.model.to_dict(),
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RootCauseClassifier":
        obj = cls(d["surface"])
        obj.classes = list(d["classes"])
        obj.encoder = F.FeatureEncoder.from_dict(d["encoder"])
        obj.model = LogisticModel.from_dict(d["model"])
        obj.metrics = d.get("metrics", {})
        return obj


class RootCauseBundle:
    """All three surfaces' classifiers, saved and loaded as one JSON file."""

    def __init__(self, models: dict[str, RootCauseClassifier] | None = None):
        self.models: dict[str, RootCauseClassifier] = models or {}

    def __getitem__(self, surface: str) -> RootCauseClassifier:
        if surface not in self.models:
            raise KeyError(
                f"no root-cause model for surface {surface!r} — run `python -m src.train`"
            )
        return self.models[surface]

    def save(self, path: str) -> None:
        payload = {
            "kind": "root_cause_bundle",
            "surfaces": {s: m.to_dict() for s, m in self.models.items()},
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "RootCauseBundle":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{path} not found. Train the models first:\n"
                f"    python data/generate_all.py\n"
                f"    python -m src.train"
            ) from exc
        return cls({s: RootCauseClassifier.from_dict(d) for s, d in payload["surfaces"].items()})
