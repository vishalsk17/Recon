"""
Per-action recovery models: P(recovered | event, action).

This is the piece v2 did not have, and the piece everything else in
improvements.md depends on. Without a per-action probability you cannot
compute expected value, you cannot rank actions, you cannot compare against
a baseline, and you cannot tell a wasted retry from a necessary one. All you
can do is map a predicted cause to a fixed action, which is what v2 did.

Structure: a T-learner
----------------------
One independent binary model per action variant, each fitted on the slice of
the exploration log where that action was actually taken. The alternative —
a single model with the action as an input feature (an S-learner) — is more
sample-efficient but, being linear, could only express an *additive* action
effect. It would learn "discounts help by 12 points on average" and would
therefore recommend the same action for everyone, which defeats the entire
purpose. A separate model per action lets the ranking flip between events,
so a reminder can win on one cart and a discount on another.

Two things this model deliberately does **not** do
--------------------------------------------------
1. It does not take the predicted root cause as an input. Stacking the
   cause classifier's output into this model would need out-of-fold
   predictions to avoid train/serve skew, and it buys little here: the raw
   signals that *determine* cause (decline code, ageing bucket, dispute
   flag) are already in the feature set, so each per-action model can learn
   "code 54 responds to a new-method prompt but not to a retry" directly.
2. It does not extrapolate to actions absent from the log. `predict` raises
   on an unknown action rather than guessing. An agent that invents a
   probability for an action it has never observed is exactly the failure
   mode the guardrail layer exists to catch — better to fail loudly.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from . import features as F
from .logistic import LogisticModel, brier_score, expected_calibration_error, reliability_table
from .root_cause import SURFACE_FEATURES, prepare
from ..schemas import CHECKOUT_ABANDONMENT, OVERDUE_RECEIVABLE, PAYMENT_FAILURE

# The action variants each surface's models cover. `@n` carries a
# parameter: hours of delay for a retry, percentage for a discount. These
# must match the `po_*` columns the generators emit.
SURFACE_ACTION_KEYS: dict[str, list[str]] = {
    PAYMENT_FAILURE: [
        "do_nothing",
        "immediate_retry",
        "delayed_retry@12",
        "delayed_retry@48",
        "prompt_new_payment_method",
    ],
    CHECKOUT_ABANDONMENT: [
        "do_nothing",
        "send_reminder_email",
        "send_reminder_whatsapp",
        "offer_bounded_discount@5",
        "offer_bounded_discount@10",
    ],
    OVERDUE_RECEIVABLE: [
        "do_nothing",
        "automated_reminder",
        "automated_reminder_with_payment_plan_offer",
        "escalate_to_collections",
    ],
}

BASELINE_ACTION = "do_nothing"


def split_action_key(key: str) -> tuple[str, float]:
    """'offer_bounded_discount@10' -> ('offer_bounded_discount', 10.0)."""
    if "@" in key:
        base, param = key.split("@", 1)
        return base, float(param)
    return key, 0.0


def make_action_key(action: str, param: float = 0.0) -> str:
    return f"{action}@{param:g}" if param else action


class RecoveryModel:
    """P(recover | event, action) for every action on one surface."""

    def __init__(self, surface: str):
        self.surface = surface
        spec = SURFACE_FEATURES[surface]
        F.assert_no_leakage(spec["numeric"] + spec["categorical"])
        self.encoder = F.FeatureEncoder(spec["numeric"], spec["categorical"])
        self.action_keys = list(SURFACE_ACTION_KEYS[surface])
        self.models: dict[str, LogisticModel] = {}
        self.train_counts: dict[str, int] = {}
        self.metrics: dict[str, Any] = {}

    # -- training ----------------------------------------------------

    def fit(self, log_df: pd.DataFrame) -> "RecoveryModel":
        """Fit on the randomised exploration log.

        `log_df` must contain `logged_action` and `logged_recovered`. The
        oracle `po_*` columns are never read here — that separation is the
        reason the benchmark's numbers mean anything.
        """
        for col in ("logged_action", "logged_recovered"):
            if col not in log_df.columns:
                raise ValueError(f"exploration log is missing {col!r}")

        prepared = prepare(self.surface, log_df)
        # Encoder is fitted across the whole surface so every per-action
        # model shares one feature space and one set of scaling statistics.
        self.encoder.fit(prepared)
        X_all = self.encoder.transform(prepared)
        actions = prepared["logged_action"].astype(str).to_numpy()
        y_all = _as_bool_array(prepared["logged_recovered"])

        for key in self.action_keys:
            mask = actions == key
            n = int(mask.sum())
            self.train_counts[key] = n
            if n < 40:
                raise ValueError(
                    f"only {n} logged observations for action {key!r} on surface "
                    f"{self.surface!r} — too few to fit. Increase N_ROWS in the "
                    f"generator or widen the exploration log."
                )
            X, y = X_all[mask], y_all[mask].astype(np.int64)

            # An action whose logged outcomes are all identical gives a
            # degenerate fit. Rather than let it silently predict 0 or 1
            # forever, fall back to a smoothed constant.
            if y.min() == y.max():
                self.models[key] = _ConstantModel(
                    p=float((y.sum() + 0.5) / (n + 1.0)), n_features=X.shape[1]
                )
                continue

            # Regularisation scales with how little data the slice has.
            l2 = max(4e-3, 3.0 / n)
            self.models[key] = LogisticModel(l2=l2, lr=0.06, max_iter=1600).fit(X, y, n_classes=2)

        return self

    def evaluate(self, log_df: pd.DataFrame) -> dict[str, Any]:
        """Per-action calibration on held-out logged data.

        Calibration is reported instead of only AUC because the economics
        layer multiplies these probabilities by rupee amounts. A model that
        ranks perfectly but is systematically 15 points high would produce
        confident, precise, wrong money figures.
        """
        prepared = prepare(self.surface, log_df)
        X_all = self.encoder.transform(prepared)
        actions = prepared["logged_action"].astype(str).to_numpy()
        y_all = _as_bool_array(prepared["logged_recovered"]).astype(float)

        per_action = []
        all_p, all_y = [], []
        for key in self.action_keys:
            mask = actions == key
            if mask.sum() < 10:
                continue
            p = self.models[key].predict_positive(X_all[mask])
            y = y_all[mask]
            all_p.append(p); all_y.append(y)
            per_action.append({
                "action": key,
                "n": int(mask.sum()),
                "observed_rate": round(float(y.mean()), 4),
                "mean_predicted": round(float(p.mean()), 4),
                "brier": round(brier_score(y, p), 4),
                "calibration_error": round(expected_calibration_error(y, p, 8), 4),
                "auc": round(_auc(y, p), 4),
            })

        p_cat = np.concatenate(all_p) if all_p else np.array([])
        y_cat = np.concatenate(all_y) if all_y else np.array([])
        return {
            "n": int(len(y_cat)),
            "overall_brier": round(brier_score(y_cat, p_cat), 4) if len(y_cat) else None,
            "overall_calibration_error": (
                round(expected_calibration_error(y_cat, p_cat, 10), 4) if len(y_cat) else None
            ),
            "overall_auc": round(_auc(y_cat, p_cat), 4) if len(y_cat) else None,
            "reliability": reliability_table(y_cat, p_cat, 10) if len(y_cat) else [],
            "per_action": per_action,
            "train_counts": dict(self.train_counts),
        }

    # -- inference ---------------------------------------------------

    def predict(self, df: pd.DataFrame, action_key: str) -> np.ndarray:
        """P(recover) under `action_key` for every row in `df`."""
        if action_key not in self.models:
            raise KeyError(
                f"no recovery model for action {action_key!r} on surface {self.surface!r}. "
                f"Known actions: {sorted(self.models)}. Refusing to guess a probability "
                f"for an unobserved action."
            )
        prepared = prepare(self.surface, df)
        X = self.encoder.transform(prepared)
        return self.models[action_key].predict_positive(X)

    def predict_all(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """P(recover) under every known action. One feature encode, reused."""
        prepared = prepare(self.surface, df)
        X = self.encoder.transform(prepared)
        return {key: self.models[key].predict_positive(X) for key in self.action_keys}

    def predict_one_all(self, row: dict) -> dict[str, float]:
        out = self.predict_all(pd.DataFrame([row]))
        return {k: float(v[0]) for k, v in out.items()}

    # -- serialisation -----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "action_keys": self.action_keys,
            "encoder": self.encoder.to_dict(),
            "models": {k: m.to_dict() for k, m in self.models.items()},
            "train_counts": self.train_counts,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RecoveryModel":
        obj = cls(d["surface"])
        obj.action_keys = list(d["action_keys"])
        obj.encoder = F.FeatureEncoder.from_dict(d["encoder"])
        obj.models = {
            k: (_ConstantModel.from_dict(m) if m.get("type") == "constant"
                else LogisticModel.from_dict(m))
            for k, m in d["models"].items()
        }
        obj.train_counts = d.get("train_counts", {})
        obj.metrics = d.get("metrics", {})
        return obj


class RecoveryBundle:
    """All three surfaces' recovery models in one JSON file."""

    def __init__(self, models: dict[str, RecoveryModel] | None = None):
        self.models: dict[str, RecoveryModel] = models or {}

    def __getitem__(self, surface: str) -> RecoveryModel:
        if surface not in self.models:
            raise KeyError(
                f"no recovery model for surface {surface!r} — run `python -m src.train`"
            )
        return self.models[surface]

    def save(self, path: str) -> None:
        payload = {
            "kind": "recovery_bundle",
            "surfaces": {s: m.to_dict() for s, m in self.models.items()},
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "RecoveryBundle":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{path} not found. Train the models first:\n"
                f"    python data/generate_all.py\n"
                f"    python -m src.train"
            ) from exc
        return cls({s: RecoveryModel.from_dict(d) for s, d in payload["surfaces"].items()})


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

class _ConstantModel:
    """Degenerate stand-in for an action slice with no outcome variation.

    Predicts a Laplace-smoothed constant. Exists so that a thin or
    unlucky slice of the log produces a defensible flat estimate instead of
    a fitted model that confidently predicts 0.0 or 1.0 for everyone.
    """

    def __init__(self, p: float, n_features: int):
        self.p = float(min(max(p, 1e-4), 1 - 1e-4))
        self.n_features = int(n_features)
        self.n_classes = 2

    def predict_positive(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.p, dtype=np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_positive(X)
        return np.column_stack([1 - p, p])

    def to_dict(self) -> dict[str, Any]:
        return {"type": "constant", "p": self.p, "n_features": self.n_features, "n_classes": 2}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "_ConstantModel":
        return cls(p=d["p"], n_features=d.get("n_features", 0))


def _as_bool_array(series: pd.Series) -> np.ndarray:
    """CSV round-trips booleans as the strings 'True'/'False'."""
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC via the rank-sum identity, ties averaged."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1, dtype=np.float64)
    # Average ranks within tied probability groups
    sorted_p = p[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
