"""
Regularised multinomial logistic regression, in numpy, serialisable to JSON.

Why not gradient-boosted trees
------------------------------
v2 used sklearn's GradientBoostingClassifier and pickled it with joblib.
This version deliberately does not, for four reasons that all matter more
here than raw accuracy would:

1. **The economics layer spends these probabilities as money.**
   `expected_net_recovery = (p_action - p_baseline) * value - costs` is only
   meaningful if `p` is a calibrated probability rather than an arbitrary
   ranking score. Logistic regression minimises log loss, so its outputs
   are calibrated by construction. A boosted-tree score is typically not,
   and feeding one into this arithmetic silently corrupts every rupee
   figure downstream. src/train.py prints a reliability table so this is
   checked rather than assumed.

2. **The signal here is mostly categorical-lookup shaped.** Root cause is
   driven by decline code, and a linear model over one-hot codes can
   already express an arbitrary per-code score. Trees buy very little; see
   the accuracy-versus-noise-ceiling report in src/train.py.

3. **JSON weights are auditable.** A reviewer can open the model file, read
   the coefficient on `decline_code=59`, and see for themselves why
   suspected fraud is predicted. That is a real advantage under financial
   governance, and a pickled ensemble does not offer it.

4. **No pickle deserialisation.** `joblib.load` on an untrusted file is
   arbitrary code execution. A system whose whole premise is defensive
   should not have a pickle load in its startup path, and this one does
   not — it parses JSON. It also sidesteps the `__main__` pickling trap
   that v2's README had to warn users about.

Fitting is full-batch Adam on the L2-penalised softmax cross-entropy.
The datasets here are small (thousands of rows, tens of features), so
full-batch is both fast and deterministic — and determinism matters when
an auditor asks whether retraining reproduces the same model.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class LogisticModel:
    """Multinomial softmax regression. Binary problems are the 2-class case."""

    def __init__(
        self,
        l2: float = 1e-3,
        lr: float = 0.08,
        max_iter: int = 1200,
        tol: float = 1e-7,
        seed: int = 42,
    ):
        self.l2 = l2
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.W: np.ndarray | None = None   # (n_features, n_classes)
        self.b: np.ndarray | None = None   # (n_classes,)
        self.n_classes: int = 0
        self.final_loss: float = float("nan")
        self.n_iter_: int = 0

    # -- internals ---------------------------------------------------

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)      # stabilise before exp
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # -- API ---------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int | None = None) -> "LogisticModel":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n, d = X.shape
        self.n_classes = int(n_classes or (y.max() + 1))
        k = self.n_classes

        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0.0, 0.01, size=(d, k))
        self.b = np.zeros(k, dtype=np.float64)

        # One-hot targets
        Y = np.zeros((n, k), dtype=np.float64)
        Y[np.arange(n), y] = 1.0

        # Adam state
        mW = np.zeros_like(self.W); vW = np.zeros_like(self.W)
        mb = np.zeros_like(self.b); vb = np.zeros_like(self.b)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        prev_loss = np.inf
        for step in range(1, self.max_iter + 1):
            P = self._softmax(X @ self.W + self.b)
            # Mean cross-entropy plus L2 (bias is not penalised)
            loss = -np.sum(Y * np.log(np.clip(P, 1e-12, 1.0))) / n
            loss += 0.5 * self.l2 * float(np.sum(self.W ** 2))

            diff = (P - Y) / n
            gW = X.T @ diff + self.l2 * self.W
            gb = diff.sum(axis=0)

            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * (gW ** 2)
            mb = beta1 * mb + (1 - beta1) * gb
            vb = beta2 * vb + (1 - beta2) * (gb ** 2)

            mW_hat = mW / (1 - beta1 ** step)
            vW_hat = vW / (1 - beta2 ** step)
            mb_hat = mb / (1 - beta1 ** step)
            vb_hat = vb / (1 - beta2 ** step)

            self.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

            if abs(prev_loss - loss) < self.tol:
                self.n_iter_ = step
                self.final_loss = float(loss)
                break
            prev_loss = loss
        else:
            self.n_iter_ = self.max_iter

        self.final_loss = float(prev_loss)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.W is None or self.b is None:
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=np.float64)
        return self._softmax(X @ self.W + self.b)

    def predict_positive(self, X: np.ndarray) -> np.ndarray:
        """P(class 1) for a binary model — the common case for the uplift
        models, where class 1 means 'recovered'."""
        if self.n_classes != 2:
            raise ValueError("predict_positive requires a 2-class model")
        return self.predict_proba(X)[:, 1]

    # -- serialisation -----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "multinomial_logistic",
            "l2": self.l2,
            "lr": self.lr,
            "max_iter": self.max_iter,
            "seed": self.seed,
            "n_classes": self.n_classes,
            "n_iter": self.n_iter_,
            "final_loss": self.final_loss,
            "W": self.W.tolist() if self.W is not None else None,
            "b": self.b.tolist() if self.b is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LogisticModel":
        m = cls(l2=d.get("l2", 1e-3), lr=d.get("lr", 0.08),
                max_iter=d.get("max_iter", 1200), seed=d.get("seed", 42))
        m.n_classes = int(d["n_classes"])
        m.W = np.asarray(d["W"], dtype=np.float64)
        m.b = np.asarray(d["b"], dtype=np.float64)
        m.final_loss = float(d.get("final_loss", float("nan")))
        m.n_iter_ = int(d.get("n_iter", 0))
        return m


# ---------------------------------------------------------------------
# Evaluation helpers. Kept here so both training scripts and the test
# suite use one implementation.
# ---------------------------------------------------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(proba, dtype=np.float64), 1e-12, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y_true)), y_true])))


def brier_score(y_true_binary: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of a probability forecast.

    Reported alongside calibration because it penalises both
    miscalibration and lack of discrimination, so it cannot be gamed by
    predicting the base rate for everything.
    """
    y = np.asarray(y_true_binary, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def reliability_table(y_true_binary: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Bucket predictions and compare predicted vs observed frequency.

    This is the check that matters most for this system: if the model says
    22% and the observed rate in that bucket is 22%, then multiplying by
    the invoice amount produces a number that means something. If it says
    22% and the truth is 45%, every rupee figure downstream is wrong.
    """
    y = np.asarray(y_true_binary, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": int(mask.sum()),
            "mean_predicted": float(p[mask].mean()),
            "observed_rate": float(y[mask].mean()),
        })
    return rows


def expected_calibration_error(y_true_binary: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Weighted mean absolute gap between predicted and observed rates."""
    rows = reliability_table(y_true_binary, p, n_bins)
    total = sum(r["n"] for r in rows)
    if total == 0:
        return float("nan")
    return float(sum(r["n"] * abs(r["mean_predicted"] - r["observed_rate"]) for r in rows) / total)
