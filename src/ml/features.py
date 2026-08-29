"""
Feature encoding, shared by training and inference.

The single most common way a model breaks in production is train/serve
skew: a feature computed one way in the training script and another way in
the request path. So there is exactly one implementation of each derived
feature here, and both paths call it.

The encoder is fitted once, serialised into the model JSON alongside the
weights, and reloaded at inference. That means the vocabulary and the
standardisation statistics are pinned to the model that was trained with
them — you cannot accidentally serve new weights against an old vocabulary.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd


class FeatureEncoder:
    """One-hot for categoricals, standardised passthrough for numerics.

    Unseen categories at inference time map to an explicit `__other__`
    column rather than silently taking the first known class (which is
    what v2 did — that quietly turns an unknown issuer into "HDFC" and
    produces a confident, wrong answer). An unknown value should look
    unknown to the model.
    """

    OTHER = "__other__"

    def __init__(self, numeric: list[str], categorical: list[str]):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.vocab: dict[str, list[str]] = {}
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.columns: list[str] = []

    # -- fitting -----------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        for col in self.categorical:
            values = sorted({str(v) for v in df[col].tolist()})
            self.vocab[col] = values + [self.OTHER]
        for col in self.numeric:
            series = pd.to_numeric(df[col], errors="coerce").astype(float)
            self.means[col] = float(series.mean())
            std = float(series.std(ddof=0))
            # A constant column would divide by zero; treat its scale as 1
            # so it contributes only through the intercept.
            self.stds[col] = std if std > 1e-9 else 1.0
        self.columns = self._build_column_names()
        return self

    def _build_column_names(self) -> list[str]:
        cols: list[str] = []
        for col in self.numeric:
            cols.append(f"num:{col}")
        for col in self.categorical:
            for v in self.vocab[col]:
                cols.append(f"cat:{col}={v}")
        return cols

    # -- transform ---------------------------------------------------

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.columns:
            raise RuntimeError("encoder is not fitted")
        n = len(df)
        blocks: list[np.ndarray] = []

        for col in self.numeric:
            series = pd.to_numeric(df[col], errors="coerce").astype(float)
            # A missing numeric becomes the training mean, i.e. zero after
            # standardising — the model treats it as "no information".
            series = series.fillna(self.means[col])
            z = (series.to_numpy() - self.means[col]) / self.stds[col]
            blocks.append(z.reshape(n, 1))

        for col in self.categorical:
            vocab = self.vocab[col]
            index = {v: i for i, v in enumerate(vocab)}
            other_i = index[self.OTHER]
            block = np.zeros((n, len(vocab)), dtype=np.float64)
            for r, raw in enumerate(df[col].tolist()):
                block[r, index.get(str(raw), other_i)] = 1.0
            blocks.append(block)

        return np.hstack(blocks) if blocks else np.zeros((n, 0))

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # -- serialisation -----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric": self.numeric,
            "categorical": self.categorical,
            "vocab": self.vocab,
            "means": self.means,
            "stds": self.stds,
            "columns": self.columns,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureEncoder":
        enc = cls(d["numeric"], d["categorical"])
        enc.vocab = {k: list(v) for k, v in d["vocab"].items()}
        enc.means = {k: float(v) for k, v in d["means"].items()}
        enc.stds = {k: float(v) for k, v in d["stds"].items()}
        enc.columns = list(d["columns"])
        return enc


# =====================================================================
# Derived features
#
# Each function takes a DataFrame of raw decision-time fields and returns a
# copy with engineered columns added. Called identically at train time and
# at inference — see the module docstring.
# =====================================================================

def _safe_log1p(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0).apply(math.log1p)


def _as_int_flag(values: Any, n: int, index: Any) -> pd.Series:
    """Coerce a possibly-missing, possibly-stringy boolean column to 0/1.

    Booleans arrive from three directions with three representations: real
    `bool` from the dataclass path, numpy bool from a coerced frame, and the
    strings "True"/"False" when a CSV has been read without coercion. The
    last is the dangerous one — `bool("False")` is `True`, so a consent or
    document-state check would silently invert. Handling it in one place
    means no feature function can get it wrong individually.
    """
    if values is None:
        return pd.Series([0] * n, index=index)
    series = pd.Series(values, index=index) if not isinstance(values, pd.Series) else values
    if series.dtype == bool:
        return series.astype(int)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)


def add_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features for the failed-payment surface."""
    out = df.copy()

    # Ticket sizes are heavily right-skewed, so the log is far more useful
    # to a linear model than the raw rupee amount.
    out["log_amount"] = _safe_log1p(out["amount"])

    # Whether a retry would land after a likely salary credit. This is the
    # payday signal, and it is the reason delayed retries beat immediate
    # ones on some events and lose on others.
    dom = pd.to_numeric(out.get("day_of_month", 15), errors="coerce").fillna(15)
    out["days_to_month_end"] = (31 - dom).clip(lower=0)
    out["is_late_month"] = (dom >= 26).astype(int)

    # Night-time failures behave differently: more batch/technical faults,
    # and outreach is not permitted anyway.
    hod = pd.to_numeric(out.get("hour_of_day", 12), errors="coerce").fillna(12)
    out["is_night"] = ((hod >= 22) | (hod <= 6)).astype(int)

    out["has_retried"] = (pd.to_numeric(out["retry_count"], errors="coerce").fillna(0) > 0).astype(int)
    return out


def add_checkout_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features for the checkout-abandonment surface."""
    out = df.copy()
    out["log_cart_value"] = _safe_log1p(out["cart_value"])

    items = pd.to_numeric(out["items_count"], errors="coerce").fillna(1).clip(lower=1)
    # Average unit price separates "one expensive thing I'm unsure about"
    # from "a basket of cheap things" — different abandonment psychology.
    out["avg_item_value"] = pd.to_numeric(out["cart_value"], errors="coerce").fillna(0) / items
    out["log_avg_item_value"] = _safe_log1p(out["avg_item_value"])
    out["is_single_item"] = (items <= 1).astype(int)

    mins = pd.to_numeric(out["minutes_since_abandon"], errors="coerce").fillna(60)
    out["log_minutes_since_abandon"] = _safe_log1p(mins)
    out["is_fresh"] = (mins <= 60).astype(int)

    # --- Session telemetry. These four are what make "the checkout broke"
    # distinguishable from "I changed my mind", and the distinction is worth
    # money: a friction cart is recovered by a free reminder, so discounting
    # it is pure margin given away. Without these columns the friction class
    # is recovered at roughly 5% and the agent cannot tell the two apart.
    errors = pd.to_numeric(out.get("checkout_errors_logged", 0), errors="coerce").fillna(0)
    attempts = pd.to_numeric(out.get("payment_attempts_started", 0), errors="coerce").fillna(0)
    session = pd.to_numeric(out.get("session_seconds", 0), errors="coerce").fillna(0)
    reached_int = _as_int_flag(out.get("reached_payment_page"), len(out), out.index)

    out["checkout_errors_logged"] = errors
    out["payment_attempts_started"] = attempts
    out["log_session_seconds"] = _safe_log1p(session)
    out["reached_payment_page_int"] = reached_int
    out["had_checkout_error"] = (errors > 0).astype(int)
    # Reached the payment step and tried to pay: unambiguous intent to buy.
    # This customer does not need a discount, they need the page to work.
    out["intent_signal"] = ((reached_int == 1) & (attempts > 0)).astype(int)
    return out


def add_receivable_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features for the overdue-receivable surface."""
    out = df.copy()
    out["log_invoice_amount"] = _safe_log1p(out["invoice_amount"])

    days = pd.to_numeric(out["days_overdue"], errors="coerce").fillna(0)
    out["log_days_overdue"] = _safe_log1p(days)
    # AR ageing buckets are how finance teams actually think about this,
    # and the reminder-versus-escalation crossover sits between buckets.
    out["age_bucket_0_30"] = ((days >= 0) & (days <= 30)).astype(int)
    out["age_bucket_31_60"] = ((days > 30) & (days <= 60)).astype(int)
    out["age_bucket_61_90"] = ((days > 60) & (days <= 90)).astype(int)
    out["age_bucket_90_plus"] = (days > 90).astype(int)

    late = pd.to_numeric(out["prior_late_payments"], errors="coerce").fillna(0)
    out["is_chronic_late"] = (late >= 3).astype(int)

    # --- Invoice document state. These are what make "this invoice is wrong"
    # distinguishable from "this customer is slow", and the distinction
    # decides whether the event may be chased at all.
    po_required = _as_int_flag(out.get("purchase_order_required"), len(out), out.index)
    po_present = _as_int_flag(out.get("purchase_order_on_invoice"), len(out), out.index)
    out["purchase_order_missing"] = ((po_required == 1) & (po_present == 0)).astype(int)
    out["invoice_revision_count"] = pd.to_numeric(
        out.get("invoice_revision_count", 0), errors="coerce").fillna(0)
    out["invoice_was_revised"] = (out["invoice_revision_count"] > 0).astype(int)
    variance = pd.to_numeric(
        out.get("amount_variance_vs_contract_pct", 0.0), errors="coerce").fillna(0.0)
    out["amount_variance_vs_contract_pct"] = variance
    out["amount_variance_material"] = (variance >= 3.0).astype(int)
    out["acknowledged_in_portal_int"] = _as_int_flag(
        out.get("acknowledged_in_portal"), len(out), out.index)
    # Never acknowledged and never paid usually means it never got matched —
    # a document problem, not a willingness problem.
    out["silent_and_unmatched"] = (
        (out["acknowledged_in_portal_int"] == 0) & (out["purchase_order_missing"] == 1)
    ).astype(int)

    # --- Deviation from the account's own payment norm. A chronic late payer
    # at 90 days overdue is behaving normally; a historically prompt payer at
    # 90 days has had something change. The ratio carries that, and neither
    # column carries it alone.
    norm = pd.to_numeric(
        out.get("avg_days_late_last_6_invoices", 0.0), errors="coerce").fillna(0.0)
    out["avg_days_late_last_6_invoices"] = norm
    out["overdue_vs_history_ratio"] = (days / (1.0 + norm)).clip(upper=40.0)
    return out


def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Relationship features, shared across all three surfaces.

    Requires the customer columns to have been joined in already.
    """
    out = df.copy()
    tenure = pd.to_numeric(out.get("tenure_months", 0), errors="coerce").fillna(0)
    successes = pd.to_numeric(out.get("prior_successful_payments", 0), errors="coerce").fillna(0)
    rp = pd.to_numeric(out.get("repeat_purchase_probability", 0.0), errors="coerce").fillna(0.0)

    out["log_tenure_months"] = _safe_log1p(tenure)
    out["log_prior_successful"] = _safe_log1p(successes)

    # Composite responsiveness score. Actions that need the customer to do
    # something (update a card, click a link) depend on this far more than
    # a silent gateway retry does, so giving the model an explicit summary
    # helps the per-action models separate cleanly.
    out["engagement"] = (
        0.35 * (tenure / 60.0).clip(upper=1.0)
        + 0.35 * (successes / 40.0).clip(upper=1.0)
        + 0.30 * rp
    ).clip(0.0, 1.0)

    out["log_annual_value"] = _safe_log1p(out.get("estimated_annual_value_inr", 0.0))
    out["coupon_affinity"] = (
        pd.to_numeric(out.get("prior_coupon_redemptions", 0), errors="coerce").fillna(0) / 4.0
    ).clip(upper=1.0)
    out["contact_fatigue"] = pd.to_numeric(
        out.get("contacts_last_7d", 0), errors="coerce"
    ).fillna(0)
    return out


# ---------------------------------------------------------------------
# Feature sets per surface. Named explicitly rather than "everything
# numeric in the frame", so that adding a column to a CSV can never
# silently change what the model trains on — including, importantly, never
# letting a `po_*` oracle column or a `true_root_cause` label leak in.
# ---------------------------------------------------------------------

PAYMENT_NUMERIC = [
    "log_amount", "retry_count", "hour_of_day", "day_of_month",
    "days_to_month_end", "is_late_month", "is_night", "has_retried",
]
PAYMENT_CATEGORICAL = ["payment_method", "bank", "decline_code"]

PAYMENT_CUSTOMER_NUMERIC = [
    "log_tenure_months", "log_prior_successful", "engagement",
    "log_annual_value", "prior_late_payments",
]

CHECKOUT_NUMERIC = [
    "log_cart_value", "items_count", "log_avg_item_value", "is_single_item",
    "log_minutes_since_abandon", "is_fresh", "hour_of_day",
    # Session telemetry — see add_checkout_features for why these matter.
    "checkout_errors_logged", "payment_attempts_started", "log_session_seconds",
    "reached_payment_page_int", "had_checkout_error", "intent_signal",
]
CHECKOUT_CATEGORICAL = ["device"]
CHECKOUT_CUSTOMER_NUMERIC = [
    "log_tenure_months", "engagement", "coupon_affinity",
    "log_annual_value", "contact_fatigue",
]

RECEIVABLE_NUMERIC = [
    "log_invoice_amount", "log_days_overdue", "days_overdue",
    "age_bucket_0_30", "age_bucket_31_60", "age_bucket_61_90", "age_bucket_90_plus",
    "prior_late_payments", "is_chronic_late",
    # Invoice document state — separates "the invoice is wrong" from "the
    # customer is slow". See add_receivable_features for why this matters.
    "purchase_order_missing", "invoice_revision_count", "invoice_was_revised",
    "amount_variance_vs_contract_pct", "amount_variance_material",
    "acknowledged_in_portal_int", "silent_and_unmatched",
    # Deviation from the account's own norm — separates cash-flow strain
    # from habitual lateness.
    "avg_days_late_last_6_invoices", "overdue_vs_history_ratio",
]
RECEIVABLE_CATEGORICAL = ["segment"]
RECEIVABLE_CUSTOMER_NUMERIC = ["log_tenure_months", "engagement", "log_annual_value"]


# Columns that must never appear in any feature set. Asserted in
# tests/test_defensive_posture.py, because label leakage is the failure
# mode that makes a demo look brilliant and a production system useless.
#
# `logged_` covers two columns and both belong here. `logged_recovered` is the
# outcome. `logged_action` is the *treatment* the exploration log happened to
# assign, which is subtler and worth spelling out: RecoveryModel.fit reads it
# by name to split the log into per-action models, and that is correct — it is
# the arm, not a covariate. Using it as a feature would be a different thing
# entirely. The recovery model exists to answer "what if we did X", so a design
# that also told it which action was in fact taken would let it score the
# factual arm by looking the answer up, and every counterfactual would be a
# guess dressed in the same confidence. It reads the column from the frame and
# never routes it through this check, so widening the prefix costs nothing and
# closes the door on anyone adding it to SURFACE_FEATURES later.
FORBIDDEN_FEATURE_PREFIXES = ("po_", "true_root_cause", "logged_", "is_fraudulent")


def assert_no_leakage(feature_names: Iterable[str]) -> None:
    bad = [
        f for f in feature_names
        if any(f.startswith(p) or f == p for p in FORBIDDEN_FEATURE_PREFIXES)
    ]
    if bad:
        raise ValueError(f"outcome/label columns leaked into features: {bad}")
