"""
Data loading: CSV on disk -> the objects the agent actually works with.

Three responsibilities, kept in one place so that training, the agent, the
benchmark and the API all see identical data:

1. **Join the customer dimension onto every event.** Every surface needs
   relationship context, and doing the join here means no downstream module
   can forget to.

2. **Split the oracle columns off.** The `po_*` counterfactual columns are
   lifted out into `SimulatedOutcomes` objects keyed by event id and handed
   back separately. A `RiskEvent` is constructed without them and has no
   field that could hold them. So the decision path cannot read an outcome
   even by accident — the only way to see one is to ask this module for the
   outcomes dict explicitly, which only src/benchmark.py does.

3. **Deterministic train/test splits.** The split is a hash of the event id,
   not a shuffled index. That means it is stable when rows are appended,
   reproducible across machines, and identical whichever module asks for it
   — so "held-out" genuinely means held out from every model, rather than
   depending on whoever called `train_test_split` last.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import pandas as pd

from . import config as C
from .schemas import (
    CHECKOUT_ABANDONMENT, OVERDUE_RECEIVABLE, PAYMENT_FAILURE,
    CustomerProfile, RiskEvent, SimulatedOutcomes,
)

# Columns copied from customers.csv onto every event row.
CUSTOMER_COLUMNS = [
    "segment", "tenure_months", "prior_successful_payments", "prior_late_payments",
    "estimated_annual_value_inr", "gross_margin_pct", "repeat_purchase_probability",
    "prior_coupon_redemptions", "contacts_last_7d", "hours_since_last_contact",
    "email_consent", "whatsapp_consent", "sms_consent", "dnd_flagged",
]

# Per-surface metadata: file, id column, amount column, and the raw
# decision-time fields that become `RiskEvent.features`.
SURFACE_SPEC: dict[str, dict[str, Any]] = {
    PAYMENT_FAILURE: {
        "path": C.PAYMENTS_CSV,
        "id_col": "txn_id",
        "amount_col": "amount",
        "time_col": "failed_at",
        "feature_cols": [
            "payment_method", "bank", "amount", "decline_code",
            "retry_count", "hour_of_day", "day_of_month",
        ],
    },
    CHECKOUT_ABANDONMENT: {
        "path": C.CHECKOUT_CSV,
        "id_col": "event_id",
        "amount_col": "cart_value",
        "time_col": "abandoned_at",
        "feature_cols": [
            "cart_value", "items_count", "device",
            "is_returning_customer", "minutes_since_abandon", "hour_of_day",
            "session_seconds", "reached_payment_page",
            "checkout_errors_logged", "payment_attempts_started",
        ],
    },
    OVERDUE_RECEIVABLE: {
        "path": C.RECEIVABLES_CSV,
        "id_col": "invoice_id",
        "amount_col": "invoice_amount",
        "time_col": "due_date",
        "feature_cols": [
            "invoice_amount", "days_overdue", "prior_late_payments",
            "dispute_flagged_in_ar", "hour_of_day",
            "avg_days_late_last_6_invoices", "purchase_order_required",
            "purchase_order_on_invoice", "invoice_revision_count",
            "amount_variance_vs_contract_pct", "acknowledged_in_portal",
        ],
    },
}

BOOL_COLUMNS = {
    "email_consent", "whatsapp_consent", "sms_consent", "dnd_flagged",
    "is_returning_customer", "dispute_flagged_in_ar", "logged_recovered",
    "is_fraudulent", "reached_payment_page",
    "purchase_order_required", "purchase_order_on_invoice", "acknowledged_in_portal",
}


# ---------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------

def _coerce_bools(df: pd.DataFrame) -> pd.DataFrame:
    """CSV writes booleans as 'True'/'False' strings; convert them back.

    Left as strings, `if row["dnd_flagged"]` would be truthy for the string
    "False" — a consent check that silently always passes. Worth being
    explicit about.
    """
    out = df.copy()
    for col in out.columns:
        if col in BOOL_COLUMNS or col.startswith("po_"):
            if out[col].dtype != bool:
                out[col] = out[col].astype(str).str.lower().isin(["true", "1", "yes"])
    return out


def _require(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{os.path.basename(path)} not found. Generate the dataset first:\n"
            f"    python data/generate_all.py"
        )


def split_tag(event_id: str, test_fraction: float = 0.25, salt: str = "v3") -> str:
    """'train' or 'test', decided by a stable hash of the event id."""
    digest = hashlib.sha256(f"{salt}:{event_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < test_fraction else "train"


# ---------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------

def load_customers_df() -> pd.DataFrame:
    _require(C.CUSTOMERS_CSV)
    return _coerce_bools(pd.read_csv(C.CUSTOMERS_CSV))


def load_customer_profiles() -> dict[str, CustomerProfile]:
    df = load_customers_df()
    profiles: dict[str, CustomerProfile] = {}
    for row in df.to_dict("records"):
        profiles[row["customer_id"]] = CustomerProfile(
            customer_id=row["customer_id"],
            tenure_months=int(row["tenure_months"]),
            prior_successful_payments=int(row["prior_successful_payments"]),
            prior_late_payments=int(row["prior_late_payments"]),
            estimated_annual_value_inr=float(row["estimated_annual_value_inr"]),
            gross_margin_pct=float(row["gross_margin_pct"]),
            repeat_purchase_probability=float(row["repeat_purchase_probability"]),
            prior_coupon_redemptions=int(row["prior_coupon_redemptions"]),
            contacts_last_7d=int(row["contacts_last_7d"]),
            hours_since_last_contact=float(row["hours_since_last_contact"]),
            email_consent=bool(row["email_consent"]),
            whatsapp_consent=bool(row["whatsapp_consent"]),
            sms_consent=bool(row["sms_consent"]),
            dnd_flagged=bool(row["dnd_flagged"]),
            segment=str(row["segment"]),
        )
    return profiles


# ---------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------

def load_surface_df(surface: str, split: str | None = None) -> pd.DataFrame:
    """Event rows with customer columns joined on.

    `split` may be 'train', 'test' or None for everything. The returned
    frame still contains the `po_*` oracle columns — this is the raw
    loader, used by training (which reads only `logged_*`) and by the
    benchmark (which reads `po_*`). Use `load_events` for the agent path,
    which strips them.
    """
    spec = SURFACE_SPEC[surface]
    _require(spec["path"])
    df = _coerce_bools(pd.read_csv(spec["path"]))
    customers = load_customers_df()

    # Only bring across customer columns the event frame does not already
    # have. Receivables carry their own `prior_late_payments` from the AR
    # system, and that is the figure finance would act on — a blind merge
    # would suffix both to `_x`/`_y` and every downstream lookup would break
    # (or worse, silently pick the wrong one). Event-level values win.
    overlapping = [c for c in CUSTOMER_COLUMNS if c in df.columns]
    to_join = [c for c in CUSTOMER_COLUMNS if c not in df.columns]

    df = df.merge(
        customers[["customer_id"] + to_join],
        on="customer_id", how="left", validate="many_to_one",
    )
    if to_join and df[to_join[0]].isna().any():
        missing = int(df[to_join[0]].isna().sum())
        raise ValueError(
            f"{missing} {surface} rows reference a customer_id absent from "
            f"customers.csv — regenerate the dataset with `python data/generate_all.py`"
        )
    if overlapping:
        df.attrs["event_level_customer_columns"] = overlapping

    df["split"] = df[spec["id_col"]].astype(str).map(split_tag)
    if split:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def build_events(surface: str, df: pd.DataFrame) -> list[RiskEvent]:
    """Turn joined rows into RiskEvents, dropping every oracle column."""
    spec = SURFACE_SPEC[surface]
    profiles = load_customer_profiles()
    events: list[RiskEvent] = []

    for row in df.to_dict("records"):
        cid = str(row["customer_id"])
        customer = profiles.get(cid)
        if customer is None:
            # Conservative default: no consent, no relationship value. A
            # missing record must never widen what the agent may do.
            from .schemas import default_customer
            customer = default_customer(cid)

        features = {col: row[col] for col in spec["feature_cols"]}
        # Relationship attributes the models and guardrails need at
        # decision time. Note: no contact addresses, by design.
        for col in CUSTOMER_COLUMNS:
            features[col] = row[col]

        events.append(RiskEvent(
            event_id=str(row[spec["id_col"]]),
            event_type=surface,
            amount_inr=float(row[spec["amount_col"]]),
            customer=customer,
            features=features,
            occurred_at=str(row.get(spec["time_col"], "")),
            occurred_at_hour=int(row.get("hour_of_day", 12)),
        ))
    return events


def extract_outcomes(surface: str, df: pd.DataFrame) -> dict[str, SimulatedOutcomes]:
    """Lift the `po_*` oracle columns into a separate, explicitly-named
    structure. Only src/benchmark.py should ever call this."""
    spec = SURFACE_SPEC[surface]
    po_cols = [c for c in df.columns if c.startswith("po_")]
    out: dict[str, SimulatedOutcomes] = {}
    for row in df.to_dict("records"):
        eid = str(row[spec["id_col"]])
        out[eid] = SimulatedOutcomes(
            event_id=eid,
            outcomes={c[len("po_"):]: bool(row[c]) for c in po_cols},
        )
    return out


def load_events(
    surface: str,
    split: str | None = None,
    limit: int | None = None,
) -> list[RiskEvent]:
    """The agent's entry point. Returns events with no outcome data attached."""
    df = load_surface_df(surface, split)
    if limit is not None and limit < len(df):
        # Deterministic head rather than a random sample, so a --limit demo
        # shows the same cases every time and a screenshot stays valid.
        df = df.head(limit)
    return build_events(surface, df)


def load_all_events(split: str | None = None, limit_per_surface: int | None = None
                    ) -> list[RiskEvent]:
    events: list[RiskEvent] = []
    for surface in (PAYMENT_FAILURE, CHECKOUT_ABANDONMENT, OVERDUE_RECEIVABLE):
        events.extend(load_events(surface, split, limit_per_surface))
    return events


def event_to_feature_row(event: RiskEvent) -> dict[str, Any]:
    """Flatten a RiskEvent into the single-row dict the models expect."""
    row = dict(event.features)
    row["customer_id"] = event.customer.customer_id
    row["hour_of_day"] = event.occurred_at_hour
    return row


def fraud_flags(surface: str = PAYMENT_FAILURE) -> dict[str, bool]:
    """Ground-truth fraud markers, for benchmark chargeback accounting only.

    Separate accessor rather than a column on the event, for the same
    reason the outcomes are: if the agent could read this it would be
    scoring against an oracle rather than a model.
    """
    df = load_surface_df(surface)
    spec = SURFACE_SPEC[surface]
    if "is_fraudulent" not in df.columns:
        return {}
    return {
        str(r[spec["id_col"]]): bool(r["is_fraudulent"])
        for r in df.to_dict("records")
    }
