"""
Customer outreach: email and WhatsApp.

This adapter is the last gate before a message would leave the building, so
it owns the message-content rules rather than borrowing them. `src/narrator.py`
imports `validate_customer_message` from here, not the other way round. The
reason is that generation is not the only way a message can arrive — an
operator could hand-write one, a template could be edited, a future caller
could construct an ActionRequest directly — and a rule enforced only at the
generator misses all three. Putting it at the point of no return means every
path is covered by the same implementation.

The content rules are about tone and legality, not spam filtering. A payment
reminder that threatens legal action, mentions a credit score, or manufactures
urgency is a compliance problem and a customer-relations problem regardless of
whether it recovers the money. The forbidden-pattern list in
config/policy.yaml is deliberately conservative and the check is a refusal,
not a warning: a message that trips it is not sent and the refusal is
recorded.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from ..guardrails import in_quiet_hours
from ..schemas import (
    ACTION_CHANNEL, OFFER_BOUNDED_DISCOUNT, OUTREACH_ACTIONS,
    PROMPT_NEW_PAYMENT_METHOD, SEND_REMINDER_EMAIL, SEND_REMINDER_WHATSAPP,
)
from .base import (
    Adapter, ActionRequest, ActionResult, STATUS_SIMULATED, simulated_ref,
)


def validate_customer_message(text: str, cfg: Mapping[str, Any]) -> list[str]:
    """Return a list of problems with a customer-facing message; empty is fine.

    Matching is case-insensitive and on word boundaries where the pattern is
    a phrase, so "Court" and "court" both trip and "courteous" does not. A
    substring check would fire on innocuous words and train whoever maintains
    the list to loosen it, which is the wrong outcome.
    """
    problems: list[str] = []
    llm_cfg = cfg.get("llm", {})
    max_chars = int(llm_cfg.get("max_message_chars", 700))
    body = (text or "").strip()

    if not body:
        problems.append("message is empty")
        return problems
    if len(body) > max_chars:
        problems.append(f"message is {len(body)} characters, over the {max_chars} limit")

    lowered = body.lower()
    for pattern in llm_cfg.get("forbidden_output_patterns", ()):
        needle = str(pattern).lower()
        if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", lowered):
            problems.append(f"contains forbidden phrase {pattern!r}")

    # A message that quotes a rupee figure it was never given is worse than
    # one that omits it, so unresolved template placeholders are a refusal.
    if re.search(r"\{\{?\s*\w+\s*\}?\}", body):
        problems.append("contains an unsubstituted template placeholder")
    return problems


def outreach_refusal(request: ActionRequest, cfg: Mapping[str, Any]) -> Optional[str]:
    """Consent, DND and quiet-hours re-check, shared by every outreach path.

    The guardrail layer has already applied all three. This exists so that
    the invoicing adapter and the messaging adapter enforce them identically
    and neither can drift, and so that the checks survive a caller that
    bypasses the pipeline.
    """
    if request.action not in OUTREACH_ACTIONS:
        return None

    if request.dnd_flagged:
        return ("customer is DND-flagged — no outreach is permitted on any "
                "channel, and this is checked again here rather than trusted "
                "from upstream")

    channel = request.channel or ACTION_CHANNEL.get(request.action, "email")
    consent_required = set(cfg["contact"].get("consent_required_channels", ()))
    # Email is treated as consent-gated too. The config lists the channels
    # where opt-in is legally mandatory; treating the rest as free-for-all
    # would be the wrong default for a system whose whole justification is
    # that it is careful, so consent is required on every channel and the
    # config list only marks which ones are non-negotiable.
    if not request.consented:
        strictness = "legally requires opt-in" if channel in consent_required else "requires opt-in by policy"
        return f"no consent recorded for {channel} — this channel {strictness}"

    start = int(cfg["contact"]["quiet_hours_start"])
    end = int(cfg["contact"]["quiet_hours_end"])
    send_hour = (int(request.local_hour) + int(request.delay_hours)) % 24
    if in_quiet_hours(send_hour, start, end):
        return (f"scheduled send hour {send_hour:02d}:00 falls inside quiet hours "
                f"({start:02d}:00–{end:02d}:00) — upstream should have deferred it")
    return None


class MessagingAdapter(Adapter):
    """Sends — or rather, would send — one message on one channel."""

    name = "messaging"
    handles = frozenset({
        PROMPT_NEW_PAYMENT_METHOD,
        SEND_REMINDER_EMAIL,
        SEND_REMINDER_WHATSAPP,
        OFFER_BOUNDED_DISCOUNT,
    })

    def _authorise(self, request: ActionRequest) -> Optional[str]:
        refusal = outreach_refusal(request, self.cfg)
        if refusal:
            return refusal

        if request.action == OFFER_BOUNDED_DISCOUNT:
            cap = float(self.cfg["limits"]["max_discount_pct"])
            if request.discount_pct <= 0:
                return "a discount action with no discount is a mistake, not an offer"
            if request.discount_pct > cap:
                return (f"discount of {request.discount_pct:g}% exceeds the "
                        f"{cap:g}% cap — refused at the egress boundary as well "
                        f"as upstream")

        if request.message_body:
            problems = validate_customer_message(request.message_body, self.cfg)
            if problems:
                return "message failed content validation: " + "; ".join(problems)
        return None

    def _simulate(self, request: ActionRequest) -> ActionResult:
        channel = request.channel or ACTION_CHANNEL.get(request.action, "email")
        bits = [f"would send via {channel}"]
        if request.discount_pct:
            bits.append(f"including a {request.discount_pct:g}% bounded discount")
        if request.delay_hours:
            bits.append(f"scheduled {request.delay_hours}h out")
        if request.message_body:
            bits.append(f"body {len(request.message_body)} chars, content-validated")
        else:
            bits.append("no body supplied — the narrator was not run for this event")
        return ActionResult(
            status=STATUS_SIMULATED, adapter=self.name, action=request.action,
            provider_ref=simulated_ref(channel, request),
            detail="; ".join(bits),
            scheduled_in_hours=int(request.delay_hours),
            notes=[f"channel={channel}"],
        )
