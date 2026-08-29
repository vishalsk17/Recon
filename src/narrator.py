"""
The one place a language model is allowed to speak, and the leash it wears.

improvements.md item 5 asks for the LLM to be given a tightly bounded role.
This module is that boundary, and the boundary is drawn at a specific place:
**the model may put already-decided facts into words, and nothing else.**

Concretely, it may not:

  * choose an action, a channel, a discount, or a delay — those come from
    src/economics.py and src/guardrails.py and are already fixed by the time
    this module is called;
  * see or invent a number — every figure in the prompt is a rendered fact
    from a `Decision`, and any figure in the output that was not in the
    prompt is a hallucination the validator rejects;
  * call a tool. No tool definitions are sent. There is no function-calling
    loop here, so there is no path from generated text to a side effect;
  * reach a customer directly. Its output is a *draft*, returned to the
    caller, re-validated at the egress boundary, and — for anything
    customer-facing — held behind the same approval queue as the action.

Three things about the design are worth stating plainly, because they are the
difference between "we prompt carefully" and "it cannot do the bad thing".

**There is no untrusted-text path into the prompt.** Everything in a fact
sheet is either a float we computed, an enum from a closed vocabulary, or a
string we wrote ourselves. No customer-supplied free text — no support ticket
body, no dispute note, no payment description — is ever interpolated. That
matters because prompt injection needs an injection *site*, and there isn't
one. It is a structural property of `build_fact_sheet`, which is why that
function is written as an explicit field-by-field render rather than a
convenient `json.dumps(decision.to_dict())`. The convenient version would
have worked today and become a vulnerability the first time somebody added a
free-text column upstream.

**Validation runs on the output, and it is the same validator the adapter
uses.** `validate_customer_message` is imported from src/adapters/messaging.py
rather than reimplemented, so a phrase added to the forbidden list protects
both paths at once. Two copies of a safety list is one copy plus a bug
waiting for someone to update only the other one. The check runs here *and*
again at egress — not because the first one is unreliable, but because the
narrator is not the only thing that can produce a message body.

**A failed draft is an error, not a fallback.** There is no template the code
quietly drops back to. A silent fallback is the worst of both worlds: the
operator believes they are reading model output and they are not, and a
misconfigured key produces plausible text instead of a stack trace. So a
missing `ANTHROPIC_API_KEY` raises, a refused draft raises, and a run that
needed narration and did not get it fails visibly. The agent itself does not
depend on this module at all — `python -m src.agent run` never imports it, and
recovery decisions are made and executed with no language model in the
process. Narration is an operator convenience layered on top, which is the
correct dependency direction for a system that moves money.

Transport is `urllib.request` from the standard library, wrapped in a
`_transport` seam so tests exercise validation and prompt construction
without network access. The seam is a module-level function rather than a
constructor argument on purpose: a test that forgets to inject a fake would
otherwise reach for the real API, and here it cannot, because
`Narrator.__init__` refuses to build without a key and the test suite has
none.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from . import config as C
from .adapters.messaging import validate_customer_message
from .schemas import (
    ACTION_CHANNEL, CHECKOUT_ABANDONMENT, Decision, OVERDUE_RECEIVABLE,
    PAYMENT_FAILURE, RiskEvent, SEGMENTS,
)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

# The *name* of the environment variable the key is read from — never a key.
#
# This line briefly held a real `sk-ant-...` credential, pasted over the
# constant. Three things went wrong at once, which is why the comment is here
# and a test now pins the value:
#
#   1. A live secret sat in the source tree.
#   2. `os.environ.get(ENV_KEY)` looked up a variable *named* after the secret,
#      which nothing sets, so narration refused every request — the failure
#      looked like a missing key rather than a corrupted constant.
#   3. Worse, the refusal message below interpolates `ENV_KEY`, so the secret
#      was printed to stdout and returned in the 503 body to anyone who called
#      `POST /api/narrate`. A credential in source is a leak; a credential in
#      an error path is a leak with a delivery mechanism.
#
# Credentials belong in the environment. Nothing in this repository should ever
# contain a value matching a provider key format — `test_defensive_posture.py`
# enforces that across every tracked file.
ENV_KEY = "ANTHROPIC_API_KEY"

# Roles are also listed in config/policy.yaml. Both are checked: config is
# the operator's switch for turning a role off, this tuple is the set of
# roles the code actually knows how to build a prompt for. A role in config
# but not here is a configuration error, not a silent no-op.
IMPLEMENTED_ROLES: tuple[str, ...] = (
    "draft_customer_message",
    "summarise_case_for_reviewer",
    "explain_decision_plainly",
)

CUSTOMER_FACING_ROLES: frozenset[str] = frozenset({"draft_customer_message"})

SURFACE_WORDS = {
    PAYMENT_FAILURE: "a payment that did not go through",
    CHECKOUT_ABANDONMENT: "a checkout that was not completed",
    OVERDUE_RECEIVABLE: "an invoice that is past its due date",
}


class NarratorError(RuntimeError):
    """Base class for every way narration can refuse."""


class MissingCredentials(NarratorError):
    """No API key. Deliberately fatal — see the module docstring."""


class RoleNotPermitted(NarratorError):
    """The caller asked for a role outside the configured allow-list."""


class DraftRejected(NarratorError):
    """The model produced text that failed validation.

    Carries the problems so the operator sees *why* rather than just that
    something went wrong, and so a reviewer can tell a length overrun from
    a forbidden phrase.
    """

    def __init__(self, problems: Sequence[str], text: str):
        self.problems = list(problems)
        self.text = text
        super().__init__("generated text was refused: " + "; ".join(self.problems))


class TransportFailed(NarratorError):
    """The API call did not succeed after the configured retries."""


# ---------------------------------------------------------------------
# Fact sheets
# ---------------------------------------------------------------------

def _money(x: float) -> str:
    return f"{float(x):,.0f} INR"


def build_fact_sheet(decision: Decision, event: Optional[RiskEvent] = None) -> str:
    """Render a decision as a flat list of facts the model may draw on.

    Written field by field, from typed values only. Adding a field here is a
    deliberate act; that is the point. See the module docstring on why this
    is not `json.dumps(decision.to_dict())` — the audit dict carries free-form
    `notes` strings and a full considered set, and while both are currently
    written by this codebase, a render that walks whatever happens to be in a
    dict is a render that will one day walk something a customer wrote.
    """
    chosen = decision.chosen
    lines = [
        f"surface: {decision.event_type} ({SURFACE_WORDS.get(decision.event_type, 'a risk event')})",
        f"amount at stake: {_money(decision.amount_inr)}",
        f"most likely reason: {decision.root_cause}",
        f"confidence in that reason: {decision.root_cause_confidence:.0%}",
        f"action decided: {chosen.candidate.action}",
    ]
    if chosen.candidate.channel:
        lines.append(f"channel: {chosen.candidate.channel}")
    if chosen.candidate.discount_pct:
        lines.append(f"discount authorised: {chosen.candidate.discount_pct:.0f}%")
    if chosen.candidate.delay_hours:
        lines.append(f"scheduled to happen in: {chosen.candidate.delay_hours} hours")
    lines += [
        f"estimated chance this recovers the money: {chosen.p_recover:.0%}",
        f"chance it recovers on its own with no action: {chosen.p_recover_baseline:.0%}",
        f"expected net recovery over doing nothing: {_money(chosen.expected_net_recovery_inr)}",
        f"needs human sign-off: {'yes' if decision.requires_human_approval else 'no'}",
    ]
    if decision.approval_reason:
        lines.append(f"reason sign-off is needed: {decision.approval_reason}")
    if chosen.probability_is_assumed:
        lines.append("note: the recovery chance above is a stated assumption, "
                     "not a fitted estimate")

    blocked = [s for s in decision.considered if s.blocked_by]
    if blocked:
        lines.append(f"options refused by policy: {len(blocked)}")
        for s in blocked[:4]:
            lines.append(f"  - {s.candidate.action}: {s.blocked_by[0]}")

    runners = [s for s in decision.considered
               if s.allowed and s.candidate.action != chosen.candidate.action]
    runners.sort(key=lambda s: -s.expected_net_recovery_inr)
    for s in runners[:2]:
        lines.append(f"option not taken: {s.candidate.action} at "
                     f"{_money(s.expected_net_recovery_inr)}")

    if event is not None:
        cust = event.customer
        # Relationship context only. No name, no address, no contact detail —
        # CustomerProfile does not carry any, which is what makes this safe
        # to hand to a third-party API at all.
        #
        # `segment` is the one field here that is a string rather than a number,
        # so it is the one field that could carry instruction-shaped text if an
        # upstream system ever put something unexpected in it. It is rendered
        # through the closed vocabulary rather than interpolated, which keeps
        # this function's guarantee — every line is a number we computed or a
        # string we wrote — true by construction rather than by assumption about
        # the data. An unrecognised value is reported as unspecified, because
        # the alternative is passing it to a model to read.
        segment = cust.segment if cust.segment in SEGMENTS else "unspecified"
        lines += [
            f"customer segment: {segment}",
            f"relationship length: {cust.tenure_months} months",
            f"successful payments to date: {cust.prior_successful_payments}",
            f"times contacted in the last 7 days: {cust.contacts_last_7d}",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

_SHARED_RULES = """
Rules that apply to every response:
- Use only the facts given. Do not introduce any number, name, date, offer or
  commitment that is not in the fact sheet.
- Do not mention legal action, courts, police, credit scores, blacklists or
  any consequence of non-payment.
- Do not invent urgency. No deadlines, no "final notice", no "act now".
- Do not apologise on the company's behalf for something the facts do not
  establish, and do not admit fault.
- Return the requested text only. No preamble, no sign-off block, no markdown,
  no explanation of what you produced.
""".strip()

_ROLE_PROMPTS: dict[str, str] = {
    "draft_customer_message": """
You are drafting one short message to a customer on behalf of a merchant, about
{surface_words}.

The action has already been decided by a system you are not part of, and you
cannot change it. Write only the message body.

Requirements:
- Plain, warm, matter-of-fact. Indian English. No emoji.
- At most 4 short sentences, under {max_chars} characters.
- Address the customer directly as "you". You have not been told their name,
  so do not use one and do not write a placeholder for one.
- If a discount is authorised, state it plainly once. If none is authorised,
  do not hint that one might be available.
- Make it easy to act, and make it clear they do not have to.
""".strip(),
    "summarise_case_for_reviewer": """
You are writing a note for the person who has to approve or reject this
decision. They are an experienced revenue-operations analyst, so be brief and
do not explain basics.

Requirements:
- 2 to 4 sentences of prose. No bullet points.
- Say what is at stake, what the system decided, and the single thing most
  worth checking before releasing it.
- If policy refused options, say which constraint bound and why that matters
  to the reviewer's judgement.
- Do not recommend approval or rejection. That is their call, not yours.
""".strip(),
    "explain_decision_plainly": """
You are explaining this decision to someone non-technical inside the merchant's
business — a finance manager who wants to know why the system did what it did.

Requirements:
- 2 to 4 sentences of prose. No jargon, no bullet points.
- Explain the reasoning, including why the alternatives were worse or refused.
- Be honest about uncertainty where the fact sheet shows it.
- Do not describe the system as certain, intelligent, or guaranteed to work.
""".strip(),
}


def build_prompt(role: str, fact_sheet: str, cfg: Mapping[str, Any],
                 decision: Optional[Decision] = None) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for a permitted role."""
    if role not in _ROLE_PROMPTS:
        raise RoleNotPermitted(f"no prompt is implemented for role {role!r}")
    llm_cfg = cfg.get("llm", {})
    surface_words = SURFACE_WORDS.get(
        decision.event_type if decision else "", "a revenue-recovery case")
    system = _ROLE_PROMPTS[role].format(
        surface_words=surface_words,
        max_chars=int(llm_cfg.get("max_message_chars", 700)),
    ) + "\n\n" + _SHARED_RULES
    user = "Fact sheet:\n" + fact_sheet
    return system, user


# ---------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------

def _transport(payload: Mapping[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    """POST to the Messages API with the standard library. Replaceable in tests.

    Deliberately minimal: one endpoint, no streaming, no tools, no retries
    (the caller owns those so the backoff is visible), and the response is
    parsed as JSON with no evaluation of any kind.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API_URL, data=body, method="POST",
        headers={
            "content-type": "application/json",
            "anthropic-version": API_VERSION,
            "x-api-key": api_key,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_text(response: Mapping[str, Any]) -> str:
    """Pull the text out of a Messages API response, tolerating block order."""
    blocks = response.get("content") or []
    parts = [b.get("text", "") for b in blocks
             if isinstance(b, Mapping) and b.get("type") == "text"]
    text = "\n".join(p for p in parts if p).strip()
    if not text:
        raise DraftRejected(["the API returned no text content"], "")
    return text


# ---------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _normalise_number(token: str) -> str:
    """Strip formatting from a numeric token without changing its magnitude.

    Commas go, and a trailing run of zeros goes **only after a decimal point**,
    so "1,250.50" and "1250.5" compare equal while "1250" stays "1250".

    That last clause is the whole point of this function existing. The first
    version was `token.replace(",", "").rstrip(".0").rstrip(".")`, which strips
    trailing zeros from integers too: "40" became "4", "87,650" became "8765",
    and "40,000" became "4". Every figure that differed only by trailing zeros
    collapsed onto the same string, so a fact sheet quoting 40,000 INR
    authorised a draft offering "40% off" — and a draft inventing "4,000,000
    INR" as well. Two tests found it independently, which is the only reason it
    is not still there: the check looked like it was working, because the
    obvious fabrications ("35%") have no trailing zero and were caught.
    """
    cleaned = token.replace(",", "")
    if "." in cleaned:
        cleaned = cleaned.rstrip("0").rstrip(".")
    return cleaned or "0"


def _numbers_in(text: str) -> set[str]:
    return {_normalise_number(m.group(0)) for m in _NUMBER.finditer(text)}


def unsupported_numbers(text: str, fact_sheet: str) -> list[str]:
    """Numbers in the output that do not appear in the fact sheet.

    A recovery message that quotes a figure nobody authorised is the specific
    failure mode worth spending code on: it is the one that survives a human
    skim, because the sentence reads perfectly well. Percentages and rupee
    amounts in a customer message are commitments, so any digit that cannot
    be traced to an input is treated as a defect.

    The comparison is deliberately loose about formatting (commas stripped,
    trailing zeros trimmed) and deliberately strict about provenance. Small
    integers up to twelve are exempt: they are ordinary prose ("a couple of
    days", "3 sentences") and treating them as commitments would reject
    perfectly good drafts and teach whoever maintains this to switch the
    check off.
    """
    allowed = _numbers_in(fact_sheet)
    problems = []
    for token in sorted(_numbers_in(text)):
        if token in allowed:
            continue
        try:
            if float(token) <= 12:
                continue
        except ValueError:
            pass
        problems.append(f"quotes {token!r}, which is not in the fact sheet")
    return problems


def validate_draft(text: str, role: str, fact_sheet: str,
                   cfg: Mapping[str, Any]) -> list[str]:
    """Every check that applies to generated text, in one place."""
    problems: list[str] = []
    body = (text or "").strip()
    if not body:
        return ["the model returned nothing"]

    if role in CUSTOMER_FACING_ROLES:
        # The shipped egress validator, not a copy of it.
        problems += validate_customer_message(body, cfg)
    else:
        # Internal text is not sent to a customer, so length and tone rules
        # are relaxed — but a fabricated number is just as wrong on a
        # reviewer's screen as on a customer's phone, and arguably worse,
        # because the reviewer is about to release money on the strength of it.
        max_chars = int(cfg.get("llm", {}).get("max_message_chars", 700)) * 2
        if len(body) > max_chars:
            problems.append(f"note is {len(body)} characters, over the {max_chars} limit")
        if re.search(r"\{\{?\s*\w+\s*\}?\}", body):
            problems.append("contains an unsubstituted template placeholder")

    problems += unsupported_numbers(body, fact_sheet)
    return problems


# ---------------------------------------------------------------------
# Narrator
# ---------------------------------------------------------------------

@dataclass
class Draft:
    """A validated piece of generated text, with its provenance."""
    role: str
    text: str
    model: str
    fact_sheet: str
    attempts: int
    latency_ms: float
    validated: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "model": self.model,
            "attempts": self.attempts,
            "latency_ms": round(self.latency_ms, 1),
            "validated": self.validated,
            "warnings": list(self.warnings),
            # The fact sheet is recorded alongside the text so that "could the
            # model have known this" is answerable later without re-deriving
            # the decision.
            "fact_sheet": self.fact_sheet,
        }


class Narrator:
    """Bounded access to a language model. Construct once, reuse."""

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None,
                 api_key: Optional[str] = None,
                 transport: Optional[Callable[..., dict[str, Any]]] = None,
                 max_attempts: int = 2,
                 timeout: float = 30.0):
        self.cfg = cfg or C.load_config()
        self.transport = transport or _transport
        self.max_attempts = max(1, int(max_attempts))
        self.timeout = float(timeout)

        llm_cfg = self.cfg.get("llm", {})
        self.model = str(llm_cfg.get("model", "claude-sonnet-4-5"))
        self.max_tokens = int(llm_cfg.get("max_tokens", 700))
        configured = tuple(llm_cfg.get("permitted_roles", ()))
        unknown = [r for r in configured if r not in IMPLEMENTED_ROLES]
        if unknown:
            raise RoleNotPermitted(
                f"config permits roles this module cannot build a prompt for: "
                f"{unknown}. Implemented roles are {list(IMPLEMENTED_ROLES)}."
            )
        self.permitted_roles = frozenset(configured)

        key = api_key if api_key is not None else os.environ.get(ENV_KEY, "")
        if not key.strip():
            raise MissingCredentials(
                f"{ENV_KEY} is not set. Narration requires a real API key by "
                f"design — there is no template fallback, because a fallback "
                f"would let a misconfigured deployment produce plausible text "
                f"that nobody generated. Set {ENV_KEY}, or do not call the "
                f"narration endpoints; the recovery pipeline itself does not "
                f"need them and will run without a key."
            )
        self._api_key = key.strip()

    # -- the only public entry point ----------------------------------

    def narrate(self, role: str, decision: Decision,
                event: Optional[RiskEvent] = None) -> Draft:
        """Generate one validated piece of text for a decided case.

        Raises rather than returning something unusable. On a validation
        failure the request is retried once with the problems fed back, and
        if the second attempt also fails the exception carries both the text
        and the reasons.
        """
        if role not in self.permitted_roles:
            raise RoleNotPermitted(
                f"role {role!r} is not in the configured allow-list "
                f"{sorted(self.permitted_roles)}"
            )
        fact_sheet = build_fact_sheet(decision, event)
        system, user = build_prompt(role, fact_sheet, self.cfg, decision)

        started = time.perf_counter()
        problems: list[str] = []
        text = ""
        for attempt in range(1, self.max_attempts + 1):
            messages = [{"role": "user", "content": user}]
            if problems:
                # One corrective turn. The correction states the rule that was
                # broken; it never supplies the missing fact, because supplying
                # it here would be this module deciding something.
                messages += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content":
                        "That draft was rejected for the following reasons:\n"
                        + "\n".join(f"- {p}" for p in problems)
                        + "\n\nRewrite it so none of them apply. Use only the "
                          "facts already given; do not add anything new."},
                ]
            response = self._call({
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system,
                "messages": messages,
                # No `tools` key. The model has no capabilities in this call
                # beyond emitting text, and that is not a setting to be
                # loosened later — see the module docstring.
            })
            text = _extract_text(response)
            problems = validate_draft(text, role, fact_sheet, self.cfg)
            if not problems:
                return Draft(
                    role=role, text=text, model=self.model, fact_sheet=fact_sheet,
                    attempts=attempt,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
        raise DraftRejected(problems, text)

    # -- transport with visible backoff -------------------------------

    def _call(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        last: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                return self.transport(payload, self._api_key, self.timeout)
            except urllib.error.HTTPError as exc:
                # 4xx other than 429 will not fix themselves, so do not retry
                # them — retrying a bad request is how a quota gets burned.
                if exc.code != 429 and 400 <= exc.code < 500:
                    detail = ""
                    try:
                        detail = exc.read().decode("utf-8")[:400]
                    except Exception:
                        pass
                    raise TransportFailed(
                        f"the API rejected the request with HTTP {exc.code}. {detail}"
                    ) from exc
                last = exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
            if attempt < 3:
                time.sleep(0.75 * attempt)
        raise TransportFailed(
            f"could not reach the Messages API after 3 attempts: {last}"
        ) from last


def available(cfg: Optional[Mapping[str, Any]] = None) -> bool:
    """Whether narration could run, without constructing anything.

    Used by the dashboard to grey out the narration controls instead of
    offering a button that will return a 503. Checking for the key rather
    than trying and catching keeps the failure out of the logs, where it
    would look like a fault rather than a configuration choice.
    """
    return bool(os.environ.get(ENV_KEY, "").strip())
