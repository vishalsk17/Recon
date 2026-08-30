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
  * call a tool. No tool definitions are sent, in any provider's dialect.
    There is no function-calling loop here, so there is no path from
    generated text to a side effect;
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
missing API key raises, a refused draft raises, and a run that needed
narration and did not get it fails visibly. The agent itself does not depend
on this module at all — `python -m src.agent run` never imports it, and
recovery decisions are made and executed with no language model in the
process. Narration is an operator convenience layered on top, which is the
correct dependency direction for a system that moves money.

---

## Any provider's key, not one vendor's

This module is deliberately not tied to a single vendor. It speaks three
request/response dialects — Anthropic Messages, OpenAI chat-completions, and
Google Gemini `generateContent` — which between them cover Claude, GPT,
Gemini, Groq, Mistral, DeepSeek, xAI, OpenRouter, and anything else exposing
an OpenAI-compatible endpoint (vLLM, Ollama, Together, a private gateway).

`PROVIDERS` is the registry. Each entry is a frozen dataclass naming the
environment variables the key may arrive in, the endpoint, the header that
carries the key, and which dialect to speak. Adding a provider is a data
change, not a code change, which is the point: there is no per-vendor branch
anywhere in the request path, so a new provider cannot come with a new
capability attached.

Resolution is either explicit or by discovery:

  * `llm.provider: openai` in config/policy.yaml pins one provider. If its
    key is absent, narration refuses — it does not quietly try another,
    because silently talking to a different vendor than the operator
    configured is exactly the class of surprise this project avoids.
  * `llm.provider: auto` (the default) scans `PROVIDERS` in declared order
    and uses the first whose environment variable is set. The winner is
    recorded on every `Draft`, so "which model wrote this" is answerable from
    the record rather than from the environment six months later.

Three properties are enforced rather than documented, because each one is a
way a multi-provider design could become less safe than a single-vendor one:

**An environment variable name is matched against a whitelist, not a
blacklist.** `_validate_registry` runs at import and requires every name in
the registry to match `[A-Z][A-Z0-9_]{2,63}` — uppercase, digits and
underscores only. This is the structural repair for the incident described in
SECURITY.md, where a real `sk-ant-...` key was pasted over the constant that
is supposed to hold a variable's *name*, and was then echoed to callers by the
refusal message. A pasted credential fails that pattern on its first
character. Nine regexes asking "does this look like a secret" can be evaded;
"is this a legal environment variable name" cannot, and it is the property
actually wanted. Multiplying providers multiplies these constants, so the
check had to stop being a per-constant assertion and become an invariant over
the registry.

**The endpoint is bounded.** `llm.base_url` and `LLM_BASE_URL` exist so an
arbitrary OpenAI-compatible gateway can be used, and an unconstrained URL in
the one module that assembles customer relationship context into a payload
would be a data-exfiltration setting. So `_check_endpoint` requires `https`,
or `http` only to `127.0.0.1`, `::1` or `localhost` — the same loopback
whitelist `server.serve()` uses, for the same reason. Credentials in the
userinfo position (`https://user:pass@host`) and query strings are both
refused, the latter because a key in a URL ends up in every log that records
one. Gemini accepts its key as `?key=`; this module always uses the
`x-goog-api-key` header instead.

**The refusal names variables and never values.** `MissingCredentials` lists
the environment variables it looked in, which is the useful thing to tell an
operator, and every name in that list has passed the whitelist above. The key
itself is never interpolated into any message, exception, or `Draft`.

Transport is `urllib.request` from the standard library, wrapped in a
`_transport` seam so tests exercise validation and prompt construction
without network access. There is exactly one `urlopen` call in this module and
every provider goes through it. The seam is a module-level function rather
than a constructor argument on purpose: a test that forgets to inject a fake
would otherwise reach for a real API, and here it cannot, because
`Narrator.__init__` refuses to build without a key and the test suite unsets
every recognised variable before it runs.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from . import config as C
from .adapters.messaging import validate_customer_message
from .schemas import (
    ACTION_CHANNEL, CHECKOUT_ABANDONMENT, Decision, OVERDUE_RECEIVABLE,
    PAYMENT_FAILURE, RiskEvent, SEGMENTS,
)

# ---------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------

# Dialects. Three request/response shapes cover every provider below; the
# string is looked up in `_PAYLOAD_BUILDERS` and `_TEXT_EXTRACTORS`, so an
# unknown dialect is an import-time error rather than a runtime surprise.
ANTHROPIC = "anthropic"
OPENAI = "openai"
GEMINI = "gemini"

# Environment variable names must look like environment variable names.
#
# This is the whitelist that makes the credential-in-a-constant incident
# structurally impossible rather than merely tested for. See the module
# docstring; the short version is that a pasted `sk-ant-...` fails on its
# first character, and no blacklist of key formats can make that promise.
_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]{2,63}")

# http is permitted only to these hosts, exactly as in server.serve(). A local
# model server (Ollama, vLLM, LM Studio) is a legitimate provider and does not
# have a certificate; anything off-machine must be encrypted.
LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})


@dataclass(frozen=True)
class Provider:
    """Everything that differs between vendors, and nothing that does not.

    Frozen because it is configuration masquerading as code: the request path
    reads these fields and never writes them, and a provider that could be
    mutated at runtime would make the audited `Draft.provider` a claim about
    the past rather than a fact.
    """

    name: str
    # In priority order. The first is canonical and is the one named in
    # documentation and error messages; later ones are aliases the ecosystem
    # actually uses (GOOGLE_API_KEY for Gemini, for instance).
    env_keys: tuple[str, ...]
    url: str
    default_model: str
    dialect: str
    # The header that carries the key, and what goes in front of it. Never a
    # query parameter: a key in a URL is a key in every access log.
    auth_header: str = "authorization"
    auth_prefix: str = "Bearer "
    extra_headers: tuple[tuple[str, str], ...] = ()
    # Key prefixes used to identify a provider from an explicitly passed key
    # when no environment variable is set. Longest match wins, so "sk-ant-"
    # is never shadowed by "sk-".
    key_prefixes: tuple[str, ...] = ()
    # OpenAI renamed this field; its compatible imitators mostly did not.
    max_tokens_field: str = "max_tokens"
    # True when the model name goes in the URL rather than the body (Gemini).
    model_in_url: bool = False
    # True when the endpoint must be supplied by the operator.
    needs_base_url: bool = False


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        name="anthropic",
        env_keys=("ANTHROPIC_API_KEY",),
        url="https://api.anthropic.com/v1/messages",
        default_model="claude-sonnet-4-5",
        dialect=ANTHROPIC,
        auth_header="x-api-key",
        auth_prefix="",
        extra_headers=(("anthropic-version", "2023-06-01"),),
        key_prefixes=("sk-ant-",),
    ),
    Provider(
        name="openai",
        env_keys=("OPENAI_API_KEY",),
        url="https://api.openai.com/v1/chat/completions",
        default_model="gpt-4o-mini",
        dialect=OPENAI,
        key_prefixes=("sk-proj-", "sk-"),
        # Reasoning models reject `max_tokens` outright; chat models accept
        # both. The newer name is the one that works on all of them.
        max_tokens_field="max_completion_tokens",
    ),
    Provider(
        name="gemini",
        env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        url="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        default_model="gemini-3.6-flash",
        dialect=GEMINI,
        auth_header="x-goog-api-key",
        auth_prefix="",
        key_prefixes=("AIza",),
        model_in_url=True,
    ),
    Provider(
        name="groq",
        env_keys=("GROQ_API_KEY",),
        url="https://api.groq.com/openai/v1/chat/completions",
        default_model="llama-3.3-70b-versatile",
        dialect=OPENAI,
        key_prefixes=("gsk_",),
    ),
    Provider(
        name="mistral",
        env_keys=("MISTRAL_API_KEY",),
        url="https://api.mistral.ai/v1/chat/completions",
        default_model="mistral-small-latest",
        dialect=OPENAI,
    ),
    Provider(
        name="deepseek",
        env_keys=("DEEPSEEK_API_KEY",),
        url="https://api.deepseek.com/chat/completions",
        default_model="deepseek-chat",
        dialect=OPENAI,
    ),
    Provider(
        name="openrouter",
        env_keys=("OPENROUTER_API_KEY",),
        url="https://openrouter.ai/api/v1/chat/completions",
        default_model="openai/gpt-4o-mini",
        dialect=OPENAI,
        key_prefixes=("sk-or-",),
    ),
    Provider(
        name="xai",
        env_keys=("XAI_API_KEY",),
        url="https://api.x.ai/v1/chat/completions",
        default_model="grok-2-latest",
        dialect=OPENAI,
        key_prefixes=("xai-",),
    ),
    # The escape hatch, and the reason `_check_endpoint` exists. Anything
    # OpenAI-compatible: a self-hosted vLLM, Ollama on loopback, Together,
    # Fireworks, an internal gateway. The endpoint is mandatory because there
    # is no sensible default, and it is checked because "point this module at
    # a URL" is otherwise an exfiltration setting.
    Provider(
        name="custom",
        env_keys=("LLM_API_KEY",),
        url="",
        default_model="",
        dialect=OPENAI,
        needs_base_url=True,
    ),
)

# Every recognised variable name, in resolution order. Used by the refusal
# message, by `available()`, and by the test suite to unset all of them.
ENV_KEYS: tuple[str, ...] = tuple(
    name for provider in PROVIDERS for name in provider.env_keys
)
# There is deliberately no `ENV_KEY` or `API_URL` singular alias here. Two
# reasons, both learned the hard way. A module-level constant holding "the" key
# variable is the exact shape a real credential once got pasted over in this
# file, and one name is easier to overwrite than a validated registry. And an
# alias would be a lie: there is no canonical variable any more, so code that
# reads one would be right for Anthropic and quietly wrong for the other eight.
# Callers that want a name want `ENV_KEYS` or `credentials_hint()`.
PROVIDERS_BY_NAME: Mapping[str, Provider] = {p.name: p for p in PROVIDERS}

# Where a custom endpoint may come from, config taking precedence.
BASE_URL_ENV = "LLM_BASE_URL"


class NarratorError(RuntimeError):
    """Base class for every way narration can refuse."""


class MissingCredentials(NarratorError):
    """No API key, for any provider. Deliberately fatal — see the docstring."""


class ProviderNotSupported(NarratorError):
    """`llm.provider` names something this module cannot speak to."""


class EndpointNotPermitted(NarratorError):
    """A configured base URL is not one this module is willing to call."""


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


def _validate_registry() -> None:
    """Refuse to import if the registry has been corrupted.

    Import-time rather than test-time. A test proves the property holds in CI;
    this makes a corrupted constant unable to run *at all*, which is what the
    incident in SECURITY.md needed — the pasted key was live on a developer's
    machine long before any suite saw it.

    Crashing on import is safe here specifically because the money path never
    imports this module: `python -m src.agent run` makes and executes every
    decision without it, so a broken narrator cannot stop a sweep. That
    property is asserted in test_narrator.py.
    """
    seen: set[str] = set()
    for provider in PROVIDERS:
        if not provider.env_keys:
            raise ValueError(f"provider {provider.name!r} names no environment variable")
        for name in provider.env_keys:
            if not _ENV_NAME.fullmatch(name):
                # Deliberately does not echo the offending value: if this fires
                # because a credential was pasted over a constant, repeating it
                # in the traceback would be the original bug again.
                raise ValueError(
                    f"provider {provider.name!r} has an environment variable name "
                    f"that is not a legal name (expected uppercase letters, digits "
                    f"and underscores). A credential may have been pasted over the "
                    f"constant; see SECURITY.md."
                )
            if name in seen:
                raise ValueError(f"{name} is claimed by two providers")
            seen.add(name)
        if provider.dialect not in _DIALECTS:
            raise ValueError(
                f"provider {provider.name!r} speaks unknown dialect "
                f"{provider.dialect!r}; known dialects are {sorted(_DIALECTS)}"
            )
        if provider.needs_base_url:
            if provider.url:
                raise ValueError(
                    f"provider {provider.name!r} both requires a base URL and "
                    f"ships one")
        else:
            _check_endpoint(provider.url.replace("{model}", "m"), provider.name)


def _check_endpoint(url: str, who: str) -> None:
    """Refuse a URL this module should not be POSTing a fact sheet to.

    The fact sheet carries relationship context — segment, tenure, contact
    history, amounts. It contains no name, address or contact detail, which is
    what makes sending it to a third party acceptable at all, but "acceptable
    to a chosen vendor" is not "acceptable to any host a config file names".
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError as exc:
        raise EndpointNotPermitted(f"{who}: {url!r} is not a URL ({exc})") from exc
    if parts.scheme == "http":
        if (parts.hostname or "") not in LOOPBACK_HOSTS:
            raise EndpointNotPermitted(
                f"{who}: plain http is permitted only to {sorted(LOOPBACK_HOSTS)} "
                f"(a local model server), not to {parts.hostname!r}. Use https."
            )
    elif parts.scheme != "https":
        raise EndpointNotPermitted(
            f"{who}: scheme {parts.scheme!r} is not permitted; use https")
    if not parts.hostname:
        raise EndpointNotPermitted(f"{who}: no host in the endpoint URL")
    if parts.username or parts.password:
        raise EndpointNotPermitted(
            f"{who}: the endpoint URL carries credentials in its userinfo. "
            f"Keys belong in the environment and are sent as a header."
        )
    if parts.query:
        raise EndpointNotPermitted(
            f"{who}: the endpoint URL has a query string. Keys and parameters "
            f"in a URL end up in every log that records the URL, so this module "
            f"sends them as headers and a request body only."
        )


@dataclass(frozen=True)
class Endpoint:
    """A provider resolved down to exactly what one request needs."""

    provider: Provider
    url: str
    model: str

    def headers(self, api_key: str) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        headers.update(dict(self.provider.extra_headers))
        headers[self.provider.auth_header] = self.provider.auth_prefix + api_key
        return headers


def provider_for_key(key: str) -> Optional[Provider]:
    """Identify a provider from the shape of an explicitly supplied key.

    Only consulted when `llm.provider` is `auto` and a key was passed in code
    rather than found in the environment — the case a caller writing
    `Narrator(api_key=...)` is in. Longest prefix wins so `sk-ant-` resolves
    to Anthropic rather than to OpenAI's broader `sk-`.

    Returns None rather than guessing when nothing matches. A wrong guess
    would send a fact sheet to the wrong vendor, so silence is correct.
    """
    candidates: list[tuple[int, Provider]] = []
    for provider in PROVIDERS:
        for prefix in provider.key_prefixes:
            if key.startswith(prefix):
                candidates.append((len(prefix), provider))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: -pair[0])
    return candidates[0][1]


def _configured_base_url(cfg: Mapping[str, Any], env: Mapping[str, str]) -> str:
    llm_cfg = cfg.get("llm", {}) or {}
    return str(llm_cfg.get("base_url") or env.get(BASE_URL_ENV, "") or "").strip()


def credentials_hint() -> str:
    """A safe, name-only description of where a key is looked for.

    Every name in this string has passed `_ENV_NAME` at import, so this cannot
    become the disclosure path the refusal message once was.
    """
    return "set one of: " + ", ".join(ENV_KEYS)


def resolve(cfg: Optional[Mapping[str, Any]] = None,
            api_key: Optional[str] = None,
            env: Optional[Mapping[str, str]] = None) -> tuple[Endpoint, str]:
    """Work out which provider to use and with which key, or refuse.

    Returns `(endpoint, key)`. Raises `MissingCredentials` when no key can be
    found, `ProviderNotSupported` when config names an unknown provider, and
    `EndpointNotPermitted` when a custom base URL is not one we will call.

    Pure with respect to its arguments — `env` is injectable — so the whole
    resolution order is testable without touching `os.environ`.
    """
    cfg = cfg if cfg is not None else C.load_config()
    env = env if env is not None else os.environ
    llm_cfg = cfg.get("llm", {}) or {}
    wanted = str(llm_cfg.get("provider", "auto") or "auto").strip().lower()

    def key_from_env(provider: Provider) -> str:
        for name in provider.env_keys:
            value = (env.get(name) or "").strip()
            if value:
                return value
        return ""

    supplied = (api_key or "").strip()

    if wanted not in ("auto", ""):
        provider = PROVIDERS_BY_NAME.get(wanted)
        if provider is None:
            raise ProviderNotSupported(
                f"llm.provider is {wanted!r}, which this build does not know. "
                f"Known providers are {sorted(PROVIDERS_BY_NAME)}. Anything with "
                f"an OpenAI-compatible endpoint can use provider 'custom' with "
                f"{BASE_URL_ENV} or llm.base_url."
            )
        key = supplied or key_from_env(provider)
        if not key:
            # Named explicitly and absent: refuse rather than falling through to
            # another vendor. Quietly using a different provider than the one
            # configured is the kind of surprise this project exists to avoid.
            raise MissingCredentials(
                f"llm.provider is pinned to {provider.name!r} but "
                f"{' / '.join(provider.env_keys)} is not set. Narration requires a "
                f"real API key by design — there is no template fallback, because a "
                f"fallback would let a misconfigured deployment produce plausible "
                f"text that nobody generated. Set "
                f"{provider.env_keys[0]}, change llm.provider, or do not call the "
                f"narration endpoints; the recovery pipeline itself does not need "
                f"them and will run without a key."
            )
        return _endpoint_for(provider, cfg, env), key

    # auto: an explicitly supplied key identifies itself; otherwise take the
    # first provider whose variable is set, in registry order.
    if supplied:
        provider = provider_for_key(supplied)
        if provider is None:
            raise ProviderNotSupported(
                "a key was supplied but its provider could not be identified from "
                "its prefix, and llm.provider is 'auto'. Set llm.provider to one "
                f"of {sorted(PROVIDERS_BY_NAME)} so the destination is explicit "
                "rather than guessed."
            )
        return _endpoint_for(provider, cfg, env), supplied

    for provider in PROVIDERS:
        key = key_from_env(provider)
        if key:
            return _endpoint_for(provider, cfg, env), key

    raise MissingCredentials(
        f"no LLM API key is set. Narration requires a real API key by design — "
        f"there is no template fallback, because a fallback would let a "
        f"misconfigured deployment produce plausible text that nobody generated. "
        f"{credentials_hint()}; or do not call the narration endpoints, because "
        f"the recovery pipeline itself does not need them and will run without a "
        f"key."
    )


def _endpoint_for(provider: Provider, cfg: Mapping[str, Any],
                  env: Mapping[str, str]) -> Endpoint:
    llm_cfg = cfg.get("llm", {}) or {}
    model = str(llm_cfg.get("model") or provider.default_model or "").strip()
    base_url = _configured_base_url(cfg, env)

    if provider.needs_base_url:
        if not base_url:
            raise EndpointNotPermitted(
                f"provider {provider.name!r} has no endpoint. Set {BASE_URL_ENV} "
                f"or llm.base_url to the full OpenAI-compatible chat-completions "
                f"URL, for example "
                f"https://your-gateway.example.com/v1/chat/completions."
            )
        url = base_url
    else:
        # An operator may still redirect a known provider — a regional endpoint
        # or a caching proxy is a legitimate reason — and the same check applies.
        url = base_url or provider.url

    if not model:
        raise EndpointNotPermitted(
            f"provider {provider.name!r} has no model. Set llm.model in "
            f"config/policy.yaml."
        )
    if provider.model_in_url:
        # Quoted because it lands in a path segment. Nothing in this project
        # puts anything but a configured constant here, and a model name with a
        # slash in it (openrouter-style) must not be able to climb the path.
        url = url.replace("{model}", urllib.parse.quote(model, safe=""))
    _check_endpoint(url, provider.name)
    return Endpoint(provider=provider, url=url, model=model)


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
# Dialects: request bodies
# ---------------------------------------------------------------------
#
# `messages` arriving here is always this module's own list of
# `{"role": "user"|"assistant", "content": str}` dicts — the neutral form the
# narrator builds. Each builder translates that into one vendor's shape.
#
# Not one of them emits a `tools`, `tool_choice`, `functions` or
# `function_call` key, and `test_narrator.py` asserts that for every provider.
# The absence is what makes generated text inert: with no tools in the payload
# there is no function-calling loop, so there is no path from a model's output
# to a side effect. That is a stronger claim than "we ignore tool calls", and
# it is the reason a new provider is a data change rather than a code change —
# there is nowhere for a per-vendor capability to be switched on.

def _payload_anthropic(endpoint: Endpoint, max_tokens: int, system: str,
                       messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "model": endpoint.model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
    }


def _payload_openai(endpoint: Endpoint, max_tokens: int, system: str,
                    messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Chat-completions. The system prompt is the first message, not a field."""
    return {
        "model": endpoint.model,
        endpoint.provider.max_tokens_field: max_tokens,
        "messages": ([{"role": "system", "content": system}]
                     + [{"role": m["role"], "content": m["content"]} for m in messages]),
    }


def _payload_gemini(endpoint: Endpoint, max_tokens: int, system: str,
                    messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """generateContent. Turns are `contents`, and the assistant is "model"."""
    return {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [
            {"role": "model" if m["role"] == "assistant" else "user",
             "parts": [{"text": m["content"]}]}
            for m in messages
        ],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }


# ---------------------------------------------------------------------
# Dialects: pulling the text back out
# ---------------------------------------------------------------------
#
# Every extractor raises `DraftRejected` rather than returning "" when there is
# no text. An empty draft would be caught by validation a moment later, but as
# "the model returned nothing" — which sends whoever is debugging to the prompt
# instead of to the transport. A malformed response body and an empty
# completion need different fixes, so they get different messages.

def _no_text(what: str) -> "DraftRejected":
    return DraftRejected([f"the API returned no text content ({what})"], "")


def _text_anthropic(response: Mapping[str, Any]) -> str:
    blocks = response.get("content") or []
    parts = [b.get("text", "") for b in blocks
             if isinstance(b, Mapping) and b.get("type") == "text"]
    text = "\n".join(p for p in parts if p).strip()
    if not text:
        raise _no_text("no text block in content[]")
    return text


def _text_openai(response: Mapping[str, Any]) -> str:
    choices = response.get("choices") or []
    parts: list[str] = []
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                # Some compatible servers return the multimodal block form.
                parts += [b.get("text", "") for b in content
                          if isinstance(b, Mapping) and isinstance(b.get("text"), str)]
    text = "\n".join(p for p in parts if p).strip()
    if not text:
        raise _no_text("no choices[].message.content")
    return text


def _text_gemini(response: Mapping[str, Any]) -> str:
    candidates = response.get("candidates") or []
    parts: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        content = candidate.get("content")
        if isinstance(content, Mapping):
            for block in content.get("parts") or []:
                if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
    text = "\n".join(p for p in parts if p).strip()
    if not text:
        # A Gemini response with no parts is usually a safety block, and saying
        # so is the difference between an operator checking their prompt and an
        # operator checking their network.
        blocked = ""
        for candidate in candidates:
            if isinstance(candidate, Mapping) and candidate.get("finishReason"):
                blocked = f", finishReason={candidate['finishReason']}"
                break
        raise _no_text(f"no candidates[].content.parts[].text{blocked}")
    return text


_DIALECTS: Mapping[str, tuple[Callable[..., dict[str, Any]],
                              Callable[[Mapping[str, Any]], str]]] = {
    ANTHROPIC: (_payload_anthropic, _text_anthropic),
    OPENAI: (_payload_openai, _text_openai),
    GEMINI: (_payload_gemini, _text_gemini),
}


def build_payload(endpoint: Endpoint, max_tokens: int, system: str,
                  messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return _DIALECTS[endpoint.provider.dialect][0](
        endpoint, max_tokens, system, messages)


def extract_text(endpoint: Endpoint, response: Mapping[str, Any]) -> str:
    return _DIALECTS[endpoint.provider.dialect][1](response)


# Validated last, because it calls `_check_endpoint` and `_DIALECTS`, both of
# which have to exist first. Import fails loudly if the registry is corrupt.
_validate_registry()


# ---------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------

def _transport(payload: Mapping[str, Any], api_key: str, timeout: float,
               endpoint: Endpoint) -> dict[str, Any]:
    """POST to a provider with the standard library. Replaceable in tests.

    Deliberately minimal: one request, no streaming, no tools, no retries
    (the caller owns those so the backoff is visible), and the response is
    parsed as JSON with no evaluation of any kind.

    There is exactly one `urlopen` in this module and every provider goes
    through it, so "what can this module talk to" is answered by
    `_check_endpoint` alone rather than by auditing one call site per vendor.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint.url, data=body, method="POST",
        headers=endpoint.headers(api_key),
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


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
    """Every check that applies to generated text, in one place.

    Provider-independent on purpose. Whatever produced the text, it passes the
    same gate — so choosing a cheaper or a local model cannot also choose a
    weaker validator.
    """
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
    # Which vendor produced this. Recorded rather than inferred, because with
    # several providers configurable the environment at read time is not
    # evidence of the environment at write time.
    provider: str = ""
    validated: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "provider": self.provider,
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
        self.max_tokens = int(llm_cfg.get("max_tokens", 700))
        configured = tuple(llm_cfg.get("permitted_roles", ()))
        unknown = [r for r in configured if r not in IMPLEMENTED_ROLES]
        if unknown:
            raise RoleNotPermitted(
                f"config permits roles this module cannot build a prompt for: "
                f"{unknown}. Implemented roles are {list(IMPLEMENTED_ROLES)}."
            )
        self.permitted_roles = frozenset(configured)

        # Resolution raises if there is no key, if config names a provider this
        # build cannot speak to, or if a custom endpoint is not one we will
        # call. All three are startup failures rather than first-request
        # failures, for the same reason the role check above is.
        self.endpoint, key = resolve(self.cfg, api_key)
        self._api_key = key
        self.provider = self.endpoint.provider.name
        self.model = self.endpoint.model

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
            # The neutral turn list. Each dialect translates it; nothing here
            # knows which vendor is on the other end.
            messages: list[dict[str, str]] = [{"role": "user", "content": user}]
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
            response = self._call(build_payload(
                self.endpoint, self.max_tokens, system, messages))
            text = extract_text(self.endpoint, response)
            problems = validate_draft(text, role, fact_sheet, self.cfg)
            if not problems:
                return Draft(
                    role=role, text=text, model=self.model,
                    provider=self.provider, fact_sheet=fact_sheet,
                    attempts=attempt,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
        raise DraftRejected(problems, text)

    # -- transport with visible backoff -------------------------------

    def _call(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        # The transport takes the endpoint explicitly rather than reading it off
        # the instance, so a test double sees exactly what the real transport
        # sees. An earlier revision sniffed the callable's signature and passed
        # three arguments to older doubles; that made the shipped four-argument
        # path the one path no test exercised, which is the opposite of what a
        # seam is for. One signature, no branch.
        last: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                return self.transport(payload, self._api_key, self.timeout,
                                      self.endpoint)
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
                        f"{self.provider} rejected the request with HTTP "
                        f"{exc.code}. {detail}"
                    ) from exc
                last = exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
            if attempt < 3:
                time.sleep(0.75 * attempt)
        raise TransportFailed(
            f"could not reach {self.provider} after 3 attempts: {last}"
        ) from last


def available(cfg: Optional[Mapping[str, Any]] = None) -> bool:
    """Whether narration could run, without constructing anything.

    Used by the dashboard to grey out the narration controls instead of
    offering a button that will return a 503. Resolution is pure and does no
    I/O, so this stays as cheap as the environment lookup it replaced while
    also honouring a pinned `llm.provider` — a build pinned to OpenAI with only
    an Anthropic key present is correctly reported unavailable.
    """
    try:
        resolve(cfg if cfg is not None else C.load_config())
    except NarratorError:
        return False
    return True


def configured_provider(cfg: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """The provider that would be used right now, or None. For the dashboard."""
    try:
        endpoint, _ = resolve(cfg if cfg is not None else C.load_config())
    except NarratorError:
        return None
    return endpoint.provider.name
