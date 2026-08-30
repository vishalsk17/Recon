# Security posture

This is a system that decides how to spend money on customers and, in a
deployed form, would send them messages. The design rule it was built under is
narrow and worth stating before anything else:

> **Dangerous capability should be absent, not defaulted off.**

A flag that ships safe still ships the capability, and every incident report
about a system like this contains the sentence "the flag was set in production".
So where a capability would be dangerous, the code does not contain it, and the
test suite asserts the absence rather than the default. That is a different
claim from "it is configured safely", and it is the claim this file makes.

What follows is organised by what the system *cannot* do, with the mechanism and
the test that holds each one down. `python -m unittest discover -s tests -t .`
runs all 296.

---

## It cannot send anything to a customer

No module under `src/adapters/` imports a network client. Not `requests`, not
`httpx`, not `urllib.request`, not `socket`, not `smtplib`, not a vendor SDK.
The adapters are the only code that would ever talk to a payment gateway or a
messaging provider, and in this build every one of them writes a simulated
result and returns.

This is enforced statically, because a static property is the only kind a
configuration mistake cannot reach. `TestAdaptersCannotReachTheNetwork` reads
each adapter's source, strips docstrings and comments (the modules discuss what
a live transport *would* import, and prose is not an import), and fails on any
import of fourteen transport libraries. A second test refuses
`__import__`, `importlib`, `eval`, `exec`, `compile`, `os.system` and
`subprocess` anywhere in that directory, because an import the static check
cannot see is worse than a visible one.

Turning execution live is deliberately a two-key operation — `dry_run: false`
in `config/policy.yaml` **and** `RECOVERY_AGENT_ALLOW_LIVE=1` in the
environment — and even with both, the live branch of every adapter raises
`LiveExecutionRefused`, because there is nothing implemented behind it. The
guard exists so the seam is visible to a reader; the emptiness behind it is what
makes the guarantee.

`GET /api/health` reports `live_transport_available: false`, and a test asserts
it, so this is checkable from outside the process.

## It cannot be reached from the network

`serve()` refuses any bind address outside a three-item whitelist —
`127.0.0.1`, `::1`, `localhost` — and there is **no `--host` or `--bind`
flag**. Not one that defaults to loopback: the flag does not exist, and
`argparse` rejects it.

There is no authentication in this build. A `--host` flag would therefore be a
flag that converts an unauthenticated money-moving control panel into a network
service, which is why the capability is absent instead of guarded. Anyone who
genuinely needs remote access should put an authenticating reverse proxy in
front, which is a deliberate act rather than a typo.

The whitelist is a set of exact strings, not a subnet interpretation, so
`127.0.0.2` and `"127.0.0.1 "` are both refused. The refusal is also *ordered*:
it happens before `ensure_dirs()`, before the dashboard assets are read, and
before the socket is created. `test_the_refusal_happens_before_anything_is_opened`
booby-traps all three of those steps and asserts the `ValueError` arrives
first — because a rejected call that has already bound the port has not
refused anything.

## It cannot be made to start a sweep over HTTP

Thirteen routes, eleven of them GET and reads. Two POSTs:

- `POST /api/approve` releases or declines a decision **that already exists**
  and was already gated. It cannot create one, and it accepts no action name,
  amount or channel — `test_the_endpoint_accepts_no_action_name` asserts the
  handler's source never reads those keys and that a body carrying them is
  refused.
- `POST /api/narrate` returns text for a person to read.

There is no route that makes the agent decide or act. An earlier build had a
sweep endpoint; it was **removed rather than guarded**, and `/api/sweep` now
returns 404. `test_the_service_layer_uses_the_agent_for_exactly_three_things`
pins the service layer to `agent.decide`, `agent.toolbelt` and
`agent.resolve_approval` by reading its source, so a fourth use fails the
suite. No `do_PUT`, `do_DELETE`, `do_PATCH`, `do_OPTIONS`, `do_TRACE` or
`do_CONNECT` exists — nothing here is edited or removed, and a route that could
delete from an append-only hashed log would defeat the point of hashing it.

Opening the dashboard cannot cause a decision to be made. `explain_event`
prices every option for an event and writes nothing; the response says so in
`call_wrote_nothing`, and `TestExplainingAnEventChangesNothing` verifies it by
comparing both the record count *and* the chain head across the call.

## It cannot serve a file that was not declared

Static assets come from a fixed two-entry dict resolved to one file, loaded once
at startup. Path traversal is not filtered, it is **unrepresentable**: there is
no user-controlled path to join onto a directory. `server.py` contains exactly
one bare `open(` call, inside `_load_assets`, asserted with a pattern that does
not count `webbrowser.open(`. Eleven traversal shapes are tested; all 404, and
the response body is checked for absence of the audit log path and `root:`.

## It cannot be made to allocate memory to a caller's specification

`Content-Length` is validated and compared against a 64 KB cap *before* the body
is read. A declared length of ten million with a two-byte body gets a 413 —
and the proof that the check happens first is simply that a reply arrives at
all. Non-numeric, negative and empty lengths get 400.

## It cannot leak internals through an error

Unhandled exceptions return exactly `"internal error; see the server log"`; the
traceback goes to the server's own output. `test_an_unexpected_failure_leaks_nothing_to_the_caller`
injects a route raising `RuntimeError("could not read /var/secrets/prod.key")`
and asserts the client sees none of it.

Every response carries `Cache-Control: no-store` (these bodies contain decision
records and customer identifiers), `X-Content-Type-Options: nosniff`,
`Referrer-Policy: no-referrer`, and a Content-Security-Policy with
`default-src 'none'` and `connect-src 'self'`. The CSP is parsed and asserted
directive by directive, and checked for absence of `http` and `*`, so even if a
future edit pasted a CDN tag into the dashboard the browser would refuse it —
belt and braces on the no-external-assets property.

## It cannot hold cardholder data

No dataclass in the decision path names a cardholder field, checked against
sixteen patterns with word boundaries so `payment_method` survives and
`pan_hash` does not. The generated CSVs are checked too, because the claim a
reader cares about is about the data sitting on disk, not about the generator's
intentions.

`CustomerProfile` holds consent flags and no contact address. The split is the
point: the agent decides *whether* to contact someone and on *which* channel,
and resolving that to an actual address is the messaging adapter's job. A leaked
decision record cannot be used to contact anybody.

The audit writer screens every payload, at any nesting depth, against a
thirty-key forbidden list (`email`, `phone`, `card_number`, `cvv`, `api_key`,
`password`, `secret`, `authorization`, `address`, `upi_id`, `vpa`, `token`,
`iban`, `primary_account_number`, …) and raises rather than writing. A record
cannot become a personal-data spill by someone adding a convenient field
upstream.

## It cannot hold a credential

Nine credential formats — Anthropic, OpenAI-style, AWS, Razorpay, GitHub,
Google, Slack, PEM private keys, long bearer literals — are scanned for across
every text file in the repository, including the generated CSVs and the 15 MB
audit trail.

**This check exists because the property failed.** On 2026-08-29 a real
`sk-ant-...` key was found in `src/narrator.py`, pasted over the `ENV_KEY`
constant that is supposed to hold the *name* `"ANTHROPIC_API_KEY"`. Three things
were wrong at once:

1. A live credential was in the source tree.
2. `os.environ.get(ENV_KEY)` then looked up a variable named after the secret,
   which nothing sets — so narration refused every request, and the symptom
   looked like an unset key rather than a corrupted constant. A broken thing
   that reports the *expected* error is hard to notice.
3. The `MissingCredentials` message interpolates `ENV_KEY`, so the secret was
   printed to stdout and returned in the body of a `503` to any caller of
   `POST /api/narrate`. A credential in source is a leak; a credential on an
   error path is a leak with a delivery mechanism.

None of the 259 tests then in the suite caught it, because all of them asserted
behaviour and this was a constant. The repair was the constant, the stale
bytecode, and five new tests: the scan, a proof that the scan fires on all nine
shapes, a proof it does not fire on the ten lookalikes in this repo (a chain
digest, a run id, the `x-api-key` header name, the audit screener's field list,
the narrator tests' deliberately-fake key, and the `sk-`, `AIza` and `Bearer `
literals the provider registry has to contain in order to recognise a key by its
prefix), a whitelist over every variable name in that registry — which replaced
the original pin on one literal string — and a check that the refusal message
cannot carry a secret.

The pin became a whitelist because the refactor to nine providers turned one
constant into ten names, and ten names cannot each be pinned to a literal. The
whitelist is the stronger test rather than the looser one: it asks "is this a
legal environment variable name", which no credential in any format satisfies,
instead of "does this look like one of nine known secret shapes", which a tenth
shape would walk straight past.

`_validate_registry()` runs that whitelist **at import**, not in the test suite,
and this is the structural repair rather than the detection one. Every name must
match `[A-Z][A-Z0-9_]{2,63}`, no provider may claim a variable another provider
already claims, and every shipped URL must pass the endpoint check below. A key
pasted over any of those names now makes the module refuse to load at all, which
is what the original incident needed: that key sat on disk serving `503`s for
some time before any test ran. Crashing on import is safe here precisely because
nothing on the money path imports this module. The refusal deliberately does not
echo the offending value, and a test plants a credential-shaped name to prove the
import fails and that the message stays clean.

The scan covers the test files too, so the examples proving it fires are
assembled from fragments at runtime rather than written as literals. Excluding
the test file was the alternative and it is worse: a test file is source, and
the one place a pasted secret must not be able to hide is the file whose job is
finding pasted secrets.

**The key was never committed.** This was established rather than assumed, by
reading every object in the repository with `git cat-file --batch-all-objects`
(which covers unreachable objects, unlike walking `git rev-list`) and scoring
each credential-shaped match by Shannon entropy. Every historical version of
`src/narrator.py` reads `ENV_KEY = "ANTHROPIC_API_KEY"`. One blob did match, and
the entropy is what settled it: the two hits inside
`tests/__pycache__/test_defensive_posture.cpython-310.pyc` score 0.08 and 0.00
bits per character over two and one distinct characters, because they are this
suite's own fragment-assembled fixtures compiled to bytecode. A real key runs
around 4.5 bits per character. So the remote was never exposed.

**If you own that key, revoke it anyway.** It was written to disk and printed on
an error path, so it should be treated as compromised regardless of who is
believed to have seen it. "It never reached a commit" bounds the blast radius; it
does not clear the key.

Two gaps the audit found on the way, both recorded because they are the mechanism
by which this leak *would* have been published: `__pycache__` is not in a
`.gitignore` and compiled modules are tracked, and the credential scan walks text
extensions only, so it skips exactly the binary files a pasted constant gets
compiled into. Neither is closed yet.

## It cannot learn from the answer

The simulator writes oracle columns — `po_*` counterfactual outcomes,
`true_root_cause`, `logged_recovered`, `is_fraudulent` — so the benchmark can
score policies against counterfactual truth. Every one of them would make a
classifier look excellent and be worthless.

`assert_no_leakage` refuses them by prefix, and it is wired into every feature
list, every root-cause spec, every uplift action key, and the columns lifted
into `RiskEvent.features`. Two further tests guard the guard: one plants each
forbidden column and asserts it raises (if the check ever stopped raising, every
other leakage test would pass while asserting nothing), and one reads the
generated CSV headers and fails if any column is neither a declared feature, a
timestamp, nor covered by the prefix list — so adding a new oracle column to the
simulator cannot silently make it eligible as a feature.

## The language model cannot change a decision

`src/narrator.py` is the only place a model speaks, and it renders
already-decided facts into language. It receives a fact sheet built from the
`Decision` object — never the raw event, never the customer row — and it has
**no tools**. It cannot select, price, approve, or re-time an action, because
nothing it returns is read as anything but text.

That absence now has to hold nine times. The module speaks to nine providers
across three request dialects — Anthropic Messages, OpenAI chat-completions and
Gemini `generateContent` — and not one of the three payload builders emits a
`tools`, `tool_choice`, `functions` or `function_call` key. With no tools in the
request there is no function-calling loop, so there is no path from a model's
output to a side effect; that is a stronger claim than "we ignore tool calls",
and it is what makes adding a provider a data change rather than a code change,
because there is nowhere for a per-vendor capability to be switched on. The
claim is asserted per provider, over every key at every depth of the payload —
a shallow check would pass a request that hid `tools` one level down, which is
where a vendor-specific format would put it.

Three roles are permitted, listed in both `config/policy.yaml` and a tuple in
the module; a role in config that the code cannot build a prompt for is a
startup error, not a silent no-op. Output is validated before return: every
number in the draft must trace back to a figure in the decision, ten forbidden
patterns (`legal action`, `police`, `credit score`, `final warning`, …) are
refused, and length is capped. A draft that fails is retried once with the
problems fed back, and then **raises**.

`validate_draft` is deliberately one gate for all nine providers, and a test
runs the same fabricated figure past every one of them. This is the answer to
the obvious objection to making the vendor configurable: choosing a cheaper or a
local model must not also choose a weaker validator. Which vendor was used is
recorded on the `Draft` rather than inferred, because with nine providers
configurable the environment at read time is no evidence of the environment at
write time.

There is no template fallback, by explicit design. A silent fallback is the
worst of both worlds: the operator believes they are reading model output and
they are not, and a misconfigured key produces plausible text instead of a stack
trace. So no key at all returns `503` listing every variable it looked in —
names only, never a value — with the provider list in the detail body, and the
refusal carries no `text`, `draft` or `body` key, because a refusal must not
contain anything that looks like a draft. Pinning `llm.provider` to a vendor
whose key is absent refuses in the same way rather than falling through to
whichever key happens to be set; quietly using a different vendor than the one
configured is precisely the surprise this project exists to avoid.

`GET /api/health` reports `narration_provider`: the vendor name, or null. A name
only — never a key and never the endpoint, because a health endpoint is the
first thing anyone reads and the last place a secret should be.

The recovery pipeline does not import this module. `python -m src.agent run`
makes and executes every decision with no language model in the process, which
is the correct dependency direction for a system that moves money.

## It cannot POST a fact sheet to an arbitrary host

`llm.base_url` and `LLM_BASE_URL` exist so that any OpenAI-compatible gateway
can be used — vLLM, Ollama, a private proxy — and for a pinned vendor they can
also redirect to a regional endpoint. Both are legitimate. Both also make the
one module that assembles relationship context into a request body pointable at
a host of someone's choosing, which is to say a configurable base URL is an
exfiltration setting unless something bounds it.

`_check_endpoint` is that bound. It requires `https`, or plain `http` only to
`127.0.0.1`, `::1` or `localhost` — the same three-item loopback whitelist
`serve()` uses, as the same exact strings, so there is one list to get right
rather than two to get differently wrong. Credentials in the userinfo position
(`https://user:pass@host`) are refused, and so is any query string: Gemini will
accept its key as `?key=`, and this module sends `x-goog-api-key` instead,
because a key in a URL is a key in every access log, proxy log and error report
that records the URL. A model name that lands in a URL path is percent-encoded,
so a slashed name cannot climb out of its segment.

Every shipped provider URL passes the same function at import, so a typo'd
registry entry cannot ship. And there is exactly one `urlopen` in the module,
which every provider goes through — so "what can this talk to" is answered by
reading one function rather than by auditing a call site per vendor.

What this does not defend against is a hostile operator: anyone who can edit
`config/policy.yaml` can name any https host they like, and the fact sheet will
go there. The check bounds accidents and typos and narrows the blast radius of a
copied config; it is not a control over someone who already has write access to
the configuration.

## It cannot quietly rewrite its own history

The audit trail is JSONL opened in append mode, always. `AuditStore` has no
method that truncates, rewrites or deletes. Each record stores the SHA-256 of
its canonical JSON plus its predecessor's digest, so any edit to any record
breaks every digest after it, and `python -m src.agent verify` reports where.

Tamper *evidence*, not tamper proofing — the honest word. Anyone who can write
the file can rewrite the whole chain from the edit forward. Publishing the chain
head somewhere the agent cannot write is what would close that gap, and it is
not done here.

The shipped trail is `run_20260828T042144Z_39af`: 1,844 events, 3,688 records,
head `1bcf90fc725fb964c1fe8ce4efd988c7e6eee519addee27bdaa39845fa5981ec`.

No test may append to it. `helpers.AuditCase` checks the size of all three
shipped files after every test and names the culprit if one grew. That rule
exists because it was broken: a falsy-empty-store bug (see the CHANGELOG) had
the suite writing 576 records into the demo trail while its own temporary trail
stayed empty, and it stayed invisible for a full run. The guard is itself tested
by feeding it a manufactured "before" reading.

---

## What this build does not defend against

Stated plainly, because a security note that lists only strengths is marketing.

**There is no authentication or authorisation.** Anyone who can reach the port
can approve any gated decision. The `X-Operator` header records who *claimed*
responsibility; it does not verify it, and the code says so where it is read.
Loopback-only binding is the entire access control story. Wiring a real identity
layer means replacing one header read, which is the smallest seam this could
have — but it has not been wired.

**The approval record is a claim, not a signature.** Nothing is cryptographically
attributable to a person. What is enforced is that a claim must be made: an
unsigned approval is refused, and so are `system`, `automation`, `none`, `null`
and `-`, with a message about responsibility.

**The chain head is stored beside the chain.** See above.

**`'unsafe-inline'` is in the CSP.** The dashboard is deliberately one file with
inline style and script, which is the trade for having no build step and no
external assets. It is mitigated by the HTML never interpolating server data
into markup, but it is a real weakening of the policy and it is not hidden.

**The kill switch is a file on disk.** `HALT` at the project root stops all
execution, checked before every action. Anyone who can delete the file can
un-stop it.

**Rate limiting does not exist** at the HTTP layer. The caps that matter —
retries per payment, contacts per customer per week, retries per sweep, discount
budget — are enforced in the decision path, not the transport, so a caller
cannot spend money by making many requests. But they can make many requests.

**The data is synthetic and so is the fraud.** Every probability, cost and
outcome comes from `data/generate_*.py`. Calibration measured against a
simulator whose generative process the models are fitted against is optimistic
about the real world in a way no amount of held-out splitting fixes.

## Reporting

This is a buildathon project, not a deployed service. There is no disclosure
process. If you find something, the useful thing is a failing test.
