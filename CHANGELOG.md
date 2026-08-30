# Changelog

Every number quoted here is recorded in
[`artifacts/verified_metrics.md`](artifacts/verified_metrics.md) with the command
that produced it. Nothing is quoted from memory.

The bugs get as much space as the features. A changelog that lists only what was
added is a sales document; the interesting question about a system that decides
how to spend money is what was wrong with it and how that was found.

---

## 3.1.0 — provider-agnostic narration

### The narrator speaks to any vendor

`src/narrator.py` was written against one vendor's HTTP API. It now speaks to
nine — Anthropic, OpenAI, Gemini, Groq, Mistral, DeepSeek, OpenRouter, xAI, and
a `custom` provider for any OpenAI-compatible endpoint — across three request and
response dialects. `llm.provider: auto` in `config/policy.yaml` takes the first
provider whose key is present, in a documented order; naming one pins it.

A provider is a frozen `Provider` record: its environment variables, its URL, its
default model, its dialect, its auth header and the prefix its keys carry. Adding
one is a data change. That matters for a reason beyond convenience: there is no
per-vendor code path, so there is nowhere for a per-vendor capability to be
switched on. Not one of the three payload builders emits a `tools`,
`tool_choice`, `functions` or `function_call` key, and the absence is now
asserted per provider over every key at every depth of the request, because a
shallow check would pass a payload that hid `tools` one level down.

`validate_draft` is deliberately one gate for all nine. A test runs the same
fabricated figure past every provider, because the obvious objection to making
the vendor configurable is that choosing a cheaper or a local model might also
choose a weaker validator, and that objection deserves an assertion rather than
an argument. `Draft` gained a `provider` field for the same reason: with nine
vendors configurable, the environment when a record is read is no evidence of the
environment that wrote it, so provenance is recorded rather than inferred.

Two defensive properties were designed in rather than added afterwards.

`_validate_registry()` runs at import. Every environment variable name must match
`[A-Z][A-Z0-9_]{2,63}`, no provider may claim a name another provider claims,
every dialect must exist, and every shipped URL must pass the endpoint check. A
credential pasted over any of those names now stops the module loading, and the
refusal does not echo the value. This is the structural repair for the incident
below: one constant became ten names, and ten names cannot each be pinned to a
literal.

`_check_endpoint` bounds where a fact sheet can be sent. A configurable base URL
is what makes an OpenAI-compatible gateway usable and it is also, unchecked, an
exfiltration setting in the one module that assembles relationship context into a
request body. It permits `https`, or plain `http` only to `127.0.0.1`, `::1` or
`localhost` — the same three strings `serve()` uses, so there is one list to get
right rather than two to get differently wrong — and refuses credentials in the
userinfo position and any query string. Gemini will accept its key as `?key=`;
this module sends `x-goog-api-key` instead, because a key in a URL is a key in
every log that records the URL.

There is still no template fallback, by explicit design and by the user's
explicit choice over the alternative. No key returns `503` listing every variable
it looked in, names only, and a pinned provider whose key is absent refuses
rather than falling through to whichever key happens to be set.

### A false claim in a comment, and what it was hiding

The refactor passed all 265 tests unchanged. That sounds reassuring and was the
opposite: the module's own comment stated that `test_narrator.py` asserted the
no-tools property "for every provider", and it did not. The suite was still
thirty-nine Anthropic-only tests, and it stayed green because of two pieces of
scaffolding that existed only to keep those tests working.

The first was a pair of `ENV_KEY` / `API_URL` aliases pointing at the first
provider in the registry. Both are gone. An alias like that is a lie once there
are ten variables — right for Anthropic and quietly wrong for the other eight —
and a module-level constant holding "the" key variable is the exact shape a real
credential once got pasted over in this file.

The second was worse. `Narrator._call` used `inspect.signature` to count the
transport's positional parameters and passed three arguments to older doubles and
four to the real one. So the four-argument path every real request takes was the
one path no test exercised, which is the opposite of what a test seam is for.
The shim is deleted, `FakeTransport` takes the shipped signature, and the double
now shapes its reply to the endpoint's dialect so the extractor under test is the
one that provider really uses.

`test_narrator.py` went from 39 tests to 70, and the suite from 265 to 296. The
new ones cover each dialect's payload shape and extractor, resolution order and
pinning, longest-prefix key identification (`sk-ant-` must beat `sk-`, or
Anthropic keys go to OpenAI), the endpoint whitelist and its refusals, and a
clean draft end to end on every provider in the registry.

Two of them started as failures and were kept as findings. A key in
`LLM_API_KEY` with no `llm.base_url` and no `llm.model` reports narration as
unavailable, not available — a half-configuration should grey the button out
rather than produce a `503` that looks like an outage. And `NarratorCase.setUp`
was clearing one environment variable, so on a machine holding a Gemini key the
no-credentials tests would have failed for a reason that had nothing to do with
the code. Both now clear every recognised variable.

### A real credential in the source tree

On 2026-08-29 a live `sk-ant-...` key was found in `src/narrator.py`, pasted over
the constant that is supposed to hold the *name* `"ANTHROPIC_API_KEY"`. Three
things were wrong at once: a live credential was in the source tree;
`os.environ.get(ENV_KEY)` then looked up a variable named after the secret, which
nothing sets, so narration refused every request and the symptom looked like an
unset key rather than a corrupted constant; and the `MissingCredentials` message
interpolates that constant, so the secret was printed to stdout and returned in
the body of a `503` to any caller of `POST /api/narrate`. A credential in source
is a leak. A credential on an error path is a leak with a delivery mechanism.

None of the 259 tests then in the suite caught it, because every one of them
asserted behaviour and this was a constant.

The key was never committed, and that was established rather than assumed. Every
object in the repository was read with `git cat-file --batch-all-objects`, which
covers unreachable objects unlike walking `git rev-list`, and every
credential-shaped match was scored by Shannon entropy. Every historical version
of `src/narrator.py` reads `ENV_KEY = "ANTHROPIC_API_KEY"`. One blob did match,
in ten commits, seven of them ancestors of `origin/main`, which is exactly the
shape of a genuine incident; the entropy is what settled it. The two hits inside
`tests/__pycache__/test_defensive_posture.cpython-310.pyc` score 0.08 and 0.00
bits per character over two and one distinct characters, because they are this
suite's own fragment-assembled fixtures compiled to bytecode. A real key runs
around 4.5. Both a naive grep and a naive dismissal would have got this wrong in
opposite directions.

**The key should still be revoked.** It was written to disk and printed on an
error path. "It never reached a commit" bounds the blast radius; it does not
clear the key.

The audit found two gaps on the way, and they are recorded here because they are
the mechanism by which this leak *would* have been published. `__pycache__` is
not in a `.gitignore` and thirty-seven compiled modules are tracked, including
`src/__pycache__/narrator.cpython-310.pyc` — committed bytecode is precisely
where a pasted constant lands. And the credential scan walks text extensions
only, so it claims no tracked file contains a credential while skipping every
binary. Neither is closed yet.

The repair so far: the constant, the stale bytecode, the import-time registry
guard described above, and five tests — the scan itself, a proof it fires on all
nine credential shapes, a proof it does not fire on the ten lookalikes this repo
legitimately contains, the whitelist over every name in the provider registry,
and a check that neither refusal message can carry a secret. The scan covers the
test files too, so the examples proving it fires are assembled from fragments at
runtime rather than written as literals. Excluding the test file was the
alternative and it is worse: the one place a pasted secret must not be able to
hide is the file whose job is finding pasted secrets.

---

## 3.0.0 — the rebuild

v2 predicted a root cause and looked up a fixed action for it. That is a
classifier with a dictionary attached, and it cannot answer the only question
that matters commercially: is this intervention worth doing at all. v3 answers
that question first and treats the classification as an input to it. All eight
items in the design brief are implemented.

**1. Optimise for money recovered, not classifier accuracy.** `src/economics.py`
scores every candidate action as expected net recovery, incremental against doing
nothing, so `do_nothing` is exactly 0.0 by construction and a negative score
means an action is worse than inaction rather than merely worse than the
alternatives. On the shipped sweep the agent recovers 4,993,231.38 INR
transactionally — 16.7% of 29,955,373.81 at risk — plus 5,541,141.85 in retained
relationship value, and it declines 4,925 options as not worth their cost.

**2. Prove incremental uplift against clear baselines.** `src/benchmark.py` scores
the agent and six baselines in realised rupees against counterfactual outcomes,
with a 2,000-replicate paired bootstrap. It does not flatter the agent, which is
why it is worth having: the agent nets 3,802,294 INR and beats doing nothing,
compliant retry-everything and its own unattended variant significantly, while
three lawful baselines post higher point estimates that the confidence intervals
will not separate from it. What the agent adds over that statistical wash is 396
decisions routed to a person, 4,925 refusals each with a stated reason, and a
hash-linked record of every one. The per-event oracle recovers 13,163,161, so
there is a large amount of headroom and the honest thing is to name it.

**3. A Recovery Command Centre dashboard.** A single-file HTML and JavaScript
dashboard served by `src/server.py`, no build step, thirteen routes of which
eleven are GET. FastAPI was specified in the brief and is not available in this
environment; a stdlib `http.server` is a substitution rather than a preference,
and it is recorded as one.

**4. Adaptive intervention selection.** Action choice falls out of the ENR
ranking rather than a lookup table, so the same root cause yields a different
action on a 500 INR cart and a 250,000 INR invoice.

**5. An LLM in a tightly bounded role.** `src/narrator.py` renders already-made
decisions into language and can do nothing else. See the 3.1.0 entry above for
what that boundary is now made of.

**6. Production readiness.** Guardrails as configuration with a stamped
`policy_version` on every record, dry-run by default requiring two independent
switches to leave, a kill-switch file, and a test suite that asserts absences
rather than defaults.

**7. The audit trail as decision evidence.** JSONL in append mode, each record
carrying the SHA-256 of its canonical JSON plus its predecessor's digest, so any
edit breaks every digest after it. Tamper evidence, not tamper proofing — anyone
who can write the file can rewrite the chain from the edit forward, and the
honest word is the one used. The shipped trail is `run_20260828T042144Z_39af`:
1,844 events, 3,688 records, head
`1bcf90fc725fb964c1fe8ce4efd988c7e6eee519addee27bdaa39845fa5981ec`. Cost is
0.83 ms mean per record, about 1.7 ms and 8 KB per decision.

**8. Lifetime value and margin awareness.** Recovery is valued on the right basis
per surface: a failed payment or abandoned cart earns the gross margin because
the cost of goods is still to come, while an overdue invoice is worth its full
face value because the goods are delivered and the cost is sunk. Getting this
wrong misprices two of the three surfaces by roughly 3x, and the practical effect
is that the agent spends far more effort per rupee on receivables than on carts.

### Bugs found on the way

**A falsy injected store, and 576 records in the demo trail.** An
`if store:`-shaped check treated a legitimately empty audit store as absent and
fell back to the default one, so the suite wrote 576 records into the shipped
demo trail while its own temporary trail stayed empty. It survived a full run
unnoticed. The trail was restored byte-exactly, and `helpers.AuditCase` now checks
the size of all three shipped files after every test and names the culprit if one
grew — a guard which is itself tested by feeding it a manufactured "before"
reading. Every suite run since is verified by md5 against the trail's recorded
digest.

**`_numbers_in` dropped trailing zeros.** The draft validator normalises numbers
before checking that each one traces back to the decision. It compared
`1,250.00` and `1250` as different figures, which made a correct draft look
fabricated. A validator that fails closed on a true statement teaches operators
to ignore it.

**An unvalidated `segment` reached the prompt.** The fact sheet is assembled field
by field from typed values specifically so no customer-supplied text can reach
the model, and one field was passing through unchecked. It is now validated
against the known segments, and a test asserts every label in the rendered sheet
is one this module wrote.

**`log_message` was not sanitising what it logged.** Fixed, and the egress
validator used by the messaging adapter is now asserted to be the same function
the narrator validates customer-facing drafts with, rather than a second copy of
similar rules.

**"~10 ms per decision" was wrong by an order of magnitude.** The measured figure
is 0.83 ms mean. It had been carried in prose from an early estimate, which is
the reason `artifacts/verified_metrics.md` exists at all: a number lands there
with the command that produced it before any document quotes it.

**`SECURITY.md` claimed "seventeen patterns" for a sixteen-item list**, and five
hints told users to run `python src/train.py`, which crashes — the module has to
be run as `python -m src.train`. Both are the same class of defect as the
comment in 3.1.0: prose that describes what the code was meant to do.

**An earlier recovery figure must not be quoted.** A pre-fix run reported
9,319,055.75 INR and 31.1% of value at risk. Those numbers are wrong and are
superseded by the figures in this entry.
