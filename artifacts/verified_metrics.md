# Verified metrics

Every number in the README, the CHANGELOG and the model card must appear here
first, with the command that produced it. The point is that no figure reaches a
document by being remembered.

Produced on 2026-08-28 against code version 3.1.0, policy version 3.1.0, on the
held-out `test` split. Reproduce the whole set with:

```
python -m src.train                     # weights + artifacts/training_report.json
python -m src.agent run --split test    # the sweep, and the audit trail
python -m src.benchmark                 # artifacts/benchmark.json
python -m src.agent verify              # hash chain
```

## The sweep

`python -m src.agent run --split test` → run `run_20260828T042144Z_39af`,
1,844 events, 3,688 audit records (one decision and one execution each).

| quantity | value |
|---|---|
| gross at risk | 33,580,605.25 INR |
| value at risk (margin on sales, face value on receivables) | 29,955,373.81 INR |
| projected transactional recovery | 4,993,231.38 INR (16.7% of value at risk) |
| projected retained future value (LTV term, weight 0.15) | 5,541,141.85 INR |
| total projected | 10,534,373.24 INR |
| options refused by guardrails | 4,925 |
| decisions gated for a person | 396 (21%) |
| retries issued against the sweep ceiling | 500 / 500 |
| discount committed against budget | 141,557.11 of 671,612 INR |

Execution statuses: 1,010 simulated, 438 no_action, 396 awaiting_approval.

Per-surface projected transactional recovery: overdue_receivable 5,947,360.51
across 302 events, checkout_abandonment 2,508,720.54 across 568,
payment_failure 2,078,292.30 across 974.

The transactional figure is **not** to be quoted as 9,319,055.75 / 31.1%. That
was measured before the `score_human_review` pricing fix and was inflated by
it. See the CHANGELOG entry.

## The benchmark

`python -m src.benchmark` → `artifacts/benchmark.json`. Paired bootstrap, 2,000
replicates, seed 20260826, reference policy `agent`. Realised money, not
projected — the simulator's counterfactual outcomes decide who actually paid.

| policy | realised net INR | recovered | contacts | gated | lawful |
|---|---|---|---|---|---|
| oracle_per_event | 13,163,161 | 1,148 | 638 | 0 | no |
| best_fixed_action | 8,816,602 | 666 | 870 | 0 | no |
| discount_everyone | 8,644,659 | 632 | 1,844 | 0 | no |
| rules_true_cause | 8,187,523 | 778 | 1,017 | 0 | no |
| rules_predicted_cause | 7,865,190 | 743 | 973 | 0 | no |
| agent_no_guardrails | 5,647,901 | 742 | 1,032 | 0 | no |
| discount_everyone_compliant | 5,483,947 | 465 | 1,010 | 0 | yes |
| contact_everyone | 5,268,094 | 582 | 870 | 0 | no |
| retry_or_chase_everything | 5,079,953 | 578 | 568 | 0 | no |
| rules_true_cause_compliant | 4,990,823 | 583 | 604 | 0 | yes |
| rules_predicted_cause_compliant | 3,954,127 | 565 | 583 | 0 | yes |
| **agent** | **3,802,294** | **603** | **727** | **396** | **yes** |
| agent_transactional | 3,801,662 | 581 | 656 | 361 | yes |
| contact_everyone_compliant | 2,529,257 | 502 | 503 | 0 | yes |
| retry_or_chase_everything_compliant | 927,441 | 374 | 189 | 0 | yes |
| agent_unattended | 908,434 | 549 | 590 | 396 | yes |
| do_nothing | 0 | 230 | 0 | 0 | yes |

Uplift of `agent` over each lawful baseline, 95% percentile interval:

| baseline | uplift INR | 95% CI | verdict |
|---|---|---|---|
| do_nothing | 3,802,294 | 1,359,164 … 6,710,975 | significant |
| retry_or_chase_everything_compliant | 2,874,853 | 702,291 … 5,601,442 | significant |
| agent_unattended | 2,893,860 | 431,275 … 5,765,212 | significant |
| contact_everyone_compliant | 1,273,037 | −1,205,884 … 4,113,454 | not distinguishable |
| agent_transactional | 632 | −7,767 … 9,105 | not distinguishable |
| rules_predicted_cause_compliant | −151,833 | −3,393,561 … 3,037,519 | not distinguishable |
| rules_true_cause_compliant | −1,188,528 | −4,640,322 … 2,339,579 | not distinguishable |
| discount_everyone_compliant | −1,681,652 | −4,621,585 … 540,095 | not distinguishable |

**This must be reported as it stands.** Three lawful baselines post a higher
point estimate than the agent, and the honest reading is that on 1,844 events
the agent is not statistically separable from the better compliant baselines: it
significantly beats doing nothing, significantly beats compliant
retry-everything, significantly beats its own unattended variant, and is a wash
against the rest. What it adds over a wash is 396 decisions routed to a person,
4,925 refusals with a stated reason each, and a hash-linked record of every one
— none of which any baseline provides. Claiming a revenue win here would be
claiming something the interval does not support.

Against the policies that ignore the rules, four are significantly *worse* than
the agent (discount_everyone, rules_predicted_cause, rules_true_cause,
best_fixed_action) and `oracle_per_event` is significantly better, as it must be
— it is the ceiling, and it sees each event's counterfactual outcome.

Price of compliance: the same agent with the policy engine removed earns
1,845,607.22 INR more, 6.16% of value at risk, and buys that with 305 extra
contacts and 757 choices the engine refuses.

## Models

`artifacts/training_report.json`, held-out split, from `python -m src.train`.

Root cause, accuracy against the majority-class baseline:
payment_failure 81.3% vs 32.2% on 974 events, log loss 0.608, confidence
calibration error 0.074; checkout_abandonment 83.5% vs 31.0% on 568, log loss
0.471, error 0.062; overdue_receivable 86.8% vs 38.7% on 302, log loss 0.364,
error 0.020.

Recovery probability: payment_failure Brier 0.1466, calibration error 0.0295,
AUC 0.7421 on 974; checkout_abandonment Brier 0.2059, error 0.0657, AUC 0.7148
on 568; overdue_receivable Brier 0.2068, error 0.0693, AUC 0.7171 on 302.

An AUC near 0.72 is modest and is not to be dressed up. The money comes from
pricing each option honestly, not from separating winners from losers sharply.

## Properties checked against the generated audit trail

Run after a fresh sweep; all four hold on 1,844 events.

Review is never chosen while an automated option was permitted: 0 violations.
Review is never priced above the best permitted option, including when every
option loses money: 0 violations. Exactly one arithmetic note per review option:
0 violations. That note recomputes to the stored ENR within 1 INR: 0 violations.

The third and fourth exist because re-pricing review after screening used to
*append* its arithmetic note rather than replace the one written during ranking,
so 5 records carried two contradictory derivations of a single stored figure.
The figure was right both times. A record a reviewer has to adjudicate before
they can use it is not evidence, which is the whole reason the log exists.

Review priced above the best option *including options policy refused*: 5 cases,
and this is correct rather than a violation. In all 5 every automated option was
blocked, so there is no permitted upside for review to be a fraction of and it
prices at minus the analyst's time (−40 INR) while the refused options sit at
−375 to −1,821. The invariant that matters is the one about *permitted* options;
a blocked option's ENR is not a price the agent can pay.

## What the audit trail costs

Measured 2026-08-29, because the figure quoted in an earlier draft of the README
(~10 ms per decision) was remembered rather than measured, and was wrong by an
order of magnitude. Method: replay the first 600 real payloads from the shipped
trail into a throwaway `AuditStore`, stripping the four chain fields so they are
recomputed, and time each `append`.

| quantity | measured |
|---|---|
| append one record | 0.83 ms mean, 0.70 ms median, 1.70 ms p95 |
| per decision (a decision record and an execution record) | ~1.7 ms, 7,906 bytes |
| the shipped trail, 3,688 records | 15,584,813 bytes, 4,226 bytes/record |
| first `chain_head()` on that trail | 411 ms, then cached for the process |
| `verify_chain()` over all 3,688 records | 1,087 ms |

Append is O(1) in trail length: `chain_head()` scans the file once and caches
the digest, and every subsequent append reads it from memory. So the 411 ms is a
one-time cost per process, not a per-record one, and the 600-record measurement
does not understate what a longer trail costs. The right way to describe the
overhead is therefore that **evidence costs under 2 ms and about 8 KB per
decision** — cheap enough that there is no efficiency argument for recording
less than the whole `considered` list.

## Re-running a sweep on the same day

Not part of the demo log — the shipped audit trail holds one run — but measured,
because it is the frequency caps doing their job and worth stating.

An immediate second sweep over the same 1,844 events: 373 events explicitly
`skipped_duplicate`, actions taken fall from 1,010 to 48, `no_action` rises from
438 to 1,028. The 396 gated decisions are re-created, because a decision that
was never executed did not consume anyone's contact budget. So a re-run recovers
almost nothing and contacts almost nobody, which is the correct behaviour and
the opposite of what a retry-storm bug looks like.

## The HTTP layer

Loopback bind refused for 6 non-loopback addresses; no `--host` or `--bind`
flag exists. Nine GET routes return 200. Refusals, each with its own status:
unknown route 404, wrong method 405, oversized body 413, malformed JSON 400,
non-object body 400, non-integer limit 400, absurd limit clamped, three path
traversal probes not served, `/api/sweep` 404 (the endpoint was removed rather
than guarded), anonymous approval 400, `system`-signed approval 400, missing
`granted` 400, string `granted` 400, unknown decision id 404, double approval
409, narration with none of the ten recognised key variables set a 503 that
names every variable it looked in and carries the provider list in its detail
body.

A signed approval returns `{"granted": true, "status": "awaiting_approval",
"detail": "released by vishal — nothing is sent to the customer while a case is
queued"}`. The hash chain verifies after every write.

`explain_event` writes nothing: decision and run record counts are unchanged
across a call, and the response asserts it in `call_wrote_nothing`. This was
checked rather than assumed, because "opening the dashboard cannot cause a
decision to be made" is the central claim the read-model design makes.

Alongside that assertion about the call, the response carries
`recorded_decision_id` / `recorded_at` / `recorded_action` — the log's latest
decision about the *event*, or nulls. Verified both ways: for two events taken
straight from `/api/decisions` the returned id, action and timestamp match the
row exactly and `GET /api/decision/<that id>` resolves, and for an event no
sweep has covered all three are null. An earlier build returned a single
`recorded: false`, which read as "this event is not in the audit trail" while
meaning "this call did not add to it" — true either way, and the dashboard
displayed the wrong reading of it for all 1,844 events that *are* in the log.
