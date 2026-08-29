# Archived audit trail

These files were written before `Guardrails._reprice_review` landed. They
contain decisions where `request_human_review` outranked a permitted automated
action, because review was priced as a haircut on the best *unscreened* option
and then compared against the *screened* set. On the held-out split that
mispricing sent 48 events to a person while an automated action was available.

They are kept rather than deleted because the log is append-only by design and
silently discarding records would be the wrong habit to build, even for
synthetic demo data. They are moved aside rather than left in place because the
dashboard reads the live log, and a demo should show what the agent does now,
not what it did before a bug was fixed.

Regenerate the live trail with:

    python -m src.agent run
