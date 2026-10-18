"""Persist local retry-helper invocations for static targets and saved judging.

No exception messages, prompts, credentials or URLs enter this ledger. A helper
invocation may contain SDK retries or browser work; it is not an HTTP counter.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

from ..extensions import db
from ..models import RequestAttempt, _utcnow
from .retry import RequestAccountingError

_observer = ContextVar("request_attempt_observer", default=None)


def _commit():
    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        raise RequestAccountingError("Local request history commit failed; execution stopped.") from exc


class RequestHistory:
    def __init__(self, benchmark_id, stage, *, prompt=None, evaluation=None,
                 attack_slug=None, sequence=None, attempt_slot=None):
        self.call_id = str(uuid4())
        self.stage = stage
        self.fields = dict(benchmark_id=benchmark_id, stage=stage, call_id=self.call_id,
                           prompt_id=prompt.id if prompt else None,
                           evaluation_id=evaluation.id if evaluation else None,
                           attack_slug=attack_slug or (prompt.attack_slug if prompt else None),
                           sequence=sequence if sequence is not None else (prompt.sequence if prompt else None),
                           attempt_slot=attempt_slot if attempt_slot is not None else (prompt.attempt if prompt else None))

    def begin(self, number, maximum):
        row = RequestAttempt(**self.fields, attempt_number=number, max_attempts=maximum)
        db.session.add(row)
        _commit()  # survive a process exit while awaiting the provider
        return row

    def finish(self, row, **fields):
        for key, value in fields.items():
            setattr(row, key, value)
        row.finished_at = _utcnow()
        _commit()

    def backoff(self, row, elapsed_ms):
        row.backoff_elapsed_ms = elapsed_ms
        _commit()

    def attach(self, prompt):
        RequestAttempt.query.filter_by(call_id=self.call_id).update({RequestAttempt.prompt_id: prompt.id})
        # Caller commits this with the captured response / archive replacement.


@contextmanager
def request_history(benchmark_id, stage, **kwargs):
    history = RequestHistory(benchmark_id, stage, **kwargs)
    token = _observer.set(history)
    try:
        yield history
    finally:
        _observer.reset(token)


def observer_for(stage):
    observer = _observer.get()
    return observer if observer and observer.stage == stage else None


def mark_interrupted_requests():
    for row in RequestAttempt.query.filter_by(status="started").all():
        row.status = "interrupted"
        row.reason = "process_interrupted"
        row.delivery = "unknown"
        # No invented request duration/end: the process exit time is unknown.
    db.session.commit()
