"""Task 8 (October 2026): durable multi-turn trajectory and recovery records.

Owned by the Task 8 lane. Imported at the bottom of app/models.py so the tables
register with SQLAlchemy without two lanes editing models.py together.
"""
from __future__ import annotations

import json

from .extensions import db
from .models import _utcnow

__all__ = ["Trajectory", "TrajectoryTurn", "TrajectoryCall", "TrajectoryArchive"]


def _loads(text, default):
    try:
        value = json.loads(text) if text else default
    except (TypeError, ValueError):
        return default
    return value if isinstance(value, type(default)) else default


class Trajectory(db.Model):
    """One run segment of one multi-turn objective attempt.

    Keyed like the result rows: (benchmark, attack, objective, best-of-N
    attempt, attacker roster index). A resume never rewrites a previous
    segment: it starts a new Trajectory that names the turn it resumed from,
    and the previous one becomes ``archived``. ``snapshot_json`` is the latest
    full conversation read back from PyRIT memory after each target reply.
    """
    __tablename__ = "trajectories"
    id = db.Column(db.Integer, primary_key=True)
    benchmark_id = db.Column(db.Integer, db.ForeignKey("benchmarks.id"), nullable=False, index=True)
    attack_slug = db.Column(db.String(80), nullable=False)
    objective_id = db.Column(db.String(40), nullable=False)
    attempt = db.Column(db.Integer, nullable=False, default=0)        # best-of-N pass
    roster_index = db.Column(db.Integer, nullable=False, default=0)   # Prompt.attempt
    objective_index = db.Column(db.Integer)                           # position in the run's objective list
    converter_used = db.Column(db.String(60))                         # roster label, as Prompt.converter_used
    orchestrator = db.Column(db.String(40))
    # running | interrupted | completed | resumed | archived | failed
    status = db.Column(db.String(16), nullable=False, default="running")
    pyrit_conversation_id = db.Column(db.String(64))
    snapshot_json = db.Column(db.Text, nullable=False, default="[]")
    turns_recorded = db.Column(db.Integer, nullable=False, default=0)  # complete pairs in the snapshot
    max_turns = db.Column(db.Integer)
    started_at = db.Column(db.DateTime, default=_utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=_utcnow, nullable=False)
    finished_at = db.Column(db.DateTime)
    resumed_from_turn = db.Column(db.Integer)
    # The trajectory this segment continues (a resume) or replaces (a restart).
    previous_trajectory_id = db.Column(db.Integer, db.ForeignKey("trajectories.id"))
    note = db.Column(db.Text)

    benchmark = db.relationship("Benchmark", backref=db.backref(
        "trajectories", cascade="all, delete-orphan", order_by="Trajectory.id"))
    turns = db.relationship("TrajectoryTurn", backref="trajectory", cascade="all, delete-orphan",
                            order_by="TrajectoryTurn.id",
                            foreign_keys="TrajectoryTurn.trajectory_id")

    @property
    def snapshot(self) -> list:
        return _loads(self.snapshot_json, [])

    def as_dict(self, with_turns: bool = True) -> dict:
        out = {c.name: getattr(self, c.name) for c in self.__table__.columns if c.name != "snapshot_json"}
        out = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in out.items()}
        out["snapshot"] = self.snapshot
        if with_turns:
            out["turns"] = [t.as_dict() for t in self.turns]
        return out


class TrajectoryTurn(db.Model):
    """One send to the multi-turn target, written before the call and completed after it."""
    __tablename__ = "trajectory_turns"
    id = db.Column(db.Integer, primary_key=True)
    trajectory_id = db.Column(db.Integer, db.ForeignKey("trajectories.id"), nullable=False, index=True)
    turn_index = db.Column(db.Integer, nullable=False)  # send order; a resumed segment starts at its first new turn
    pyrit_conversation_id = db.Column(db.String(64))
    prompt_text = db.Column(db.Text, nullable=False, default="")
    response_text = db.Column(db.Text)
    status = db.Column(db.String(16), nullable=False, default="started")  # started | ok | error | interrupted
    error = db.Column(db.Text)
    latency_ms = db.Column(db.Integer)
    started_at = db.Column(db.DateTime, default=_utcnow, nullable=False)
    finished_at = db.Column(db.DateTime)
    request_call_id = db.Column(db.String(36), index=True)

    def as_dict(self) -> dict:
        return {c.name: (v.isoformat() if hasattr(v, "isoformat") else v)
                for c in self.__table__.columns for v in [getattr(self, c.name)]}


class TrajectoryCall(db.Model):
    """Links a request-ledger call (RequestAttempt.call_id) to its trajectory.

    RequestAttempt belongs to the shared models module, so the link lives here.
    Covers target_send, attacker_call and judge_internal invocations.
    """
    __tablename__ = "trajectory_calls"
    id = db.Column(db.Integer, primary_key=True)
    trajectory_id = db.Column(db.Integer, db.ForeignKey("trajectories.id"), nullable=False, index=True)
    call_id = db.Column(db.String(36), nullable=False, index=True)
    stage = db.Column(db.String(24), nullable=False)
    turn_index = db.Column(db.Integer)

    trajectory = db.relationship("Trajectory", backref=db.backref(
        "calls", cascade="all, delete-orphan", order_by="TrajectoryCall.id"))


class TrajectoryArchive(db.Model):
    """Rows and the trajectory they came from, kept when a resume, rerun or restart replaces them.

    Archived rows never enter current results. ``conversation_retry_id`` links
    the ConversationRetry record that the results page already shows.
    """
    __tablename__ = "trajectory_archives"
    id = db.Column(db.Integer, primary_key=True)
    benchmark_id = db.Column(db.Integer, db.ForeignKey("benchmarks.id"), nullable=False, index=True)
    attack_slug = db.Column(db.String(80), nullable=False)
    objective_id = db.Column(db.String(40))
    attempt = db.Column(db.Integer)
    roster_index = db.Column(db.Integer)
    reason = db.Column(db.String(16), nullable=False)  # resume | rerun | restart | superseded
    original_rows_json = db.Column(db.Text, nullable=False, default="[]")
    trajectory_json = db.Column(db.Text)
    conversation_retry_id = db.Column(db.Integer, db.ForeignKey("conversation_retries.id"))
    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)

    benchmark = db.relationship("Benchmark", backref=db.backref(
        "trajectory_archives", cascade="all, delete-orphan", order_by="TrajectoryArchive.id.desc()"))

    @property
    def original_rows(self) -> list:
        return _loads(self.original_rows_json, [])

    @property
    def trajectory(self) -> dict:
        return _loads(self.trajectory_json, {})
