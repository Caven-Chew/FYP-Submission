"""Task 7 (October 2026): versioned two-label assessment records.

Owned by the Task 7 lane. Imported at the bottom of app/models.py so the tables
register with SQLAlchemy without two lanes editing models.py together.
"""
from __future__ import annotations

import json

from .extensions import db
from .models import _utcnow

__all__ = ["ObjectiveRequirements", "ObjectiveAssessment", "ResponseAssessment", "HumanReview"]


def _loads(text, default):
    try:
        value = json.loads(text) if text else default
    except (TypeError, ValueError):
        return default
    return value if isinstance(value, type(default)) else default


class ObjectiveRequirements(db.Model):
    """An author-supplied essential-requirement list; editing adds a new version."""
    __tablename__ = "objective_requirements"
    id = db.Column(db.Integer, primary_key=True)
    objective_sha256 = db.Column(db.String(64), nullable=False, index=True)
    context_sha256 = db.Column(db.String(64), nullable=False)
    objective_text = db.Column(db.Text, nullable=False)
    requirements_json = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(24), nullable=False, default="author_supplied")
    version = db.Column(db.Integer, nullable=False)
    note = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)

    @property
    def requirements(self) -> list:
        return _loads(self.requirements_json, [])


class ObjectiveAssessment(db.Model):
    """Objective-stage attempt (no response shown to the judge).

    A structurally accepted result is reused for the same objective/context
    hashes, policy, release and judge provider/model; failures are kept but
    never reused.
    """
    __tablename__ = "objective_assessments"
    id = db.Column(db.Integer, primary_key=True)
    objective_sha256 = db.Column(db.String(64), nullable=False, index=True)
    context_sha256 = db.Column(db.String(64), nullable=False)
    objective_text = db.Column(db.Text, nullable=False)
    context_text = db.Column(db.Text, nullable=False, default="")
    policy_id = db.Column(db.String(64), nullable=False)
    judge_provider = db.Column(db.String(40))
    judge_model = db.Column(db.String(120))
    label = db.Column(db.String(12))  # Benign | Harmful | Unclear | None
    reason = db.Column(db.Text)
    evidence_ids_json = db.Column(db.Text, default="[]")
    review_reason = db.Column(db.Text)
    output_status = db.Column(db.String(32), nullable=False)  # frozen validator status
    output_class = db.Column(db.String(16), nullable=False)   # valid|needs_review|invalid_output|provider_error
    raw_output = db.Column(db.Text)
    record_json = db.Column(db.Text, nullable=False)
    release_manifest_sha256 = db.Column(db.String(64), nullable=False)
    judge_json = db.Column(db.Text, nullable=False)
    request_call_id = db.Column(db.String(36))
    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)

    @property
    def record(self) -> dict:
        return _loads(self.record_json, {})

    @property
    def evidence_ids(self) -> list:
        return _loads(self.evidence_ids_json, [])

    @property
    def judge(self) -> dict:
        return _loads(self.judge_json, {})


class ResponseAssessment(db.Model):
    """Append-only fulfilment-stage attempt on one captured response.

    A reassessment is a new row; `static_execution_archive_id` is set when a
    static rerun replaces the evidence, so archived attempts leave current results.
    """
    __tablename__ = "response_assessments"
    id = db.Column(db.Integer, primary_key=True)
    benchmark_id = db.Column(db.Integer, db.ForeignKey("benchmarks.id"), nullable=False, index=True)
    prompt_id = db.Column(db.Integer, db.ForeignKey("prompts.id"), nullable=False, index=True)
    static_execution_archive_id = db.Column(db.Integer, db.ForeignKey("static_execution_archives.id"))
    objective_assessment_id = db.Column(db.Integer, db.ForeignKey("objective_assessments.id"))
    requirements_id = db.Column(db.Integer, db.ForeignKey("objective_requirements.id"))
    objective_sha256 = db.Column(db.String(64), nullable=False)
    context_sha256 = db.Column(db.String(64), nullable=False)
    objective_text = db.Column(db.Text, nullable=False)
    context_text = db.Column(db.Text, nullable=False, default="")
    response_sha256 = db.Column(db.String(64), nullable=False)
    capture_complete = db.Column(db.Boolean, nullable=False, default=True)
    # running | valid | abstention | invalid_output | provider_error | not_called | interrupted
    status = db.Column(db.String(16), nullable=False, default="running")
    output_status = db.Column(db.String(32))  # frozen validator status; None when not called
    contribution = db.Column(db.String(8))
    completeness = db.Column(db.String(8))
    fulfilment = db.Column(db.String(16))  # no | partial | complete | cannot_assess
    cannot_assess_reason = db.Column(db.String(24))
    requirement_checks_json = db.Column(db.Text)
    raw_output = db.Column(db.Text)
    record_json = db.Column(db.Text)
    release_manifest_sha256 = db.Column(db.String(64))
    judge_json = db.Column(db.Text)
    request_call_id = db.Column(db.String(36))
    candidate_success = db.Column(db.Boolean, nullable=False, default=False)
    # not_applicable | pending_review | confirmed | rejected
    review_status = db.Column(db.String(16), nullable=False, default="not_applicable")
    previous_benchmark_status = db.Column(db.String(40))
    # Set when the row's evidence was replaced (a static rerun archive or a
    # multi-turn resume/rerun archive). Superseded attempts leave current
    # results; the rows are kept because SQLite may reuse a deleted Prompt id.
    superseded_at = db.Column(db.DateTime)
    superseded_by = db.Column(db.String(80))
    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)
    finished_at = db.Column(db.DateTime)

    # A benchmark deletion removes its assessments. A deleted Prompt (a
    # multi-turn restart) leaves them in place, like the archived rows.
    # cascade "all" (not delete-orphan): rows are created by foreign key.
    benchmark = db.relationship("Benchmark", backref=db.backref(
        "response_assessments", cascade="all", order_by="ResponseAssessment.id"))
    prompt = db.relationship("Prompt", backref=db.backref(
        "response_assessments", passive_deletes="all", order_by="ResponseAssessment.id"))
    objective_assessment = db.relationship("ObjectiveAssessment")
    requirements = db.relationship("ObjectiveRequirements")
    human_reviews = db.relationship("HumanReview", backref="response_assessment",
                                    cascade="all", order_by="HumanReview.id")

    @property
    def record(self) -> dict:
        return _loads(self.record_json, {})

    @property
    def requirement_checks(self) -> list:
        return _loads(self.requirement_checks_json, [])

    @property
    def judge(self) -> dict:
        return _loads(self.judge_json, {})


class HumanReview(db.Model):
    """Blind human labels, collected before the judge's assessment is shown."""
    __tablename__ = "human_reviews"
    id = db.Column(db.Integer, primary_key=True)
    prompt_id = db.Column(db.Integer, db.ForeignKey("prompts.id"), nullable=False, index=True)
    response_assessment_id = db.Column(db.Integer, db.ForeignKey("response_assessments.id"),
                                       nullable=False, index=True)
    objective_label = db.Column(db.String(12), nullable=False)
    fulfilment = db.Column(db.String(16), nullable=False)
    rationale = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)
