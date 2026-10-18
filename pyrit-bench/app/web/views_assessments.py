"""Task 7 (October 2026): two-label assessment routes (requirements, review, retry).

Owned by the Task 7 lane. Registered on the shared blueprint by app/web/__init__.py.
"""
from __future__ import annotations

import random

from flask import abort, current_app, flash, redirect, render_template, request, url_for

from ..engine import two_label
from ..engine.datasets import load_structured_objectives
from ..extensions import db
from ..models import Benchmark, Prompt
from ..two_label_models import ObjectiveAssessment, ObjectiveRequirements, ResponseAssessment
from .blueprint import bp
from .security import csrf_protect


@bp.app_context_processor
def _two_label_helpers():
    # Templates call two_label.current_assessment(p), two_label.run_method(b), ...
    return {"two_label": two_label}


# ---------- requirement lists ------------------------------------------------

def _versions(objective_sha256: str) -> list:
    return (ObjectiveRequirements.query
            .filter_by(objective_sha256=objective_sha256, context_sha256=two_label.text_sha(""))
            .order_by(ObjectiveRequirements.version.desc()).all())


def _entry(text: str, source: str, objective_id=None) -> dict:
    sha = two_label.text_sha(text)
    return {"sha": sha, "anchor": "obj-" + sha[:12], "text": text, "source": source,
            "objective_id": objective_id, "versions": _versions(sha)}


def objective_entries(bench) -> dict:
    """Objectives of one run, objectives.json and any other objective with a list.

    The app supplies no separate objective context, so every list is keyed with
    the empty context, exactly as the run assesses it.
    """
    seen: set[str] = set()

    def add(bucket, text, source, objective_id=None):
        if text and text.strip() and two_label.text_sha(text) not in seen:
            seen.add(two_label.text_sha(text))
            bucket.append(_entry(text, source, objective_id))

    structured = load_structured_objectives()
    by_id = {row.get("id"): row.get("objective") for row in structured}
    run_rows: list = []
    if bench is not None:
        for p in sorted(bench.prompts, key=lambda x: (x.attack_slug, x.sequence)):
            attempts = [a for a in p.response_assessments]
            text = (attempts[-1].objective_text if attempts else
                    (p.scoring_context or {}).get("objective") or by_id.get(p.objective_id))
            add(run_rows, text, p.attack_slug, p.objective_id)
    json_rows: list = []
    for row in structured:
        add(json_rows, row.get("objective"), "objectives.json", row.get("id"))
    other_rows: list = []
    for row in ObjectiveRequirements.query.order_by(ObjectiveRequirements.id).all():
        add(other_rows, row.objective_text, "requirement list")
    return {"run": run_rows, "structured": json_rows, "other": other_rows}


def known_objective_text(sha: str, bench):
    """Resolve a posted objective hash to its exact text (form fields alter newlines)."""
    for bucket in objective_entries(bench).values():
        for entry in bucket:
            if entry["sha"] == sha:
                return entry["text"]
    for model in (ObjectiveAssessment, ResponseAssessment):
        row = model.query.filter_by(objective_sha256=sha).first()
        if row is not None:
            return row.objective_text
    return None


@bp.route("/requirements", methods=["GET", "POST"])
def requirements_page():
    bid = request.values.get("bid", type=int)
    bench = db.session.get(Benchmark, bid) if bid else None
    back = url_for("main.requirements_page", bid=bench.id) if bench else url_for("main.requirements_page")
    if request.method == "POST":
        err = csrf_protect()
        if err:
            return err
        sha = (request.form.get("objective_sha256") or "").strip()
        objective = (known_objective_text(sha, bench) if sha
                     else (request.form.get("objective_text") or "").replace("\r\n", "\n"))
        if not objective:
            flash("Requirement list not saved: unknown or blank objective.", "error")
            return redirect(back)
        try:
            requirements = two_label.parse_requirements(request.form.get("requirements", ""),
                                                        request.form.get("format", "lines"))
            row = two_label.save_requirements(objective, requirements, note=request.form.get("note"))
        except (ValueError, TypeError) as exc:
            flash(f"Requirement list not saved: {exc}", "error")
            return redirect(back)
        flash(f"Saved requirement list version {row.version}. Earlier versions are kept.", "success")
        return redirect(back + "#obj-" + row.objective_sha256[:12])
    return render_template("requirements.html", bench=bench, entries=objective_entries(bench))


# ---------- judge-only retry ---------------------------------------------------

@bp.route("/benchmarks/<int:bid>/prompts/<int:pid>/retry-two-label", methods=["POST"])
def assessments_retry(bid: int, pid: int):
    err = csrf_protect()
    if err:
        return err
    b = Benchmark.query.get_or_404(bid)
    p = Prompt.query.filter_by(id=pid, benchmark_id=bid).first_or_404()
    try:
        attempt = two_label.claim_retry(b, p)
    except ValueError as exc:
        abort(409, description=str(exc))
    try:
        two_label.start_retry(current_app._get_current_object(), attempt.id)
    except Exception:
        attempt.status = "interrupted"
        b.status = attempt.previous_benchmark_status
        db.session.commit()
        raise
    flash("Retrying two-label assessment on the saved response. No new target request will be made.", "success")
    return redirect(url_for("main.benchmarks_detail", bid=bid))


# ---------- blind human review ----------------------------------------------

@bp.route("/benchmarks/<int:bid>/review-queue")
def assessments_review_queue(bid: int):
    """The only entry point to the blind form: every row with a finished
    two-label attempt, shuffled, with no labels, badges, candidate wording or
    counts, so a reviewer is never shown the judge's view first."""
    from ..attacks import by_slug
    from .text import shorten
    b = Benchmark.query.get_or_404(bid)
    two_label.preload(b.id)
    rows = []
    for p in b.prompts:
        if not p.counts_toward_stats or two_label.row_method(p) == "severity":
            continue
        attempt = two_label.current_assessment(p)
        if attempt is None or attempt.status in ("running", "interrupted"):
            continue
        rows.append({"prompt_id": p.id, "attack": (by_slug(p.attack_slug) or {}).get("name", p.attack_slug),
                     "excerpt": shorten(p.prompt_text, 160), "reviewed": bool(attempt.human_reviews)})
    random.SystemRandom().shuffle(rows)
    return render_template("_two_label_review_queue.html", b=b, rows=rows)

@bp.route("/benchmarks/<int:bid>/prompts/<int:pid>/review", methods=["GET", "POST"])
def assessments_review(bid: int, pid: int):
    b = Benchmark.query.get_or_404(bid)
    p = Prompt.query.filter_by(id=pid, benchmark_id=bid).first_or_404()
    attempt = two_label.current_assessment(p)
    if attempt is None or attempt.status in ("running", "interrupted"):
        abort(404, description="This response has no finished two-label assessment to review.")
    review = attempt.human_reviews[-1] if attempt.human_reviews else None
    here = url_for("main.assessments_review", bid=bid, pid=pid)
    if request.method == "POST":
        err = csrf_protect()
        if err:
            return err
        if review is None:
            try:
                two_label.record_review(attempt, objective_label=request.form.get("objective_label", ""),
                                        fulfilment=request.form.get("fulfilment", ""),
                                        rationale=request.form.get("rationale", ""))
            except ValueError as exc:
                flash(f"Review not saved: {exc}", "error")
        return redirect(here)
    if review is None:
        # Blind form: only the task, the prompt as sent and the response reach
        # the page. The judge's labels, reasons and raw output are not passed.
        requirements = attempt.requirements.requirements if attempt.requirements else []
        return render_template("_two_label_review.html", b=b, p=p, blind=True,
                               objective_text=attempt.objective_text, prompt_text=p.prompt_text,
                               response_lines=two_label.numbered_lines(p.response or ""),
                               requirements=requirements, labels=two_label.LABELS,
                               fulfilments=two_label.FULFILMENT_NAMES)
    return render_template("_two_label_review.html", b=b, p=p, blind=False, attempt=attempt,
                           review=review, objective_text=attempt.objective_text,
                           prompt_text=p.prompt_text,
                           response_lines=two_label.numbered_lines(p.response or ""),
                           fulfilments=two_label.FULFILMENT_NAMES)
