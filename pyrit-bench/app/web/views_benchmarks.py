"""Benchmark list / create / detail / status / export / delete views."""
from __future__ import annotations

import csv
import io
import json
from flask import (
    Response, abort, current_app, flash, jsonify, redirect, render_template,
    request, send_file, url_for
)
from sqlalchemy import func
from ..activity import get_activity
from ..artifacts import artifact_file, delete_benchmark_artifacts, has_artifacts
from ..attacks import ATTACKS, by_slug, by_tier
from ..extensions import db
from ..models import Benchmark, Prompt, Target
from ..engine import (
    SUCCESS_SEVERITY, cancel_and_join, request_pause, request_resume, start_benchmark
)
from ..settings import (
    JUDGE_METHOD, JUDGE_METHODS, expected_rows, prompt_cap,
)
from ..engine import two_label

from .blueprint import bp
from .charts import reach_chart
from .error_hints import parse_error_hint
from .report_pdf import render_benchmark_pdf
from .rollups import benchmark_rollups, taxonomy_rollup, two_label_rollup, unit_of


def _two_label_summary(b):
    """Shared two-label rollup, or None for a severity-only run."""
    return two_label_rollup(b) if two_label.run_method(b) != "severity" else None
from .security import csrf_protect
from ..engine.scoring_recovery import claim_retry, start_scoring_retry, fail_evaluation
from ..engine.control import request_cancel
from ..engine.runner import finish_stopped, start_prompt_retry, start_benchmark_recovery


# ---------- Benchmarks ---------------------------------------------------

@bp.route("/benchmarks")
def benchmarks_list():
    rows = Benchmark.query.order_by(Benchmark.created_at.desc()).all()
    return render_template("benchmarks.html", benchmarks=rows)


@bp.route("/benchmarks/compare")
def benchmarks_compare():
    """Overlay several benchmarks: reach curves, ASR, per-taxonomy rates.

    This is where "same target, different attacker/threshold" is actually read
    off. Curves are overlaid on one axis; the ASR and per-taxonomy tables put
    the numbers side by side. Accepts ?ids=1,2,3 (repeatable ?ids= also works).
    """
    ids: list[int] = []
    for raw in request.args.getlist("ids"):
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                ids.append(int(part))
    # de-dupe, preserve order
    seen: set[int] = set()
    ids = [i for i in ids if not (i in seen or seen.add(i))]

    benches = [db.session.get(Benchmark, i) for i in ids]
    benches = [b for b in benches if b is not None]

    if len(benches) < 2:
        flash("Pick at least two benchmarks to compare.", "warning")
        return redirect(url_for("main.benchmarks_list"))

    # Overlaid reach-curve chart (one line per benchmark).
    chart = reach_chart([(b.name, b.reach_curve) for b in benches])

    # Union of taxonomy categories across the selected runs, so the table has a
    # stable row set even when runs used different attacks.
    tax_rates: dict[str, dict[int, dict]] = {}
    for b in benches:
        for cat, agg in taxonomy_rollup(b).items():
            tax_rates.setdefault(cat, {})[b.id] = agg
    tax_rows = sorted(tax_rates.items())

    return render_template(
        "benchmarks_compare.html", benches=benches, compare_chart=chart,
        tax_rows=tax_rows, tl_summaries={b.id: _two_label_summary(b) for b in benches},
    )


@bp.route("/benchmarks/new", methods=["GET", "POST"])
def benchmarks_new():
    targets = Target.query.order_by(Target.name).all()
    if not targets:
        flash("Add a target first.", "warning")
        return redirect(url_for("main.targets_list"))

    from ..engine import load_structured_objectives
    objectives = load_structured_objectives()
    objective_categories = sorted({o["category"] for o in objectives if o.get("category")})

    if request.method == "POST":
        err = csrf_protect()
        if err:
            return err
        slugs = request.form.getlist("attack_slugs")
        if not slugs:
            flash("Pick at least one attack.", "error")
            return redirect(url_for("main.benchmarks_new"))
        try:
            target_id = int(request.form["target_id"])
        except (KeyError, ValueError):
            flash("Invalid target selection.", "error")
            return redirect(url_for("main.benchmarks_new"))
        # New benchmarks use the attacker and judge configured in .env; the
        # runner records what each start resolved (settings.RESOLVED_MODELS_KEY).
        llm_config = {}
        # Store optional run limits; model and fallback settings come from .env.
        raw_cap = (request.form.get("prompts_per_attack") or "").strip()
        if raw_cap:
            try:
                if (n := int(raw_cap)) > 0:
                    llm_config["MAX_PROMPTS_PER_ATTACK"] = n
            except ValueError:
                pass
        selected_categories = set(request.form.getlist("exclude_objective_categories"))
        excl = [c for c in objective_categories if c in selected_categories]
        if excl:
            llm_config["EXCLUDE_OBJECTIVE_CATEGORIES"] = excl
        # Judge method (Task 7); stored on the run so a later .env change never
        # alters it. A two-label run records the verified release identity.
        method = (request.form.get("judge_method") or JUDGE_METHOD).strip().lower()
        if method not in JUDGE_METHODS:
            flash("Unknown judge method.", "error")
            return redirect(url_for("main.benchmarks_new"))
        llm_config["JUDGE_METHOD"] = method
        if method != "severity":
            try:
                llm_config[two_label.CONFIG_KEY] = two_label.release_record()
            except (OSError, ValueError, TypeError, AttributeError):
                flash("The frozen two-label contract release failed its integrity check; "
                      "the run was not created.", "error")
                return redirect(url_for("main.benchmarks_new"))
        b = Benchmark(
            name=request.form.get("name", "").strip() or "Benchmark",
            target_id=target_id,
            attack_slugs_json=json.dumps(slugs),
            llm_config_json=json.dumps(llm_config),
            status="queued",
        )
        db.session.add(b)
        db.session.commit()
        start_benchmark(current_app._get_current_object(), b.id)
        flash(f"Benchmark '{b.name}' queued.", "success")
        return redirect(url_for("main.benchmarks_detail", bid=b.id))

    # group attacks by potency tier (baseline -> adaptive) for display
    tiers = by_tier()
    return render_template(
        "benchmark_new.html", targets=targets, tiers=tiers,
        total_attacks=len(ATTACKS),
        objective_count=len(objectives),
        objective_categories=objective_categories,
        max_prompts=current_app.config["MAX_PROMPTS_PER_ATTACK"],
        judge_method=JUDGE_METHOD,
    )


@bp.route("/benchmarks/<int:bid>")
def benchmarks_detail(bid: int):
    b = Benchmark.query.get_or_404(bid)
    roll = benchmark_rollups(b)
    error_hint = parse_error_hint(b.status_detail) if b.status_detail else None
    curve = b.reach_curve
    per_attack = b.reach_curve_by_attack
    chart_series = [("All attacks", curve)]
    # Only worth splitting out per attack when more than one was run.
    if len(per_attack) > 1:
        chart_series += [((by_slug(k) or {}).get("name", k), v)
                         for k, v in per_attack.items()]
    return render_template(
        "benchmark_detail.html", b=b, by_slug=by_slug, error_hint=error_hint,
        rollup=roll["rollup"], grouped=roll["grouped"], display_groups=roll["display_groups"],
        sev_hist=roll["sev_hist"], avg_severity=roll["avg_severity"], tax_rollup=roll["tax_rollup"],
        rubric_stats=roll["rubric_stats"], mixed_rubrics=roll["mixed_rubrics"],
        totals=roll["totals"],
        success_severity=SUCCESS_SEVERITY,
        reach_curve=curve, reach_chart=reach_chart(chart_series),
        # Retention may have pruned a capture whose ref is still on the row,
        # so the template checks the disk before offering a link.
        has_artifacts=has_artifacts, artifact_file=artifact_file,
        # Current fine-grained phase, for the first server render (the poll keeps
        # it live thereafter). None unless a run thread is active for this id.
        activity=get_activity(b.id),
        judge_method=two_label.run_method(b), tl_summary=_two_label_summary(b),
    )


@bp.route("/benchmarks/<int:bid>/capture/<ref>/<kind>")
def benchmarks_capture(bid: int, ref: str, kind: str):
    """Serve one failure capture: the screenshot inline, the HTML as a download.

    The captured HTML is a hostile page's own markup. Rendering it on this
    origin would let it script against the benchmarking tool, which is served
    on localhost with no authentication — so it is only ever handed over as a
    gzip download, never as text/html. `artifact_file` validates the ref
    against a strict character whitelist and coerces the id to int, so a
    traversal attempt resolves to None and falls through to 404 here.
    """
    path = artifact_file(bid, ref, kind)
    if path is None:
        abort(404)
    if kind == "html":
        resp = send_file(path, mimetype="application/gzip", as_attachment=True,
                         download_name=f"benchmark{bid}-{ref}.html.gz")
    else:
        resp = send_file(path, mimetype="image/png")
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Content-Security-Policy"] = "default-src 'none'"
    return resp


@bp.route("/benchmarks/<int:bid>/status.json")
def benchmarks_status(bid: int):
    b = Benchmark.query.get_or_404(bid)
    roll = benchmark_rollups(b)

    # Live counts derived from Prompt rows (persisted as each prompt completes).
    # Avoids needing to commit progress to the Benchmark row mid-run.
    counts = Prompt.scored_filter()
    total_live = (db.session.query(func.count(Prompt.id))
                  .filter_by(benchmark_id=b.id).filter(counts).scalar() or 0)
    succ_live = (
        db.session.query(func.count(Prompt.id))
        .filter_by(benchmark_id=b.id, succeeded=True).filter(counts).scalar() or 0
    )
    # Rows written so far that carry no evidence — shown next to the totals so a
    # run in trouble reads as "nothing is getting through" rather than as a
    # target holding firm.
    rows_live = (db.session.query(func.count(Prompt.id))
                 .filter_by(benchmark_id=b.id).scalar() or 0)
    captured_live = roll["totals"]["counted"]
    risk_live = round(100.0 * succ_live / total_live, 1) if total_live else None

    # Per-attack progress
    per_attack_rows = (
        db.session.query(Prompt.attack_slug, func.count(Prompt.id), func.sum(Prompt.succeeded))
        .filter_by(benchmark_id=b.id).filter(Prompt.scored_filter())
        .group_by(Prompt.attack_slug)
        .all()
    )
    per_attack = [
        {
            "slug": slug,
            "name": (by_slug(slug) or {}).get("name", slug),
            "total": int(total or 0),
            "succ": int(succ or 0),
        }
        for slug, total, succ in per_attack_rows
    ]

    # Currently running attack = the slug of the most recently inserted Prompt row.
    latest = (
        db.session.query(Prompt.attack_slug)
        .filter_by(benchmark_id=b.id)
        .order_by(Prompt.id.desc()).first()
    )
    current_slug = latest[0] if latest else None
    current_name = (by_slug(current_slug) or {}).get("name") if current_slug else None

    # Expected total rows, so the progress fraction and bar stay in the same
    # units as the numerator. `p-done` counts every Prompt row written, and a
    # single seed/objective fans out into many rows — a converter `sweep` tries
    # each chain, best-of-N re-runs `attempts` times, a multi-turn conversation
    # records one row per turn. Summing per-attack fan-out (see expected_rows)
    # instead of `attacks × cap` is what stops the display reading "556 / 280".
    # The benchmark's own cap, not the current .env value: a run queued under a
    # preset at 3 prompts/attack must not have its bar drawn against 10 (and a
    # later .env change must not retroactively move the denominator of a run
    # that has already finished).
    cap = prompt_cap(b, current_app.config["MAX_PROMPTS_PER_ATTACK"])
    expected_total = sum(
        expected_rows(a, cap)
        for slug in b.attack_slugs
        if (a := by_slug(slug))
    )

    tl_summary = _two_label_summary(b)
    if tl_summary is not None:
        # Same rollup and escaped partial as the first render.
        tl_summary = {**tl_summary, "html": render_template(
            "_two_label_summary.html", b=b, tl_summary=tl_summary)}

    return jsonify(
        two_label=tl_summary,
        prompt_results_html=render_template(
            '_benchmark_prompt_results.html', b=b, by_slug=by_slug,
            display_groups=roll['display_groups'], artifact_file=artifact_file),
        status=b.status, total=total_live, succ=succ_live, risk=risk_live,
        captured=captured_live, scored=total_live, unscored=captured_live-total_live,
        rows=rows_live, rows_total=rows_live, excluded=rows_live - captured_live,
        latency=b.avg_latency_ms, detail=b.status_detail,
        expected_total=expected_total,
        current_attack_slug=current_slug,
        current_attack_name=current_name,
        per_attack=per_attack,
        rollup=roll["rollup"], tax_rollup=roll["tax_rollup"],
        sev_hist=roll["sev_hist"], avg_severity=roll["avg_severity"],
        rubric_stats=roll["rubric_stats"], mixed_rubrics=roll["mixed_rubrics"],
        # Fine-grained "what's happening right now" — set by the run thread as it
        # moves through generate/send/judge/rate-limit/throttle/pause phases.
        activity=get_activity(b.id),
    )


@bp.route("/benchmarks/<int:bid>/curve.csv")
def benchmarks_curve_csv(bid: int):
    """Export the success-vs-attempt-budget curve, overall and per attack."""
    b = Benchmark.query.get_or_404(bid)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["attack", "attempt_budget_n", "objectives_reached", "objectives_total",
                "reach_rate_pct"])
    for pt in b.reach_curve:
        w.writerow(["ALL", pt["n"], pt["reached"], pt["groups"], f"{pt['rate']:.1f}"])
    for slug, curve in b.reach_curve_by_attack.items():
        for pt in curve:
            w.writerow([slug, pt["n"], pt["reached"], pt["groups"], f"{pt['rate']:.1f}"])
    return Response(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=benchmark-{b.id}-curve.csv"},
    )


@bp.route("/benchmarks/<int:bid>/csv")
def benchmarks_csv(bid: int):
    b = Benchmark.query.get_or_404(bid)
    buf = io.StringIO()
    w = csv.writer(buf)
    # `status` and `counts` come first among the new columns because every
    # analysis has to filter on them: a row that never reached the target is not
    # a zero. `verdict` is the judge's reasoning, so a severity can be defended
    # rather than asserted; `severity_source` and `latency_measured` say which
    # numbers were measured and which were not. The run-level model columns are
    # appended last so readers that index the older columns keep working.
    w.writerow(["benchmark", "target", "architecture", "attack", "category", "taxonomy",
                "objective_id", "attempt_group", "attempt", "seq", "prompt", "converter",
                "response", "latency_ms", "latency_measured", "refused", "succeeded",
                "severity", "severity_source", "status", "counts_toward_stats",
                "verdict", "note", "error", "judge_model", "models_used",
                "has_valid_score", "scoring_evaluations", "archived_executions",
                "observed_target_invocations", "observed_judge_invocations", "latency_basis",
                # Task 7 two-label columns, trailing so older readers keep working.
                "judge_method", "objective_label", "objective_status", "fulfilment",
                "contribution", "completeness", "cannot_assess_reason", "candidate_success",
                "review_status", "human_objective_label", "human_fulfilment", "two_label_attempts",
                # Task 8: multi-turn ledger counts per conversation (blank for static rows).
                "conversation_target_sends", "conversation_attacker_calls", "conversation_judge_internal_calls",
                # Task 7 review: assessment unit, recorded fence deviation, two-label judge calls.
                "unit", "fence_stripped", "two_label_judge_invocations"])
    models = b.models_used
    summary = b.llm_config_summary
    method = two_label.run_method(b)
    two_label.preload(b.id)
    from ..engine.trajectory import attempt_of, conversation_call_counts
    conversation_calls = conversation_call_counts(b.id)
    for p in b.prompts:
        atk = by_slug(p.attack_slug) or {}
        attempts = p.current_request_attempts
        observed = {stage: sum(a.stage == stage for a in attempts)
                    for stage in ("target_send", "judge", two_label.STAGE)}
        tl = two_label.current_assessment(p)
        objective = tl.objective_assessment if tl is not None else None
        review = tl.human_reviews[-1] if tl is not None and tl.human_reviews else None
        fenced = [name for name, record in (("objective", objective.record if objective else {}),
                                            ("fulfilment", tl.record if tl else {}))
                  if record.get("fence_stripped")]
        review_cols = [unit_of(p), ",".join(fenced), observed[two_label.STAGE] or ""]
        two_label_cols = [
            two_label.row_method(p) if method != "severity" else method,
            (objective.label or "") if objective else "",
            objective.output_class if objective else "",
            (tl.fulfilment or "") if tl else "", (tl.contribution or "") if tl else "",
            (tl.completeness or "") if tl else "", (tl.cannot_assess_reason or "") if tl else "",
            int(bool(tl.candidate_success)) if tl else "", tl.review_status if tl else "",
            review.objective_label if review else "", review.fulfilment if review else "",
            len(two_label.attempts_for(p)) or ""]
        if atk.get("kind") == "dynamic" and p.objective_id:
            calls = conversation_calls.get((p.attack_slug, p.objective_id, p.attempt or 0, attempt_of(p.sequence)), {})
            conversation_cols = [calls.get("target_send", ""), calls.get("attacker_call", ""),
                                 calls.get("judge_internal", "")]
        else:
            conversation_cols = ["", "", ""]
        w.writerow([b.name, b.target.name, b.target.architecture or "",
                    p.attack_slug, atk.get("category", ""),
                    p.taxonomy_category or "", p.objective_id or "",
                    p.attempt_group or "", p.attempt or 0,
                    p.sequence, p.prompt_text[:500], p.converter_used or "",
                    (p.response or "")[:500], p.latency_ms or "",
                    "" if p.latency_measured is None else int(bool(p.latency_measured)),
                    int(bool(p.refused)), int(bool(p.succeeded)),
                    p.severity if p.severity is not None else "",
                    p.severity_source or "", p.status or "",
                    int(bool(p.counts_toward_stats)),
                    (p.verdict or "")[:500], (p.note or "")[:300],
                    # Full error text, untruncated — Python tracebacks put the
                    # actual exception last, so a prefix cut loses the answer.
                    p.error or "", models["judge"], summary, int(p.has_valid_score), len(p.current_scoring_evaluations), len(p.execution_history),
                    observed["target_send"] or "", observed["judge"] or "",
                    "static send completion including retries/backoff, minus throttle rest" if atk.get("kind") == "static" else "strategy-specific; may be derived",
                    *two_label_cols, *conversation_cols, *review_cols])
    return Response(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=benchmark-{b.id}.csv"},
    )


@bp.route("/benchmarks/<int:bid>/pdf")
def benchmarks_pdf(bid: int):
    """Generate a PDF summary report for the benchmark."""
    b = Benchmark.query.get_or_404(bid)
    try:
        pdf_bytes = render_benchmark_pdf(b)
    except ImportError as exc:
        # reportlab missing — render a plain-text 500 with a clear hint
        return Response(
            f"PDF export requires the 'reportlab' package: {exc}\n"
            f"Install with: pip install reportlab",
            status=500, mimetype="text/plain",
        )
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=benchmark-{b.id}.pdf"},
    )


@bp.route("/benchmarks/<int:bid>/pause", methods=["POST"])
def benchmarks_pause(bid: int):
    """Ask a running benchmark to pause at its next checkpoint.

    The background thread pauses cooperatively — it stops between prompts, not
    mid-request — so a target call already in flight finishes first. The status
    flips to "paused" immediately here so the page reflects the click, and the
    run loop settles onto a checkpoint within a prompt or two (see
    engine/control.py).
    """
    err = csrf_protect()
    if err:
        return err
    b = Benchmark.query.get_or_404(bid)
    if b.status == "running":
        request_pause(b.id)
        b.status = "paused"
        db.session.commit()
    else:
        flash(f"'{b.name}' is not running, so it can't be paused.", "error")
    return redirect(url_for("main.benchmarks_detail", bid=b.id))


@bp.route("/benchmarks/<int:bid>/resume", methods=["POST"])
def benchmarks_resume(bid: int):
    """Resume a paused or interrupted benchmark.

    "paused": the run thread is still alive in this process — clear the
    in-memory flag and the loop continues where it was.

    "interrupted": the process restarted, so the run thread is gone. Launch a
    fresh one; it re-enters the attack loop and skips attacks already recorded
    as completed (see engine/runner._run_attacks). An unfinished static attack
    keeps its saved rows: failed sends and converter errors or refusals are
    retried after their previous execution is archived, and missing slots are
    filled, apart from those best-of-N skipped after an earlier success (see
    engine/static_attack.run_static_attack). Unfinished multi-turn and Azure
    attacks continue from their saved turns (Task 8): finished objectives are
    kept, an interrupted conversation is replayed to PyRIT as a prepended
    conversation, and replaced rows are archived, never deleted.
    started_at is preserved so the run's duration still covers the whole run."""
    err = csrf_protect()
    if err:
        return err
    b = Benchmark.query.get_or_404(bid)
    if b.status == "paused":
        request_resume(b.id)
        b.status = "running"
        db.session.commit()
    elif b.status == "interrupted":
        b.status = "running"
        db.session.commit()
        # current_app is a proxy; the thread needs the real app object.
        start_benchmark(current_app._get_current_object(), b.id)
    else:
        flash(f"'{b.name}' is not paused or interrupted.", "error")
    return redirect(url_for("main.benchmarks_detail", bid=b.id))


@bp.route("/benchmarks/<int:bid>/stop", methods=["POST"])
def benchmarks_stop(bid: int):
    err = csrf_protect()
    if err:
        return err
    b = Benchmark.query.get_or_404(bid)
    if b.status in ("queued", "running", "paused"):
        b.status = "stopping"
        db.session.commit()
        request_cancel(b.id)
        flash("Stopping after the current request or attack objective finishes. Results will be kept.", "success")
    elif b.status == "interrupted":
        finish_stopped(b.id)
    return redirect(url_for("main.benchmarks_detail", bid=b.id))


@bp.route("/benchmarks/<int:bid>/delete", methods=["POST"])
def benchmarks_delete(bid: int):
    err = csrf_protect()
    if err:
        return err
    b = Benchmark.query.get_or_404(bid)
    # A queued/running benchmark has a daemon thread writing Prompt rows for it
    # in the background (see runner.start_benchmark). Deleting the row out from
    # under that thread doesn't stop it — it keeps committing Prompt rows
    # against a benchmark_id that no longer exists (SQLite doesn't enforce the
    # FK), which silently orphans data instead of erroring.
    #
    # A PAUSED benchmark is the exception we can handle cleanly: its thread is
    # parked at a checkpoint, so we signal a cancel and wait for it to unwind
    # (cancel_and_join) before removing the row. The run loop only checks the
    # cancel between prompts, so nothing is half-written when it stops.
    if b.status == "paused":
        if not cancel_and_join(b.id):
            flash("The worker is still stopping. Results have been kept; try deleting again after it stops.", "error")
            return redirect(url_for("main.benchmarks_detail", bid=b.id))
    elif b.status in ("queued", "running", "stopping", "scoring"):
        flash(
            f"Cannot delete '{b.name}' — it is still {b.status}. "
            "Pause it first (or wait for it to finish or fail), then delete.",
            "error",
        )
        return redirect(url_for("main.benchmarks_detail", bid=b.id))
    db.session.delete(b)
    db.session.commit()
    # cascade="all, delete-orphan" removes the Prompt rows; nothing but this
    # removes the failure captures those rows pointed at.
    delete_benchmark_artifacts(bid)
    return redirect(url_for("main.benchmarks_list"))


@bp.route("/benchmarks/<int:bid>/prompts/<int:pid>/executions.json")
def benchmarks_prompt_executions(bid: int, pid: int):
    """Lossless audit export; only the current row enters benchmark totals."""
    p = Prompt.query.filter_by(id=pid, benchmark_id=bid).first_or_404()
    return jsonify(
        selection_rule="Only the current Prompt row enters current-result summaries; archives are excluded.",
        timestamp_timezone="UTC",
        request_attempts=[a.as_dict() for a in p.request_attempts],
        current=p.execution_snapshot(),
        previous=[{"archive_id": item.id, "replaced_at": item.replaced_at.isoformat(),
                   "original_row": item.original_row} for item in p.execution_history],
        scoring_evaluations=[{
            "id": e.id, "static_execution_archive_id": e.static_execution_archive_id,
            "context": e.context, "elapsed_ms": e.elapsed_ms,
            "previous_benchmark_status": e.previous_benchmark_status,
            "started_at": e.started_at.isoformat() if e.started_at else None,
            "finished_at": e.finished_at.isoformat() if e.finished_at else None,
            "status": e.status, "response_sha256": e.context.get("response_sha256"),
            "request_sha256": e.context.get("request_sha256"),
            "verdict": e.verdict, "severity": e.severity,
            "severity_source": e.severity_source, "succeeded": e.succeeded,
        } for e in p.scoring_evaluations],
        two_label_assessments=[{
            "id": a.id, "static_execution_archive_id": a.static_execution_archive_id,
            "status": a.status, "fulfilment": a.fulfilment, "contribution": a.contribution,
            "completeness": a.completeness, "cannot_assess_reason": a.cannot_assess_reason,
            "candidate_success": a.candidate_success, "review_status": a.review_status,
            "response_sha256": a.response_sha256, "objective_sha256": a.objective_sha256,
            "requirements_id": a.requirements_id, "release_manifest_sha256": a.release_manifest_sha256,
            "judge": a.judge, "request_call_id": a.request_call_id, "record": a.record,
            "objective": ({"id": a.objective_assessment.id, "label": a.objective_assessment.label,
                           "output_status": a.objective_assessment.output_status,
                           "record": a.objective_assessment.record}
                          if a.objective_assessment else None),
            "human_reviews": [{"objective_label": r.objective_label, "fulfilment": r.fulfilment,
                               "rationale": r.rationale, "created_at": r.created_at.isoformat()}
                              for r in a.human_reviews],
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "finished_at": a.finished_at.isoformat() if a.finished_at else None,
            # A superseded attempt belongs to replaced evidence and is not a current result.
            "superseded": a.superseded_at is not None,
            "superseded_at": a.superseded_at.isoformat() if a.superseded_at else None,
            "superseded_by": a.superseded_by,
        } for a in p.response_assessments],
    )


@bp.route("/benchmarks/<int:bid>/request-attempts.json")
def benchmarks_request_attempts(bid: int):
    b = Benchmark.query.get_or_404(bid)
    return jsonify(
        scope="Observed local helper invocations: static target sends and saved-judge calls per row, and for multi-turn attacks the target sends, attacker calls and PyRIT-internal judge calls linked to each conversation (Task 8). SDK/internal retries, converter calls and Azure scans are not counted. Missing history is unknown, not zero.",
        timestamp_timezone="UTC",
        request_attempts=[a.as_dict() for a in b.request_attempts],
    )


@bp.route("/benchmarks/<int:bid>/prompts/<int:pid>/retry-scoring", methods=["POST"])
def benchmarks_retry_scoring(bid: int, pid: int):
    err = csrf_protect()
    if err:
        return err
    b = Benchmark.query.get_or_404(bid)
    p = Prompt.query.filter_by(id=pid, benchmark_id=bid).first_or_404()
    try:
        evaluation = claim_retry(b, p)
    except ValueError as exc:
        abort(409, description=str(exc))
    try:
        start_scoring_retry(current_app._get_current_object(), evaluation.id)
    except Exception:
        fail_evaluation(evaluation, "Scoring worker could not start; no judge request was made.")
        b.status = evaluation.previous_benchmark_status
        db.session.commit()
        raise
    flash("Retrying scoring on the saved response. No new target request will be made.", "success")
    return redirect(url_for("main.benchmarks_detail", bid=bid))


@bp.route("/benchmarks/<int:bid>/prompts/<int:pid>/rerun", methods=["POST"])
def benchmarks_rerun_prompt(bid: int, pid: int):
    err = csrf_protect()
    if err:
        return err
    b = Benchmark.query.get_or_404(bid)
    p = Prompt.query.filter_by(id=pid, benchmark_id=bid).first_or_404()
    if not p.can_rerun:
        abort(400, description="Only failed prompt attempts or conversations with an objective can be rerun.")
    previous = b.status
    # Claim the benchmark atomically so double-clicks cannot start two workers.
    claimed = Benchmark.query.filter(
        Benchmark.id == bid,
        Benchmark.status.in_(("completed", "failed", "completed_with_target_errors", "stopped", "interrupted")),
    ).update({Benchmark.status: "running"}, synchronize_session=False)
    db.session.commit()
    if not claimed:
        flash("Wait until this benchmark stops before rerunning a prompt.", "warning")
    else:
        try:
            start_prompt_retry(current_app._get_current_object(), bid, pid, previous)
        except Exception:
            b.status = previous
            db.session.commit()
            raise
        flash("Resuming the conversation from its saved turns (or from the beginning when none are usable)." if p.retry_label == "Rerun conversation" else "Rerunning the failed prompt. Its result will update when finished.", "success")
    return redirect(url_for("main.benchmarks_detail", bid=bid))


@bp.route("/benchmarks/<int:bid>/retry-failed", methods=["POST"])
def benchmarks_retry_failed(bid: int, skip_failed: bool = False):
    err = csrf_protect()
    if err:
        return err
    b = Benchmark.query.get_or_404(bid)
    previous = b.status
    claimed = Benchmark.query.filter(
        Benchmark.id == bid,
        Benchmark.status.in_(("completed", "failed", "completed_with_target_errors", "stopped", "interrupted")),
    ).update({Benchmark.status: "running"}, synchronize_session=False)
    db.session.commit()
    if not claimed:
        flash("This benchmark is already active. Wait for it to stop before retrying.", "warning")
    else:
        try:
            if skip_failed:
                start_benchmark_recovery(current_app._get_current_object(), bid, retry_failed=False)
            else:
                start_benchmark_recovery(current_app._get_current_object(), bid)
        except Exception:
            b.status = previous
            db.session.commit()
            raise
        flash("Skipping existing failures and continuing unattempted tests. Failed results remain recorded."
              if skip_failed else "Retrying failed parts and continuing unfinished tests. Completed results will be kept.", "success")
    return redirect(url_for("main.benchmarks_detail", bid=bid))


@bp.route("/benchmarks/<int:bid>/skip-failed", methods=["POST"])
def benchmarks_skip_failed(bid: int):
    return benchmarks_retry_failed(bid, skip_failed=True)
