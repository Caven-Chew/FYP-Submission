"""Benchmark list / create / detail / status / export / delete views."""
from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import os
from flask import (
    Response, abort, current_app, flash, jsonify, redirect, render_template,
    request, send_file, session, url_for
)
from sqlalchemy import func
from ..activity import get_activity
from ..artifacts import artifact_file, delete_benchmark_artifacts, has_artifacts
from ..attacks import ATTACKS, PRESETS, by_slug, by_tier
from ..extensions import db
from ..models import Benchmark, Prompt, Target
from ..engine import (
    SUCCESS_SEVERITY, cancel_and_join, request_pause, request_resume, start_benchmark
)
from ..settings import (
    CONVERTER_BACKUP, CONVERTER_ON_BLOCK, expected_rows, prompt_cap,
)

from .blueprint import bp
from .charts import reach_chart
from .error_hints import parse_error_hint
from .report_pdf import render_benchmark_pdf
from .rollups import benchmark_rollups, taxonomy_rollup
from .security import csrf_protect


# Roles whose model the benchmark form can override, with a blank -> "use .env"
# option first so the default path needs no input.
LLM_ROLE_FIELDS = [
    ("ATTACKER", "Attacker model",
     "Generates adversarial prompts for dynamic attacks and the semantic converters. "
     "A less safety-trained model is usually the stronger attacker."),
    ("JUDGE", "Judge model",
     "Scores each target response 1-5. The workshop uses an HPC LLM judge "
     "and a separate HPC LLM attacker model for supported prompt rewrites."),
]

LLM_PROVIDER_OPTIONS = (
    ("prof_llm", "HPC LLM (university service)"),
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
    ("venice", "Venice"),
)
LLM_PROVIDERS = tuple(slug for slug, _label in LLM_PROVIDER_OPTIONS)
_DISPLAY_PROVIDER_NAMES = {slug: label.split(" (")[0] for slug, label in LLM_PROVIDER_OPTIONS}

# Workshop-only convenience gate. This is deliberately a facilitator control,
# not a security boundary: the application is local and the permitted attack
# catalog is shipped on the participant's machine. Keeping only the digest out
# of the template prevents the password appearing in page source or results.
_ATTACK_UNLOCK_DIGEST = (
    "50f87d0665459ff3e0d85c7f2fe4207fba639abee68d1edfd1639584095a9517"
)
_ATTACK_UNLOCK_SESSION_KEY = "workshop_extra_attacks_unlocked"


def valid_attack_unlock_password(candidate: str) -> bool:
    digest = hashlib.sha256((candidate or "").encode("utf-8")).hexdigest()
    return hmac.compare_digest(digest, _ATTACK_UNLOCK_DIGEST)


def extra_attacks_unlocked() -> bool:
    return session.get(_ATTACK_UNLOCK_SESSION_KEY) is True


def env_llm_defaults() -> dict:
    """What each role would use from .env, shown as the '(env default)' hint."""
    out = {}
    for role, _, _ in LLM_ROLE_FIELDS:
        prov = (os.getenv(f"{role}_PROVIDER") or "").strip().lower()
        display = _DISPLAY_PROVIDER_NAMES.get(prov, prov)
        out[role] = f"{display}/{os.getenv(f'{role}_MODEL') or 'default model'}" if prov else "not configured"
    return out


def parse_attacker_roster(raw: str) -> list[dict]:
    """Parse the roster textarea: one `provider:model` per line.

    e.g. "openai:gpt-4o-mini" / "anthropic:claude-3-5-haiku-latest". Blank lines
    ignored; an unknown provider is skipped. Roster entries reuse the API key
    from .env for their provider (kept simple — inline keys stay on the single
    attacker fields).
    """
    out: list[dict] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        prov, _, model = line.partition(":")
        prov = prov.strip().lower()
        if prov not in LLM_PROVIDERS:
            continue
        entry = {"provider": prov}
        if model.strip():
            entry["model"] = model.strip()
        out.append(entry)
    return out


def collect_llm_overrides(form) -> dict:
    """Read per-role LLM overrides from the benchmark form.

    A role is only overridden when a provider is picked; a blank provider means
    "use the .env default". API keys entered here are stored on the benchmark
    rather than the environment, so they never touch .env.
    """
    cfg: dict[str, dict] = {}
    for role, _, _ in LLM_ROLE_FIELDS:
        provider = (form.get(f"{role.lower()}_provider") or "").strip().lower()
        if provider not in LLM_PROVIDERS:
            continue
        entry = {"provider": provider}
        model = (form.get(f"{role.lower()}_model") or "").strip()
        if model:
            entry["model"] = model
        api_key = (form.get(f"{role.lower()}_api_key") or "").strip()
        if api_key:
            entry["api_key"] = api_key
        endpoint = (form.get(f"{role.lower()}_endpoint") or "").strip()
        if endpoint:
            entry["endpoint"] = endpoint
        cfg[role] = entry
    return cfg


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
        tax_rows=tax_rows,
    )


@bp.route("/benchmarks/new", methods=["GET", "POST"])
def benchmarks_new():
    targets = Target.query.order_by(Target.name).all()
    if not targets:
        flash("Add a target first.", "warning")
        return redirect(url_for("main.targets_list"))

    if request.method == "POST":
        err = csrf_protect()
        if err:
            return err
        action = (request.form.get("form_action") or "").strip()
        if action == "unlock_attacks":
            if valid_attack_unlock_password(
                request.form.get("attack_unlock_password", "")
            ):
                session[_ATTACK_UNLOCK_SESSION_KEY] = True
                flash("Additional attacks unlocked for this session.", "success")
            else:
                session.pop(_ATTACK_UNLOCK_SESSION_KEY, None)
                flash("Incorrect facilitator password.", "error")
            return redirect(url_for("main.benchmarks_new"))
        if action == "lock_attacks":
            session.pop(_ATTACK_UNLOCK_SESSION_KEY, None)
            flash("Additional attacks locked.", "success")
            return redirect(url_for("main.benchmarks_new"))

        slugs = request.form.getlist("attack_slugs")
        if not slugs:
            flash("Pick at least one attack.", "error")
            return redirect(url_for("main.benchmarks_new"))
        workshop_slugs = set(PRESETS["workshop"]["attack_slugs"])
        extra_slugs = [slug for slug in slugs if slug not in workshop_slugs]
        if extra_slugs and not extra_attacks_unlocked():
            flash(
                "Additional attacks are locked. Ask the facilitator to unlock them.",
                "error",
            )
            return redirect(url_for("main.benchmarks_new"))
        try:
            target_id = int(request.form["target_id"])
        except (KeyError, ValueError):
            flash("Invalid target selection.", "error")
            return redirect(url_for("main.benchmarks_new"))
        llm_config = collect_llm_overrides(request.form)
        # Converter fallback: a backup converter model to retry a rewrite the
        # primary model's content filter blocked, and what to do if the backup is
        # also blocked. Stored as reserved (non-role) keys on the same config dict
        # — role builders only read the ATTACKER/JUDGE keys, so these are inert to
        # them and read directly by engine.static_attack.
        backup = (request.form.get("converter_backup") or "").strip()
        if backup:
            llm_config["CONVERTER_BACKUP"] = backup
        on_block = (request.form.get("converter_on_block") or "skip").strip().lower()
        llm_config["CONVERTER_ON_BLOCK"] = "stop" if on_block == "stop" else "skip"
        # Per-run prompt cap. Stored on the benchmark (reserved non-role key, as
        # above) rather than written to .env, so a workshop run at 2 leaves the
        # environment — and the next research run — untouched. Blank means "use
        # the .env cap"; a junk value is ignored the same way.
        raw_cap = (request.form.get("prompts_per_attack") or "").strip()
        if raw_cap:
            try:
                if (n := int(raw_cap)) > 0:
                    llm_config["MAX_PROMPTS_PER_ATTACK"] = n
            except ValueError:
                pass
        excl = [
            c.strip() for c in
            (request.form.get("exclude_objective_categories") or "").split(",")
            if c.strip()
        ]
        if excl:
            llm_config["EXCLUDE_OBJECTIVE_CATEGORIES"] = excl
        roster = parse_attacker_roster(request.form.get("attacker_roster", ""))
        b = Benchmark(
            name=request.form.get("name", "").strip() or "Benchmark",
            target_id=target_id,
            attack_slugs_json=json.dumps(slugs),
            llm_config_json=json.dumps(llm_config),
            attacker_roster_json=json.dumps(roster),
            status="queued",
        )
        db.session.add(b)
        db.session.commit()
        start_benchmark(current_app._get_current_object(), b.id)
        flash(f"Benchmark '{b.name}' queued.", "success")
        return redirect(url_for("main.benchmarks_detail", bid=b.id))

    # group attacks by potency tier (baseline -> adaptive) for display
    tiers = by_tier()
    from ..engine import load_structured_objectives
    return render_template(
        "benchmark_new.html", targets=targets, tiers=tiers,
        total_attacks=len(ATTACKS),
        llm_roles=LLM_ROLE_FIELDS, env_llm=env_llm_defaults(),
        llm_provider_options=LLM_PROVIDER_OPTIONS,
        objective_count=len(load_structured_objectives()),
        presets=PRESETS,
        workshop_attack_slugs=set(PRESETS["workshop"]["attack_slugs"]),
        extra_attacks_unlocked=extra_attacks_unlocked(),
        max_prompts=current_app.config["MAX_PROMPTS_PER_ATTACK"],
        env_converter_backup=CONVERTER_BACKUP,
        env_converter_on_block=CONVERTER_ON_BLOCK,
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
        has_artifacts=has_artifacts,
        # Current fine-grained phase, for the first server render (the poll keeps
        # it live thereafter). None unless a run thread is active for this id.
        activity=get_activity(b.id),
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
    counts = Prompt.counting_filter()
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
    risk_live = round(100.0 * succ_live / total_live, 1) if total_live else 0.0

    # Per-attack progress
    per_attack_rows = (
        db.session.query(Prompt.attack_slug, func.count(Prompt.id), func.sum(Prompt.succeeded))
        .filter_by(benchmark_id=b.id).filter(Prompt.counting_filter())
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
    # preset at 2 prompts/attack must not have its bar drawn against 10 (and a
    # later .env change must not retroactively move the denominator of a run
    # that has already finished).
    cap = prompt_cap(b, current_app.config["MAX_PROMPTS_PER_ATTACK"])
    expected_total = sum(
        expected_rows(a, cap)
        for slug in b.attack_slugs
        if (a := by_slug(slug))
    )

    return jsonify(
        status=b.status, total=total_live, succ=succ_live, risk=risk_live,
        rows=rows_live, excluded=rows_live - total_live,
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
    # numbers were measured and which were not.
    w.writerow(["benchmark", "target", "architecture", "attack", "category", "taxonomy",
                "objective_id", "attempt_group", "attempt", "seq", "prompt", "converter",
                "response", "latency_ms", "latency_measured", "refused", "succeeded",
                "severity", "severity_source", "status", "counts_toward_stats",
                "verdict", "note", "error"])
    for p in b.prompts:
        atk = by_slug(p.attack_slug) or {}
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
                    p.error or ""])
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
    fresh one; it re-enters the attack loop but skips attacks already recorded as
    completed and redoes only the interrupted one, so it picks up where the old
    process died (see engine/runner._run_attacks). started_at is preserved so
    the run's duration still covers the whole run."""
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
        cancel_and_join(b.id)
    elif b.status in ("queued", "running"):
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
