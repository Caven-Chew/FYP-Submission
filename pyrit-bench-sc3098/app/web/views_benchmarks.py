"""Benchmark list / create / detail / status / export / delete views."""
from __future__ import annotations

import csv
import io
import json
import re
from uuid import uuid4
from functools import wraps
from flask import (
    Response, abort, current_app, flash, jsonify, redirect, render_template,
    request, send_file, url_for
)
from sqlalchemy import func
from sqlalchemy.orm import selectinload
from ..engine import interactive
from ..activity import get_activity
from ..artifacts import artifact_file, delete_benchmark_artifacts, has_artifacts
from ..attacks import ATTACKS, by_slug, by_tier
from ..extensions import db
from ..models import Benchmark, Prompt, Target
from ..engine import (
    SUCCESS_SEVERITY, cancel_and_join, request_pause, request_resume, start_benchmark
)
from ..settings import (
    COURSE_PROMPT_CAP, expected_rows, prompt_cap,
)

from .blueprint import bp
from .charts import reach_chart
from .error_hints import parse_error_hint
from .report_pdf import render_benchmark_pdf
from .rollups import benchmark_rollups, taxonomy_rollup
from .security import csrf_protect
from .evidence import (differences, is_student_run, judge_label, judge_note, judge_record, marker_note,
                       mask_keys, ordered_pair, paired_cases, run_problems, summary, target_name, target_note,
                       verdict_class)
from ..engine.control import request_cancel
from ..engine.runner import finish_stopped, start_prompt_retry, start_benchmark_recovery

def _batch_admission(view):
    @wraps(view)
    def wrapped(bid, *args, **kwargs):
        with interactive.admission_lock():
            b = Benchmark.query.get_or_404(bid)
            if not interactive.is_interactive(b) and interactive.target_busy(b.target_id):
                abort(409, description="End the active conversation before running another request against this target.")
            return view(bid, *args, **kwargs)
    return wrapped


MAX_CUSTOM_PROMPT_FILE_BYTES = 1_000_000


def _custom_text_from_request() -> str:
    pasted = (request.form.get("custom_prompts") or "").strip()
    uploaded = request.files.get("custom_prompt_file")
    if pasted and uploaded and uploaded.filename:
        raise ValueError("Use either pasted prompts or a .txt file, not both.")
    if uploaded and uploaded.filename:
        if not uploaded.filename.lower().endswith(".txt"):
            raise ValueError("Custom prompt uploads must be .txt files.")
        raw = uploaded.read(MAX_CUSTOM_PROMPT_FILE_BYTES + 1)
        if len(raw) > MAX_CUSTOM_PROMPT_FILE_BYTES:
            raise ValueError("The custom prompt file is too large (maximum 1 MB).")
        try:
            pasted = raw.decode("utf-8-sig").strip()
        except UnicodeDecodeError:
            raise ValueError("The custom prompt file must use UTF-8 text encoding.") from None
    if len(pasted.encode("utf-8")) > MAX_CUSTOM_PROMPT_FILE_BYTES:
        raise ValueError("The pasted prompts are too large (maximum 1 MB).")
    return pasted


def _custom_prompts_from_request() -> tuple[list[str], str | None]:
    """Legacy list parser, retained for integrations."""
    try:
        return [s.strip() for s in _custom_text_from_request().splitlines() if s.strip()], None
    except ValueError as exc:
        return [], str(exc)


#: Separator lines for scripted conversations. Turns: three or more hyphens, or
#: one to three en/em dashes (Word's AutoFormat turns "---" into a dash).
#: Conversations: three or more "=". Spaces, tabs and non-breaking spaces around
#: the marks are allowed. Mirrored by the detector in benchmark_new.html.
TURN_SEPARATOR = re.compile(r"(?m)^[^\S\n]*(?:-{3,}|[–—]{1,3})[^\S\n]*$")
CONVERSATION_SEPARATOR = re.compile(r"(?m)^[^\S\n]*={3,}[^\S\n]*$")


def parse_student_input(text: str, mode: str) -> tuple[list[str], list[list[str]]]:
    """Read pasted or uploaded text as one prompt, a list, or scripted conversations.

    Conversations: turns are separated by a line containing only ``---`` (or an
    en/em dash) and conversations by a line containing only ``===``. Empty
    blocks (for example a trailing ``===``) are ignored, and Windows line
    endings are accepted. In list mode a separator-looking line is rejected
    rather than sent as a prompt.
    """
    if mode not in {"single", "list", "conversation"}:
        raise ValueError("Choose Scripted conversation or Multi-turn conversation.")
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise ValueError("Enter at least one prompt or upload a UTF-8 .txt file.")
    if mode == "single":
        return [text.strip()], []
    if mode == "list":
        lines = text.split("\n")
        marked = [n for n, line in enumerate(lines, 1)
                  if TURN_SEPARATOR.fullmatch(line) or CONVERSATION_SEPARATOR.fullmatch(line)]
        if marked:
            raise ValueError(
                f"Line(s) {', '.join(map(str, marked[:5]))} look like conversation separators "
                "(---, === or a dash on its own line); choose Scripted conversation, or "
                "remove those lines to send each remaining line as a separate prompt.")
        prompts = [line.strip() for line in text.splitlines() if line.strip()]
        if len(prompts) > 100:
            raise ValueError(f"Use at most 100 prompts per list (you entered {len(prompts)}).")
        return prompts, []
    blocks = [b for b in CONVERSATION_SEPARATOR.split(text) if b.strip()]
    if not blocks:
        raise ValueError("Enter at least one prompt, not just conversation separators.")
    conversations = []
    for number, block in enumerate(blocks, 1):
        turns = [t.strip() for t in TURN_SEPARATOR.split(block)]
        turns = [t for t in turns if t]
        if not 1 <= len(turns) <= 20:
            raise ValueError(
                f"Conversation {number} has {len(turns)} turn(s); each needs 1–20 turns separated "
                "by a line containing only --- (an en or em dash on its own line also works). "
                "A single prompt is also a complete script.")
        conversations.append(turns)
    if len(conversations) > 20:
        raise ValueError(f"Use at most 20 conversations (you entered {len(conversations)}), "
                         "separated by a line containing only ===.")
    return [], conversations


# ---------- Benchmarks ---------------------------------------------------

@bp.route("/benchmarks")
def benchmarks_list():
    rows = (Benchmark.query.options(selectinload(Benchmark.prompts), selectinload(Benchmark.target))
            .order_by(Benchmark.created_at.desc()).all())
    # Plain verdict counts per student run (finding #61).
    counts = {b.id: summary(b) for b in rows if is_student_run(b)}
    return render_template("benchmarks.html", benchmarks=rows, counts=counts)


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
            if part.isdecimal() and len(part) <= 18:  # SQLite cannot bind an integer above 2**63
                ids.append(int(part))
    # de-dupe, preserve order
    seen: set[int] = set()
    ids = [i for i in ids if not (i in seen or seen.add(i))]

    benches = [db.session.get(Benchmark, i) for i in ids]
    benches = [b for b in benches if b is not None]

    if any(interactive.is_interactive(b) for b in benches):
        flash("Interactive turns depend on earlier replies. Review their transcripts individually; batch comparison does not apply.", "warning")
        return redirect(url_for("main.benchmarks_list"))
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

    # Students' before/after view: the older run is "before" unless they swap.
    pair = None
    if len(benches) == 2 and all(is_student_run(b) for b in benches):
        before, after = ordered_pair(*benches, swap=request.args.get("order") == "swap")
        pair = {"before": before, "after": after, "diffs": differences(before, after),
                "marker_note": marker_note(before, after), "cases": paired_cases(before, after),
                "target_note": target_note(before, after),
                "judge_note": judge_note(before, after),
                "judges": (judge_label(before), judge_label(after)),
                "targets": (target_name(before), target_name(after)),
                "problems": {"before": run_problems(before), "after": run_problems(after)},
                "ids": ",".join(str(b.id) for b in benches),
                "swapped": request.args.get("order") == "swap"}
    else:
        flash("Before/after needs exactly two runs of your own prompts. This page shows research "
              "metrics only; tick exactly two such runs on Benchmarks to compare them side by side.", "warning")
    return render_template(
        "benchmarks_compare.html", benches=benches, compare_chart=chart,
        tax_rows=tax_rows, pair=pair, verdict_class=verdict_class, mask=mask_keys,
    )


RUBRIC_HELP = {
    "harm": "General harmfulness of the reply (for the public harmful-request datasets).",
    "leak": "How much of the hidden instructions or secret the reply reveals.",
}


def export_cell(value):
    # Spreadsheet imports must treat untrusted target output as data, not formulae.
    # Excel cells hold at most 32,767 characters, so longer evidence is cut with a note.
    value = str('' if value is None else value)
    if len(value) > 32000:
        value = value[:32000] + ' [cut: full text remains in the saved run]'
    return "'" + value if value.lstrip().startswith(('=', '+', '-', '@')) else value


def rubric_help() -> dict:
    from ..engine.scoring import RUBRICS, STUDENT_FAILURES
    return {key: RUBRIC_HELP.get(key) or STUDENT_FAILURES.get(key, "") for key in RUBRICS}


def show_public_baselines() -> bool:
    """Course build hides the public harmful-request datasets unless enabled (finding #48)."""
    flag = current_app.config.get("SHOW_PUBLIC_BASELINES", False)
    return flag is True or str(flag).strip().lower() in {"1", "true", "yes", "on"}


@bp.app_template_filter("attack_description")
def attack_description(text):
    if current_app.config.get("JUDGE_METHOD") != "dual_v1_v2":
        return text
    text = re.sub(r" Scored on the (?:harm|leak) rubric[^.]*\.", "", text)
    return text.replace("The harm score does not measure over-refusal.",
        "v1 measures harm; v2 also checks refusal against the target's permitted scope.")


@bp.app_template_filter("judge_version_results")
def judge_version_results_filter(verdict):
    from ..engine.scoring import judge_version_results
    return judge_version_results(verdict)


@bp.app_template_filter("assessment_label")
def assessment_label(rubric):
    from ..engine.scoring import OWN_PROMPT_ASSESSMENTS
    if rubric == "checklist_v2":
        return "Five safety criteria (v2; highest score)"
    return OWN_PROMPT_ASSESSMENTS.get(rubric, (str(rubric).replace("_", " ").title(), False))[0]


def _launch_assessment():
    """Snapshot the configured v2 policy; launch forms cannot replace it."""
    from ..engine.scoring import own_prompt_assessment
    method = current_app.config.get("JUDGE_METHOD", "sc3098")
    if method in ("checklist_v2", "dual_v1_v2"):
        return own_prompt_assessment("checklist_v2", current_app.config["V2_TARGET_RULES"], method=method)
    return own_prompt_assessment(request.form.get("assessment"), request.form.get("assessment_rules"), method=method)


def _render_new(targets, status=200, full=False):
    from ..engine import load_structured_objectives
    from ..engine.scoring import OWN_PROMPT_ASSESSMENTS
    posted = request.method == "POST"
    tiers = by_tier(full)
    return render_template(
        "benchmark_new.html", targets=targets, tiers=tiers,
        own_assessments=OWN_PROMPT_ASSESSMENTS,
        total_attacks=sum(len(attacks) for _, _, attacks in tiers),
        objective_count=len(load_structured_objectives()),
        show_baselines=show_public_baselines(),
        form=request.form if posted else {},
        start_id=request.form.get("start_id") or str(uuid4()),
        selected_attacks=set(request.form.getlist("attack_slugs")) if posted else set(),
        max_prompts=current_app.config["MAX_PROMPTS_PER_ATTACK"]), status


@bp.route("/benchmarks/new", methods=["GET", "POST"])
def benchmarks_new():
    return _benchmarks_new(full=False)


# Every attack of the FYP app (user request, 4 Oct 2026). Deliberately unlinked and left
# out of the student documents: the teaching team gives the address when it is needed.
@bp.route("/full-benchmarks/new", methods=["GET", "POST"])
def full_benchmarks_new():
    return _benchmarks_new(full=True)


def _benchmarks_new(full: bool):
    with interactive.admission_lock():
        return _benchmarks_new_locked(full)


def _benchmarks_new_locked(full: bool):
    targets = Target.query.order_by(Target.name).all()
    if not targets:
        flash("Add a target first: the chatbot you want to test.", "warning")
        return redirect(url_for("main.targets_list"))
    if request.method != "POST":
        return _render_new(targets, full=full)

    err = csrf_protect()
    if err:
        return err
    from ..engine.llms import judge_setup_problem
    from ..targets.registry import build_conversation_target
    # Only the open tab of the form is launched; a form without the field (an old
    # bookmark or a script) still combines both, as before.
    tab = request.form.get("tab", "")
    try:
        target = db.session.get(Target, int(request.form.get("target_id", "") or 0))
        if target is None:
            raise ValueError("Choose the target (your chatbot) to test.")
        # Public baseline datasets: optional, never pre-ticked (user decision, 1 Oct 2026).
        offered = {a["slug"] for _, _, attacks in by_tier(full) for a in attacks}
        slugs = [] if tab == "own" else [slug for slug in request.form.getlist("attack_slugs")
                                         if slug in offered]
        if slugs and not show_public_baselines():
            # Hidden is off, not just out of sight (finding #48): a crafted form cannot queue them.
            raise ValueError("Public baseline attacks are switched off in the course version.")
        if tab == "own" and request.form.get("custom_mode") == "interactive":
            if problem := judge_setup_problem():
                raise ValueError(problem)
            cfg = {**_launch_assessment(), **judge_record()}
            b = interactive.start_session(current_app._get_current_object(), target.id,
                request.form.get("name", "").strip() or f"{target.name} — conversation",
                cfg, request.form.get("start_id", ""))
            return redirect(url_for("main.conversation_page", bid=b.id))
        if interactive.target_busy(target.id):
            raise ValueError("End the active conversation before starting another run against this target.")
        text = "" if tab == "public" else _custom_text_from_request()
        custom_prompts, conversations = [], []
        llm_config: dict = {}
        if current_app.config.get("JUDGE_METHOD") in ("checklist_v2", "dual_v1_v2"):
            llm_config.update(_launch_assessment())
        if text:
            mode = request.form.get("custom_mode", "")
            if mode not in {"single", "list", "conversation"}:
                raise ValueError("Choose Scripted conversation or Multi-turn conversation.")
            custom_prompts, conversations = parse_student_input(text, mode)
            # Save the selected assessment and written policy for scoring and re-scoring.
            # The score is a recommendation for human review.
            llm_config.update(_launch_assessment())
            if any(len(turns) > 1 for turns in conversations):
                build_conversation_target(target)  # validate history only for multi-turn scripts
            slugs.append("custom_conversation" if conversations else "custom_prompt_list")
        if not slugs:
            raise ValueError({"public": "Tick at least one attack.",
                              "own": "Paste your prompts or upload a .txt file."}.get(
                                  tab, "Add your own prompts, or tick at least one public baseline attack."))
        raw_cap = request.form.get("prompts_per_attack", "").strip()
        if raw_cap:
            cap = int(raw_cap)
            limit = 100 if show_public_baselines() else COURSE_PROMPT_CAP
            if not 1 <= cap <= limit:
                raise ValueError(f"Prompts per public dataset must be between 1 and {limit}.")
            llm_config["MAX_PROMPTS_PER_ATTACK"] = cap
        if problem := judge_setup_problem():
            raise ValueError(problem)
        if conversations:
            llm_config["SCRIPTED_SCORE_UNIT"] = "turn"
        llm_config.update(judge_record())  # finding #23: the judge and threshold used, saved with the run
    except (ValueError, TypeError, RuntimeError) as exc:
        flash(str(exc) or "Invalid run settings.", "error")
        return _render_new(targets, status=400, full=full)
    b = Benchmark(
        name=request.form.get("name", "").strip() or f"{target.name} — own prompts",
        target_id=target.id,
        attack_slugs_json=json.dumps(slugs),
        custom_prompts_json=json.dumps(custom_prompts),
        custom_conversations_json=json.dumps(conversations),
        llm_config_json=json.dumps(llm_config),
        status="queued",
    )
    db.session.add(b)
    db.session.commit()
    start_benchmark(current_app._get_current_object(), b.id)
    flash(f"Run '{b.name}' queued.", "success")
    return redirect(url_for("main.benchmarks_detail", bid=b.id))


@bp.route("/benchmarks/<int:bid>")
def benchmarks_detail(bid: int):
    b = Benchmark.query.get_or_404(bid)
    if interactive.is_interactive(b):
        return redirect(url_for("main.conversation_page", bid=bid))
    roll = benchmark_rollups(b)
    error_hint = parse_error_hint(b.status_detail) if b.status_detail else None
    curve = b.reach_curve
    per_attack = b.reach_curve_by_attack
    chart_series = [("All attacks", curve)]
    # Only worth splitting out per attack when more than one was run.
    if len(per_attack) > 1:
        chart_series += [((by_slug(k) or {}).get("name", k), v)
                         for k, v in per_attack.items()]
    # The FYP run-page layout (user decision, 3 Oct 2026); `student` only words the Resume tooltip.
    return render_template(
        "benchmark_detail.html", b=b, by_slug=by_slug, error_hint=error_hint,
        student=is_student_run(b), judge_label=judge_label(b), mask=mask_keys, marker=b.llm_config.get("CANARY"),
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
    if interactive.is_interactive(b):
        from .views_interactive import _display
        return jsonify(_display(b))
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
    # preset at 3 prompts/attack must not have its bar drawn against 10 (and a
    # later .env change must not retroactively move the denominator of a run
    # that has already finished).
    cap = prompt_cap(b, current_app.config["MAX_PROMPTS_PER_ATTACK"])
    expected_total = sum(
        expected_rows(a, cap, b)
        for slug in b.attack_slugs
        if (a := by_slug(slug))
    )

    # As in the FYP run page: captured = a reply came back; scored = it also has a
    # valid score (the counted rows above); unscored = captured but not yet scored.
    captured_live = (db.session.query(func.count(Prompt.id))
                     .filter(Prompt.benchmark_id == b.id, func.length(func.trim(func.coalesce(Prompt.response, ""))) > 0)
                     .scalar() or 0)
    return jsonify(
        # Use the same escaped, masked tables as the initial page. Rows are saved
        # throughout a run; they must appear before the terminal page reload.
        prompt_results_html=render_template(
            "_benchmark_prompt_results.html", b=b, by_slug=by_slug,
            display_groups=roll["display_groups"], mask=mask_keys,
            marker=b.llm_config.get("CANARY"),
        ),
        status=b.status, total=total_live, succ=succ_live, risk=risk_live,
        rows=rows_live, excluded=rows_live - total_live,
        captured=captured_live, scored=total_live, unscored=max(0, captured_live - total_live),
        rows_total=rows_live, run_time=b.run_time_label,
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
    if interactive.is_interactive(b):
        abort(400, description="Attempt-budget curves do not apply to interactive conversations.")
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
    if interactive.is_interactive(b):
        from .views_interactive import conversation_csv
        return conversation_csv(b)
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
                "verdict", "note", "error", "rubric", "canary_detected", "transcript_json",
                "failure_criterion", "reference_facts", "assessment_rules", "v1_score", "v1_verdict", "v2_score", "v2_verdict", "prompt_role"])
    # Key-shaped strings in model text are masked here too, keeping the planted marker (finding #37).
    keep = b.llm_config.get("CANARY")
    from ..engine.scoring import judge_version_results
    for p in b.prompts:
        versions = judge_version_results(p.verdict)
        atk = by_slug(p.attack_slug) or {}
        w.writerow([export_cell(value) for value in [b.name, b.target.name, b.target.architecture or "",
                    p.attack_slug, atk.get("category", ""),
                    p.taxonomy_category or "", p.objective_id or "",
                    p.attempt_group or "", p.attempt or 0,
                    p.sequence, p.prompt_text or "", p.converter_used or "",
                    mask_keys(p.response, keep), p.latency_ms or "",
                    "" if p.latency_measured is None else int(bool(p.latency_measured)),
                    int(bool(p.refused)), int(bool(p.succeeded)),
                    p.severity if p.severity is not None else "",
                    p.severity_source or "", p.status or "",
                    int(bool(p.counts_toward_stats)),
                    mask_keys(p.verdict, keep), p.note or "",
                    # Full error text, untruncated — Python tracebacks put the
                    # actual exception last, so a prefix cut loses the answer. Key-masked like the run page.
                    mask_keys(p.error, keep), p.rubric or "",
                    "" if p.canary_detected is None else str(p.canary_detected),
                    mask_keys(p.transcript_json, keep), b.llm_config.get("FAILURE_CRITERION", ""),
                    b.llm_config.get("REFERENCE_FACTS", ""), mask_keys(b.llm_config.get("ASSESSMENT_RULES", ""), keep),
                    versions.get("v1", {}).get("score"), mask_keys(versions.get("v1", {}).get("verdict", ""), keep),
                    versions.get("v2", {}).get("score"), mask_keys(versions.get("v2", {}).get("verdict", ""), keep),
                    "converter_input_not_sent" if p.converter_not_sent else "target_prompt"]])
    return Response(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=benchmark-{b.id}.csv"},
    )


@bp.route("/benchmarks/<int:bid>/pdf")
def benchmarks_pdf(bid: int):
    """Generate a PDF summary report for the benchmark."""
    b = Benchmark.query.get_or_404(bid)
    try:
        if interactive.is_interactive(b):
            from .views_interactive import conversation_pdf
            pdf_bytes = conversation_pdf(b)
        else:
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
    if interactive.is_interactive(b):
        abort(400, description="Use the conversation page. Interactive target messages cannot be replayed.")
    if b.status == "running":
        request_pause(b.id)
        b.status = "paused"
        db.session.commit()
    else:
        flash(f"'{b.name}' is not running, so it can't be paused.", "error")
    return redirect(url_for("main.benchmarks_detail", bid=b.id))


@bp.route("/benchmarks/<int:bid>/resume", methods=["POST"])
@_batch_admission
def benchmarks_resume(bid: int):
    """Resume a paused or interrupted benchmark.

    "paused": the run thread is still alive in this process — clear the
    in-memory flag and the loop continues where it was.

    "interrupted": the process restarted, so the run thread is gone. Launch a
    recovery worker that keeps every completed test case and fills in only the
    unattempted ones (engine/runner._recover_async with retry_failed=False).
    A fresh _run_attacks would delete the unfinished attack's rows first, and a
    student run has only one attack, so every judged case would be re-sent
    (review finding #2). The run is claimed with the same atomic UPDATE as the
    retry routes, so a double-submit cannot start two workers. started_at is
    preserved so the run's duration still covers the whole run. A research
    run (no student attack) keeps the original full restart: its Azure scans
    cannot resume inside an attempted scan, and recovery would skip them."""
    err = csrf_protect()
    if err:
        return err
    b = Benchmark.query.get_or_404(bid)
    if interactive.is_interactive(b):
        abort(400, description="Use the conversation page. Interactive target messages cannot be replayed.")
    if b.status == "paused":
        request_resume(b.id)
        b.status = "running"
        db.session.commit()
    elif b.status == "interrupted":
        claimed = Benchmark.query.filter(
            Benchmark.id == bid, Benchmark.status == "interrupted",
        ).update({Benchmark.status: "running"}, synchronize_session=False)
        db.session.commit()
        if not claimed:
            flash("This run is already being resumed.", "warning")
            return redirect(url_for("main.benchmarks_detail", bid=bid))
        # Dual static/scripted runs also have durable captured replies. Recover
        # their missing cases without replacing replies whose judge was lost.
        keep_captures = is_student_run(b) or (
            b.llm_config.get('JUDGE_METHOD') == 'dual_v1_v2' and
            all((by_slug(slug) or {}).get('kind') in {'static', 'scripted'} for slug in b.attack_slugs))
        try:
            # current_app is a proxy; the thread needs the real app object.
            if keep_captures:
                start_benchmark_recovery(current_app._get_current_object(), bid, retry_failed=False)
            else:
                start_benchmark(current_app._get_current_object(), bid)
        except Exception:
            b.status = "interrupted"
            db.session.commit()
            raise
        flash("Continuing unfinished test cases; completed results are kept." if keep_captures
              else "Resumed from the next unfinished attack.", "success")
    else:
        flash(f"'{b.name}' is not paused or interrupted.", "error")
    return redirect(url_for("main.benchmarks_detail", bid=b.id))


@bp.route("/benchmarks/<int:bid>/stop", methods=["POST"])
def benchmarks_stop(bid: int):
    err = csrf_protect()
    if err:
        return err
    b = Benchmark.query.get_or_404(bid)
    if interactive.is_interactive(b):
        interactive.end_session(current_app._get_current_object(), bid)
        return redirect(url_for("main.conversation_page", bid=bid))
    if b.status in ("queued", "running", "paused"):
        b.status = "stopping"
        db.session.commit()
        request_cancel(b.id)
        flash("Stopping after the current request or attack objective finishes. Results will be kept.", "success")
    elif b.status == "interrupted":
        finish_stopped(b.id)
    return redirect(url_for("main.benchmarks_detail", bid=b.id))


@bp.route("/benchmarks/<int:bid>/delete", methods=["POST"])
@_batch_admission
def benchmarks_delete(bid: int):
    err = csrf_protect()
    if err:
        return err
    b = Benchmark.query.get_or_404(bid)
    if interactive.is_interactive(b) and (interactive.snapshot(b)["active"] or interactive.worker_busy(bid)):
        abort(409, description="End the conversation and wait for resource cleanup before deleting it.")
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
    elif b.status in ("queued", "running", "stopping"):
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


@bp.route("/benchmarks/<int:bid>/prompts/<int:pid>/rerun", methods=["POST"])
@_batch_admission
def benchmarks_rerun_prompt(bid: int, pid: int):
    err = csrf_protect()
    if err:
        return err
    b = Benchmark.query.get_or_404(bid)
    if interactive.is_interactive(b):
        abort(400, description="Use the conversation page. Interactive target messages cannot be replayed.")
    p = Prompt.query.filter_by(id=pid, benchmark_id=bid).first_or_404()
    if not p.can_rerun:
        abort(400, description="Only failed prompt attempts or conversations with an objective can be rerun.")
    previous = b.status
    label = p.retry_label
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
        flash("Re-scoring the stored reply with the judge; the chatbot is not contacted again."
              if label == "Re-score" else
              "Rerunning the conversation from the beginning." if label == "Rerun conversation"
              else "Rerunning the failed prompt. Its result will update when finished.", "success")
    return redirect(url_for("main.benchmarks_detail", bid=bid))


@bp.route("/benchmarks/<int:bid>/retry-failed", methods=["POST"])
@_batch_admission
def benchmarks_retry_failed(bid: int, skip_failed: bool = False):
    err = csrf_protect()
    if err:
        return err
    b = Benchmark.query.get_or_404(bid)
    if interactive.is_interactive(b):
        abort(400, description="Use the conversation page. Interactive target messages cannot be replayed.")
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
