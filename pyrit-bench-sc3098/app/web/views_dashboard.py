"""Dashboard view."""
from __future__ import annotations

from flask import flash, jsonify, redirect, render_template, url_for
from sqlalchemy import desc, func
from ..extensions import db
from ..models import Benchmark, Prompt, Target

from ..settings import JUDGE_GUIDE, SETTINGS_SHORT
from .blueprint import bp
from .security import csrf_protect


# The Add-Target form is driven by the provider registry in targets/registry.py:
# providers_for_form() returns {"api": [...], "ui": [...]} so the template can
# render the two top-level connection types (API / UI) with per-provider
# click-to-fill suggestions and (for custom HTTP) payload presets. Adding a
# provider is now a one-class change in targets/ — the form follows.


# ---------- Dashboard ----------------------------------------------------

@bp.route("/")
def dashboard():
    targets = Target.query.count()
    benchmarks = Benchmark.query.count()
    # Rows that actually reached a target. Counting every row ever written put
    # converter refusals, failed sends and attacks that never started into the
    # denominator of the headline success rate on the front page.
    counts = Prompt.counting_filter() & (Prompt.attack_slug != "custom-interactive")
    total = db.session.query(func.count(Prompt.id)).filter(counts).scalar() or 0
    succ = (db.session.query(func.count(Prompt.id))
            .filter(counts, Prompt.succeeded.is_(True)).scalar() or 0)
    recent = Benchmark.query.order_by(desc(Benchmark.created_at)).limit(8).all()
    return render_template(
        "dashboard.html",
        targets=targets, benchmarks=benchmarks,
        total_prompts=total, total_successes=succ,
        success_rate=round(100.0 * succ / total, 1) if total else 0.0,
        recent=recent, judge=_judge_summary(),
    )


def _judge_summary() -> dict:
    """What the dashboard says about the judge. Reads settings only; sends nothing."""
    from ..engine.llms import _ROLE_DEFAULT_MODEL, judge_setup_problem, role_setting
    provider = (role_setting("JUDGE", "PROVIDER") or "").strip().lower()
    return {"provider": provider,
            "model": role_setting("JUDGE", "MODEL") or _ROLE_DEFAULT_MODEL.get(provider, ""),
            "problem": judge_setup_problem()}


# What to check after a failed Test judge, per judge option in README.md, Configuration and troubleshooting.
_JUDGE_TEST_ADVICE = {
    "profllm": ("Check that you are on the campus network or NTU VPN and that LLM_ENDPOINT and "
                f"LLM_API_KEY in {SETTINGS_SHORT} are exactly as given"),
}


def _judge_failure_text(res: dict) -> str:
    """The student-facing reason a judge check failed, with what to check next."""
    if res["setup"]:
        return f"nothing was sent. {res['error']}"
    advice = _JUDGE_TEST_ADVICE.get(res["provider"], f"Check the judge lines in {SETTINGS_SHORT}")
    error = (res["error"] or "").rstrip()
    error += "" if error.endswith((".", "!", "?", "…")) else "."
    return f"{error} {advice} ({JUDGE_GUIDE}), restart the app and click Test judge again."


@bp.route("/judge/status.json")
def judge_status():
    """The dashboard's automatic check on load: one short fixed request to the judge
    (llms.check_judge). The page shows "Judge is live", or the reason and the
    Test judge button when the judge did not answer."""
    from ..engine.llms import check_judge
    res = check_judge()
    return jsonify({"ok": res["ok"], "provider": res["provider"], "model": res["model"],
                    "message": "" if res["ok"] else _judge_failure_text(res)})


@bp.route("/judge/test", methods=["POST"])
def judge_test():
    """Dashboard "Test judge": one short fixed request to the configured judge.

    Never more than one call (llms.check_judge); the result is flashed with the
    provider and model, and a failure shows a trimmed, key-masked message.
    """
    err = csrf_protect()
    if err:
        return err
    from ..engine.llms import check_judge
    res = check_judge()
    label = f"{res['provider'] or 'no provider'}, model {res['model'] or 'not set'}"
    if res["ok"]:
        flash(f"Test judge: the judge answered ({label}). You can start a run.", "success")
    elif res["setup"]:
        flash(f"Test judge: {_judge_failure_text(res)}", "error")
    else:
        flash(f"Test judge: the judge did not answer ({label}): {_judge_failure_text(res)}", "error")
    return redirect(url_for("main.dashboard"))
