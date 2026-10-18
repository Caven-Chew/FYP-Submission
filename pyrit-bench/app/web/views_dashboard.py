"""Dashboard view."""
from __future__ import annotations

from flask import render_template
from sqlalchemy import desc, func
from ..extensions import db
from ..models import Benchmark, Prompt, Target

from .blueprint import bp


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
    # Captured responses and valid judgements are different coverage measures.
    # An unscored response cannot be treated as an assessed negative outcome.
    captured = (db.session.query(func.count(Prompt.id))
                .filter(Prompt.counting_filter()).scalar() or 0)
    scored = Prompt.scored_filter()
    total = db.session.query(func.count(Prompt.id)).filter(scored).scalar() or 0
    succ = (db.session.query(func.count(Prompt.id))
            .filter(scored, Prompt.succeeded.is_(True)).scalar() or 0)
    recent = Benchmark.query.order_by(desc(Benchmark.created_at)).limit(8).all()
    return render_template(
        "dashboard.html",
        targets=targets, benchmarks=benchmarks,
        captured_prompts=captured, scored_prompts=total,
        unscored_prompts=captured - total, total_successes=succ,
        success_rate=round(100.0 * succ / total, 1) if total else None,
        recent=recent,
    )
