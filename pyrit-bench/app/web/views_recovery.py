"""Task 8 (October 2026): multi-turn trajectory and recovery routes.

Owned by the Task 8 lane. Registered on the shared blueprint by app/web/__init__.py.
"""
from __future__ import annotations

from flask import abort, current_app, flash, has_request_context, jsonify, redirect, request, url_for
from sqlalchemy.orm import selectinload

from ..engine.runner import start_conversation_resume
from ..engine.trajectory import call_links
from ..extensions import db
from ..models import Benchmark
from ..trajectory_models import Trajectory
from .blueprint import bp
from .security import csrf_protect

#: Benchmark states in which no worker runs, so one conversation may be resumed.
RESUME_STATES = ("completed", "failed", "completed_with_target_errors", "stopped", "interrupted")


def benchmark_trajectories(benchmark_id: int) -> list:
    """All of a benchmark's trajectories with their turns, loaded once per request.

    The results partial (also rendered for status.json) asks per conversation
    card; two queries in total replace one query per card.
    """
    def load():
        return (Trajectory.query.filter_by(benchmark_id=benchmark_id)
                .options(selectinload(Trajectory.turns)).order_by(Trajectory.id).all())
    if not has_request_context():
        return load()
    # Cached on the request, not on g: an app context (and its g) can outlive
    # several requests, and a later request must see new trajectories.
    cache = getattr(request, "_benchmark_trajectories", None)
    if cache is None:
        cache = request._benchmark_trajectories = {}
    if benchmark_id not in cache:
        cache[benchmark_id] = load()
    return cache[benchmark_id]


@bp.app_template_global()
def trajectories_for(benchmark_id: int, attack_slug: str, objective_id: str) -> list:
    return [t for t in benchmark_trajectories(benchmark_id)
            if t.attack_slug == attack_slug and t.objective_id == objective_id]


@bp.app_template_global()
def unfinished_trajectories(benchmark_id: int) -> list:
    return [t for t in benchmark_trajectories(benchmark_id) if t.status in ("interrupted", "failed")]


@bp.app_template_global()
def trajectory_call_links(benchmark_id: int) -> dict:
    return call_links(benchmark_id)


@bp.app_template_global()
def trajectory_resume_states() -> tuple:
    return RESUME_STATES


@bp.route("/benchmarks/<int:bid>/trajectories.json")
def benchmarks_trajectories_json(bid: int):
    """Lossless export of saved multi-turn trajectories, their sends and archives."""
    b = Benchmark.query.get_or_404(bid)
    links = call_links(bid)
    return jsonify(
        scope=("Every multi-turn target send is saved before and after the call; the snapshot is "
               "the latest conversation read back from PyRIT memory. Archived trajectories and rows "
               "are excluded from current results."),
        timestamp_timezone="UTC",
        trajectories=[{**t.as_dict(), "ledger_call_ids": sorted(c for c, (tid, _turn) in links.items() if tid == t.id)}
                      for t in b.trajectories],
        archives=[{"id": a.id, "created_at": a.created_at.isoformat(), "reason": a.reason,
                   "attack_slug": a.attack_slug, "objective_id": a.objective_id, "attempt": a.attempt,
                   "roster_index": a.roster_index, "conversation_retry_id": a.conversation_retry_id,
                   "original_rows": a.original_rows, "trajectory": a.trajectory or None}
                  for a in b.trajectory_archives],
    )


@bp.route("/benchmarks/<int:bid>/trajectories/<int:tid>/resume", methods=["POST"])
def benchmarks_resume_trajectory(bid: int, tid: int):
    """Continue one interrupted conversation from its saved turns."""
    err = csrf_protect()
    if err:
        return err
    b = Benchmark.query.get_or_404(bid)
    t = Trajectory.query.filter_by(id=tid, benchmark_id=bid).first_or_404()
    if t.status != "interrupted":
        abort(400, description="Only an interrupted conversation can be resumed.")
    previous = b.status
    # Claim the benchmark atomically so double-clicks cannot start two workers.
    claimed = Benchmark.query.filter(Benchmark.id == bid, Benchmark.status.in_(RESUME_STATES)).update(
        {Benchmark.status: "running"}, synchronize_session=False)
    db.session.commit()
    if not claimed:
        flash("Wait until this benchmark stops before resuming a conversation.", "warning")
    else:
        try:
            start_conversation_resume(current_app._get_current_object(), bid, tid, previous)
        except Exception:
            Benchmark.query.filter_by(id=bid).update({Benchmark.status: previous}, synchronize_session=False)
            db.session.commit()
            raise
        flash(f"Resuming conversation {t.objective_id} from turn {t.turns_recorded + 1}. Saved turns are "
              "replayed, not sent again (a browser UI target restarts from turn 1 instead).", "success")
    return redirect(url_for("main.benchmarks_detail", bid=bid))
