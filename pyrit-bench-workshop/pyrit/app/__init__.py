"""Flask app factory."""
import secrets
import sqlite3
import warnings

from flask import Flask
from sqlalchemy import event
from sqlalchemy.engine import Engine

from .config import Config
from .expiry import ensure_workshop_active

from .extensions import db

# Placeholder keys shipped in .env / .env.example that must never be used to
# sign real sessions.
_WEAK_KEYS = {
    "dev-only-change-me",
    "change-me",
    "change-me-to-a-random-string",
}


@event.listens_for(Engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, _record):
    """Enable WAL and a busy timeout so the background writer thread and the
    web-request readers don't trip over 'database is locked'."""
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()


def create_app() -> Flask:
    ensure_workshop_active()
    app = Flask(__name__)
    app.config.from_object(Config)

    # Sessions (and the CSRF token stored in them) are signed with SECRET_KEY.
    # If it's a placeholder, generate a random one for this run so the token
    # can't be forged with a known key.
    if not app.config.get("SECRET_KEY") or app.config["SECRET_KEY"] in _WEAK_KEYS:
        app.config["SECRET_KEY"] = secrets.token_hex(32)
        warnings.warn(
            "SECRET_KEY was unset or a known placeholder; using a random key for "
            "this run. Set a fixed SECRET_KEY in .env to keep sessions valid across "
            "restarts.",
            stacklevel=1,
        )

    db.init_app(app)

    from .web import bp
    app.register_blueprint(bp)

    with app.app_context():
        from . import models  # noqa: F401
        db.create_all()
        _migrate_add_columns()
        _mark_interrupted_benchmarks()
        # Sweep failure captures left by older runs (and by benchmarks deleted
        # while their files were in use) so instance/artifacts/ cannot grow
        # without bound between runs. See app/artifacts.py for the policy.
        from .artifacts import prune
        prune()

    return app


def _mark_interrupted_benchmarks() -> None:
    """Mark benchmarks left mid-run by a restart as 'interrupted' (resumable).

    Benchmarks run in daemon threads that don't survive a restart, so any row
    still queued/running/paused at startup lost its thread. Rather than failing
    them outright, mark them 'interrupted': the attacks that finished are already
    recorded (Benchmark.completed_attacks), so a Resume can relaunch the run and
    pick up at the next unfinished attack (see engine/runner._run_attacks).

    'paused' is included on purpose. A paused run's thread is also gone after a
    restart, and the old code only swept queued/running — so a paused benchmark
    used to survive as a zombie: still shown "paused" with nothing behind it, and
    Resume silently cleared an in-memory flag no thread was watching. It is now
    caught here like the others.

    finished_at is deliberately left unset — an interrupted run is not finished.
    """
    from .models import Benchmark

    stale = Benchmark.query.filter(
        Benchmark.status.in_(("queued", "running", "paused"))
    ).all()
    if not stale:
        return
    note = (
        "[startup] The server restarted while this benchmark was still running, "
        "and runner threads do not survive a restart. Marked 'interrupted' — "
        "press Resume to continue from the next unfinished attack."
    )
    for b in stale:
        b.status = "interrupted"
        b.status_detail = f"{b.status_detail}\n{note}" if b.status_detail else note
    db.session.commit()


def _backfill_converter_refusals(conn) -> None:
    """Retro-flag rows where the converter LLM refused and we sent its refusal.

    Runs once, when the `status` column is first added. Until this was caught,
    a converter model declining to rewrite a harmful seed produced a row that
    looked like a clean, judged, unsuccessful attack — evidence that the target
    resisted something never actually aimed at it. Those rows are still on disk
    from every run before the fix, and they drag published success rates down.

    Uses engine.converters.converter_declined so there is exactly ONE definition
    of what a refusal looks like; the historic rows have no record of the
    converter's input, so `before` is omitted and only the wordlist rule runs.
    """
    from sqlalchemy import text
    from .engine.converters import converter_declined

    rows = conn.execute(text(
        "SELECT id, prompt_text FROM prompts "
        "WHERE converter_used IS NOT NULL AND status = 'ok' "
        "  AND trim(coalesce(prompt_text, '')) != ''"
    )).fetchall()
    for row_id, prompt_text in rows:
        reason = converter_declined(prompt_text)
        if not reason:
            continue
        conn.execute(
            text("UPDATE prompts SET status = 'converter_refused', "
                 "    note = COALESCE(note || ' | ', '') || :note "
                 "WHERE id = :id"),
            {"id": row_id, "note": f"backfilled: {reason}"},
        )


def _migrate_add_columns() -> None:
    """Add columns introduced after the initial schema. Idempotent; runs on
    every startup so an existing DB doesn't need to be dropped."""
    from sqlalchemy import inspect, text

    insp = inspect(db.engine)

    prompt_cols = {c["name"] for c in insp.get_columns("prompts")}
    with db.engine.begin() as conn:
        if "severity" not in prompt_cols:
            conn.execute(text("ALTER TABLE prompts ADD COLUMN severity INTEGER"))
        if "taxonomy_category" not in prompt_cols:
            conn.execute(text("ALTER TABLE prompts ADD COLUMN taxonomy_category VARCHAR(60)"))
        if "objective_id" not in prompt_cols:
            conn.execute(text("ALTER TABLE prompts ADD COLUMN objective_id VARCHAR(40)"))
        if "attempt" not in prompt_cols:
            conn.execute(text("ALTER TABLE prompts ADD COLUMN attempt INTEGER DEFAULT 0"))
        if "attempt_group" not in prompt_cols:
            conn.execute(text("ALTER TABLE prompts ADD COLUMN attempt_group VARCHAR(120)"))
        if "capture_ref" not in prompt_cols:
            conn.execute(text("ALTER TABLE prompts ADD COLUMN capture_ref VARCHAR(80)"))
        # Add EVERY new column before running any backfill: the status backfill
        # writes to `note`, so doing this column-by-column-with-its-own-backfill
        # crashes on first boot against an existing DB.
        needs_row_backfill = "status" not in prompt_cols
        if needs_row_backfill:
            conn.execute(text("ALTER TABLE prompts ADD COLUMN status VARCHAR(24)"))
        if "note" not in prompt_cols:
            conn.execute(text("ALTER TABLE prompts ADD COLUMN note TEXT"))
        if "verdict" not in prompt_cols:
            conn.execute(text("ALTER TABLE prompts ADD COLUMN verdict TEXT"))
        # severity_source and latency_measured are deliberately NOT backfilled:
        # for historic rows we genuinely do not know whether a severity came
        # from the judge or from a hardcoded 4, and guessing "judge" would bake
        # the very confusion these columns exist to remove. NULL means unknown.
        if "severity_source" not in prompt_cols:
            conn.execute(text("ALTER TABLE prompts ADD COLUMN severity_source VARCHAR(12)"))
        if "latency_measured" not in prompt_cols:
            conn.execute(text("ALTER TABLE prompts ADD COLUMN latency_measured BOOLEAN"))

        # Now that every column exists, classify the rows that predate them.
        if needs_row_backfill:
            # Read the old rows for what they actually say, and nothing more.
            #
            # The middle branch matters: `error` used to double as a notes
            # column, so a row can carry error text and still be a perfectly
            # good result. A row with a response AND a severity was judged,
            # which means the send worked whatever the error column says — the
            # only such rows are converter fallbacks, and treating those as
            # failed sends would throw away real data. Conversely a response
            # with no severity is the vision model's "[no response captured
            # — ...]" placeholder, which is a failure and must stay one.
            conn.execute(text(
                "UPDATE prompts SET status = CASE "
                "  WHEN error LIKE 'converter-error:%' THEN 'converter_error' "
                "  WHEN error IS NOT NULL AND error != '' "
                "       AND trim(coalesce(response, '')) != '' "
                "       AND severity IS NOT NULL THEN 'ok' "
                "  WHEN error IS NOT NULL AND error != '' THEN 'send_error' "
                "  ELSE 'ok' END"
            ))
            # Those same rows keep their text, just in the column that means
            # "commentary" rather than the one that means "this row failed".
            conn.execute(text(
                "UPDATE prompts SET note = error, error = NULL "
                "WHERE status = 'ok' AND error IS NOT NULL AND error != ''"
            ))
            _backfill_converter_refusals(conn)

        # One-time re-scan for the curly-apostrophe leak. The original refusal
        # matcher only knew straight ASCII apostrophes, so a converter that
        # declined with typographic quotes ("I'm sorry, I can't assist…" with
        # U+2019) slipped through, was sent to the target, and got scored — 499
        # fabricated no-op rows across existing benchmarks. Now that
        # looks_like_refusal normalises the quotes, re-run the same scan once so
        # those historic rows are re-flagged converter_refused and drop out of
        # every rate. Gated on PRAGMA user_version so it runs exactly once and
        # never re-scans on later boots (new rows are handled live).
        if (conn.execute(text("PRAGMA user_version")).scalar() or 0) < 1:
            _backfill_converter_refusals(conn)
            conn.execute(text("PRAGMA user_version = 1"))

    bench_cols = {c["name"] for c in insp.get_columns("benchmarks")}
    with db.engine.begin() as conn:
        if "llm_config_json" not in bench_cols:
            conn.execute(text("ALTER TABLE benchmarks ADD COLUMN llm_config_json TEXT DEFAULT '{}'"))
        if "attacker_roster_json" not in bench_cols:
            conn.execute(text("ALTER TABLE benchmarks ADD COLUMN attacker_roster_json TEXT DEFAULT '[]'"))
        if "completed_attacks_json" not in bench_cols:
            conn.execute(text("ALTER TABLE benchmarks ADD COLUMN completed_attacks_json TEXT DEFAULT '[]'"))

    target_cols = {c["name"] for c in insp.get_columns("targets")}
    with db.engine.begin() as conn:
        if "architecture" not in target_cols:
            conn.execute(text("ALTER TABLE targets ADD COLUMN architecture TEXT"))
        if "api_key" not in target_cols:
            conn.execute(text("ALTER TABLE targets ADD COLUMN api_key TEXT"))
