"""Target CRUD, connection tests, and the browser capture helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path

from app.web.redact import mask_key_like
from flask import (
    current_app, flash, jsonify, redirect, render_template, request, url_for
)
from sqlalchemy.exc import IntegrityError
from ..extensions import db
from types import SimpleNamespace
from ..models import Benchmark, Target
from uuid import uuid4
from ..targets import capture_login_session, capture_ui_selectors, providers_for_form
from functools import wraps
from ..engine import interactive
from ..engine import test_target

from .blueprint import bp
from .error_hints import parse_error_hint
from .security import csrf_protect


# The target form asks for waits in SECONDS, because that is the unit a person
# reasons about; everything below the form speaks milliseconds. Convert here, in
# the one funnel both the save route and the connection test already go through,
# and store only the _ms key. Nothing else in the app learns that the form ever
# used seconds, and targets saved before this change keep working untouched.
SECONDS_FIELD_TO_MS_KEY = {
    "ui_page_load_s":    "ui_timeout_ms",
    "ui_llm_load_s":     "ui_initial_settle_ms",
    "ui_response_min_s": "ui_stream_settle_ms",
    "ui_response_max_s": "ui_response_max_ms",
}


MS_KEY_TO_SECONDS_FIELD = {ms: s for s, ms in SECONDS_FIELD_TO_MS_KEY.items()}


def _target_admission(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        with interactive.admission_lock():
            return view(*args, **kwargs)
    return wrapped


def rules_of_engagement() -> str:
    """The shared rules sentence from app/data/gdp_checklist_notes.json ('' if unavailable)."""
    try:
        notes = json.loads((Path(current_app.root_path) / "data" / "gdp_checklist_notes.json")
                           .read_text(encoding="utf-8"))
        return str(notes.get("rules_of_engagement") or "")
    except (OSError, ValueError, AttributeError):
        return ""


def seconds_to_ms(value: str) -> str | None:
    """'2.5' seconds -> '2500' ms. None when the value isn't a number."""
    try:
        return str(int(round(float(value) * 1000)))
    except (TypeError, ValueError):
        return None


def parse_kv(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        ms_key = SECONDS_FIELD_TO_MS_KEY.get(k)
        if ms_key:
            ms = seconds_to_ms(v)
            if ms is None:
                # Blank or junk — drop the key so the target falls back to its
                # default instead of crashing on int("") at send time.
                continue
            k, v = ms_key, ms
        out[k] = v
    return out


def derive_model_name(provider: str, extra: dict) -> str:
    """Label a target when no model field was submitted.

    Raw-HTTP targets carry the model inside the JSON request body, so read it
    from there; fall back to the provider slug so the (NOT NULL) column is set.
    """
    bt = extra.get("body_template")
    if bt:
        try:
            m = json.loads(bt).get("model")
            if m:
                return str(m)
        except (ValueError, AttributeError):
            pass
    return provider or "model"


def build_test_payload(t) -> dict:
    """Run a single-ping test and attach a friendly hint + trimmed response.

    Reuses parse_error_hint() so a failed connection test surfaces the same
    actionable guidance the benchmark detail page shows on a failed run.
    """
    res = test_target(t)
    if res.get("error"):
        # Never echo a key-shaped string from a provider error back to the page.
        res["error"] = mask_key_like(str(res["error"]))
        hint = parse_error_hint(res["error"])
        if hint:
            res["hint"] = hint
    if res.get("response"):
        res["response"] = res["response"][:1000]
    if res.get("debug"):
        # Cap each collected string so the JSON payload stays sane — this is
        # for a human to eyeball what response_sel matched, not a data dump.
        d = res["debug"]
        res["debug"] = {
            "response_selector": d.get("response_selector", ""),
            "baseline": [s[:500] for s in d.get("baseline", [])],
            "attempts": [
                {
                    "collected": [s[:500] for s in a.get("collected", [])],
                    "added": [s[:500] for s in a.get("added", [])],
                }
                for a in d.get("attempts", [])
            ],
            # Carried through explicitly — this whitelist rebuild used to drop
            # it, which is why the diagnosis could only reach the UI by being
            # glued onto the error message.
            "vision_diagnosis": d.get("vision_diagnosis"),
            # Shows "po" -> "pong": what was first seen vs what was finally
            # accepted after the streaming settle.
            "stream_settle": d.get("stream_settle"),
            "stream_note": d.get("stream_note"),
            "diagnosis_state": d.get("diagnosis_state"),
            "recovered_reply": d.get("recovered_reply"),
            "recovery_note": d.get("recovery_note"),
            "recovery_rejected": d.get("recovery_rejected"),
        }
    return res


# ---------- Targets ------------------------------------------------------

def _render_targets(prefill: dict | None = None, status: int = 200):
    """The Targets page; `prefill` refills the Add-a-target form (see the JS)."""
    rows = Target.query.order_by(Target.created_at.desc()).all()
    providers_meta = providers_for_form()
    in_docker = bool(current_app.config.get("IN_DOCKER"))
    return render_template(
        "targets.html", targets=rows,
        providers_meta=providers_meta,
        providers_json=json.dumps(providers_meta),
        edit=None, prefill=prefill, rules=rules_of_engagement(),
        in_docker=in_docker,
        page_url_hint="http://host.docker.internal:8501" if in_docker else "http://localhost:8501",
    ), status


def _prefill_from_form(form) -> dict:
    """Everything the Add-a-target form posted, so an error never empties it (finding #57).

    The API key is not sent back into the page; `had_key` asks for it again.
    """
    ui = form.get("provider") == "ui_browser" or form.get("conn_type") == "ui"
    if ui:
        extra = {k[len("extra_config_"):]: v for k, v in form.items() if k.startswith("extra_config_ui_")}
    else:
        extra = {k.strip(): v.strip() for k, v in (line.split("=", 1) for line in
                 form.get("extra_config", "").splitlines() if "=" in line)}
    return {"name": form.get("name", ""), "conn_type": "ui" if ui else "api",
            "provider": form.get("provider", ""), "endpoint_url": form.get("endpoint_url", ""),
            "extra": extra, "architecture": form.get("architecture", ""), "notes": form.get("notes", ""),
            "had_key": bool(form.get("api_key", "").strip())}


def _prefill_from_target(t) -> dict:
    """A saved target as Add-a-target form values, for Duplicate and edit."""
    extra = {}
    for key, value in (t.extra or {}).items():
        if key in MS_KEY_TO_SECONDS_FIELD:
            try:
                key, value = MS_KEY_TO_SECONDS_FIELD[key], f"{int(value) / 1000:g}"
            except (TypeError, ValueError):
                continue
        extra[key] = value if isinstance(value, str) else json.dumps(value)
    return {"name": f"{t.name} (copy)", "conn_type": "ui" if t.provider == "ui_browser" else "api",
            "provider": t.provider, "endpoint_url": t.endpoint_url or "", "extra": extra,
            "architecture": t.architecture or "", "notes": t.notes or "",
            "had_key": bool(t.api_key)}


@bp.route("/targets")
def targets_list():
    return _render_targets()


@bp.route("/targets/<int:tid>/duplicate")
def targets_duplicate(tid: int):
    """Duplicate and edit: the Add-a-target form prefilled from a saved target.

    Saved targets are never changed (their runs refer to them), so fixing a
    wrong URL or selector means saving a corrected copy (finding #57).
    """
    t = Target.query.get_or_404(tid)
    flash(f"Copy of '{t.name}': change what was wrong, keep or change the name, then click "
          "Create target. The original target and its runs are kept.", "success")
    return _render_targets(_prefill_from_target(t))


@bp.route("/targets", methods=["POST"])
def targets_create():
    err = csrf_protect()
    if err:
        return err
    name = request.form["name"].strip()
    # Target.name is UNIQUE — check up front so a re-used name gets a clear,
    # actionable message. The form comes back with everything that was posted,
    # including captured selectors, instead of a blank form (finding #57).
    if Target.query.filter_by(name=name).first():
        flash(f"A target named '{name}' already exists. Choose a different name; your other "
              "entries are kept below.", "error")
        return _render_targets(_prefill_from_form(request.form), 400)
    provider = request.form["provider"]
    extra = parse_kv(request.form.get("extra_config", ""))
    model_name = request.form.get("model_name", "").strip() or derive_model_name(provider, extra)
    t = Target(
        name=name,
        provider=provider,
        model_name=model_name,
        endpoint_url=request.form.get("endpoint_url", "").strip() or None,
        api_key_env_var=request.form.get("api_key_env_var", "").strip() or None,
        api_key=request.form.get("api_key", "").strip() or None,
        architecture=request.form.get("architecture", "").strip() or None,
        notes=request.form.get("notes", "").strip() or None,
        extra_json=json.dumps(extra),
    )
    db.session.add(t)
    try:
        db.session.commit()
        flash(f"Target '{t.name}' created.", "success")
    except IntegrityError:
        db.session.rollback()
        flash(f"A target named '{name}' already exists. Choose a different name; your other "
              "entries are kept below.", "error")
        return _render_targets(_prefill_from_form(request.form), 400)
    except Exception as e:  # noqa: BLE001
        db.session.rollback()
        flash(f"Error: {e}", "error")
    return redirect(url_for("main.targets_list"))


@bp.route("/targets/test", methods=["POST"])
@_target_admission
def targets_test_config():
    """Test an *unsaved* target configuration posted from the Add-Target form.

    Builds an in-memory target from the submitted fields (never persisted) and
    sends one ping prompt, so the user can verify a config before creating it.
    """
    err = csrf_protect()
    if err:
        return err
    if interactive.target_busy(None):
        return jsonify(ok=False, error="End the active conversation before opening another target operation."), 409
    provider = request.form.get("provider", "").strip()
    if not provider:
        return jsonify(ok=False, error="Pick a connection type / provider first."), 400
    extra = parse_kv(request.form.get("extra_config", ""))
    fake = SimpleNamespace(
        provider=provider,
        model_name=request.form.get("model_name", "").strip(),
        endpoint_url=request.form.get("endpoint_url", "").strip() or None,
        api_key_env_var=request.form.get("api_key_env_var", "").strip() or None,
        api_key=request.form.get("api_key", "").strip() or None,
        extra=extra,
        extra_json=json.dumps(extra),
    )
    return jsonify(**build_test_payload(fake))


@bp.route("/targets/capture", methods=["POST"])
@_target_admission
def targets_capture():
    """Open a real browser so the user can click-to-capture UI selectors.

    Launches a headed Chromium (on the machine running this local app), shows an
    overlay, and blocks until the user picks the input / send / reply elements.
    Returns the three selectors so the Add-Target form can fill them in — no
    DevTools or hand-written CSS required.
    """
    err = csrf_protect()
    if err:
        return err
    if interactive.target_busy(None):
        return jsonify(ok=False, error="End the active conversation before opening another target operation."), 409
    url = request.form.get("url", "").strip()
    if not url:
        return jsonify(ok=False, error="Enter the chat page URL first."), 400
    storage_state = request.form.get("storage_state", "").strip() or None
    try:
        result = capture_ui_selectors(url, storage_state=storage_state)
    except Exception as e:  # noqa: BLE001 — never 500 the form; report inline
        return jsonify(ok=False, error=f"{type(e).__name__}: {e}")
    return jsonify(**result)


@bp.route("/targets/login", methods=["POST"])
@_target_admission
def targets_login():
    """Open a real browser so the user can log in to an auth-gated UI target once.

    Saves the resulting browser session (cookies + localStorage) to a file under
    the app's instance folder and returns its path, which the Add-Target form
    stores on the target so capture, tests and benchmark runs stay signed in.
    """
    err = csrf_protect()
    if err:
        return err
    if interactive.target_busy(None):
        return jsonify(ok=False, error="End the active conversation before opening another target operation."), 409
    url = request.form.get("url", "").strip()
    if not url:
        return jsonify(ok=False, error="Enter the chat page URL first."), 400
    storage_path = request.form.get("storage_state", "").strip()
    if not storage_path:
        sess_dir = os.path.join(current_app.instance_path, "ui_sessions")
        os.makedirs(sess_dir, exist_ok=True)
        storage_path = os.path.join(sess_dir, f"session-{uuid4().hex}.json")
    try:
        result = capture_login_session(url, storage_path)
    except Exception as e:  # noqa: BLE001 — never 500 the form; report inline
        return jsonify(ok=False, error=f"{type(e).__name__}: {e}")
    return jsonify(**result)


@bp.route("/targets/<int:tid>/test", methods=["POST"])
@_target_admission
def targets_test_saved(tid: int):
    """Send one ping prompt to a saved target and report the result."""
    err = csrf_protect()
    if err:
        return err
    t = Target.query.get_or_404(tid)
    if interactive.target_busy(tid):
        return jsonify(ok=False, error="End the active conversation before testing this target."), 409
    return jsonify(**build_test_payload(t))


@bp.route("/targets/<int:tid>/delete", methods=["POST"])
def targets_delete(tid: int):
    err = csrf_protect()
    if err:
        return err
    if interactive.target_busy(tid):
        return jsonify(error="End the active conversation before deleting this target."), 409
    t = Target.query.get_or_404(tid)
    # Check for linked benchmarks before deleting — SQLite doesn't enforce FK
    # constraints by default, so we guard in Python to avoid orphaned rows.
    linked = Benchmark.query.filter_by(target_id=tid).count()
    if linked:
        flash(
            f"Cannot delete '{t.name}' — it has {linked} benchmark(s). "
            "Delete those benchmarks first.",
            "error",
        )
        return redirect(url_for("main.targets_list"))
    db.session.delete(t)
    db.session.commit()
    flash(f"Target '{t.name}' deleted.", "success")
    return redirect(url_for("main.targets_list"))
