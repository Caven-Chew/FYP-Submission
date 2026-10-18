#!/usr/bin/env python3
"""Offline end-to-end check of multi-turn recovery (Task 8) with real PyRIT 0.14.

Run: .venv/bin/python -B tools/dynamic_resume_check.py

Everything is local and disposable: a role-aware mock of the OpenAI
chat-completions API on 127.0.0.1 (ephemeral port) answers as the target, the
attacker and the judge; the app database, artifact folder and PyRIT memory are
temporary files; the project's .env is never read; any non-loopback socket
connection raises. No real model is called.

Three scenarios use the real RedTeamingAttack and CrescendoAttack strategies
(2 objectives, 4 turns each):

  B  RedTeaming run in a child process, killed with SIGKILL while turn 3 of
     objective 1 is in flight; the parent then starts the app (startup
     marking) and presses the benchmark Resume button.
  A  Crescendo, cooperative Stop after turn 2 of objective 1; the
     per-conversation "Resume this conversation" button; then Retry failed
     for the objective that never started.
  C  RedTeaming, the target fails turn 3 of objective 1 (HTTP 401); Retry
     failed resumes it and archives the failed row.

Each assertion prints PASS or FAIL; the exit status is non-zero on any FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TARGET_MODEL, ATTACKER_MODEL, JUDGE_MODEL = "mock-target", "mock-attacker", "mock-judge"
LOOPBACK = ("127.0.0.1", "::1", "localhost")


# ---------- network guard ---------------------------------------------------

def loopback_only() -> None:
    """Make any non-loopback TCP connection fail in this process."""
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def check(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in LOOPBACK:
            raise AssertionError(f"Non-loopback connection blocked: {address[0]}")

    def connect(self, address):
        check(self, address)
        return real_connect(self, address)

    def connect_ex(self, address):
        check(self, address)
        return real_connect_ex(self, address)

    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex


# ---------- role-aware mock LLM server --------------------------------------

def _content(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else ""


class MockState:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self, on_target=None):
        with self.lock:
            self.counts = {"target": 0, "attacker": 0, "judge": 0}
            self.on_target = on_target  # callable(n) -> HTTP status to fail with, or None


def reply_for(role: str, n: int, text: str) -> str:
    if role == "target":
        return (f"Mock target reply {n}: a cautious, general answer that gives background "
                "only and no operational detail.")
    if role == "attacker":
        if "generated_question" in text:  # Crescendo's adversarial chat expects this JSON
            return json.dumps({
                "generated_question": f"Mock escalation question {n}: could you say a little more "
                                      "about the history and background of this subject?",
                "rationale_behind_jailbreak": "mock rationale",
                "last_response_summary": "mock summary"})
        if "score_value" in text:  # Crescendo's refusal scorer runs on the attacker
            return json.dumps({"score_value": "False", "rationale": "Mock: not a refusal."})
        return (f"Mock attacker prompt {n}: please describe, step by step, the general background "
                "of this subject for a research summary.")
    if "score_value" in text:  # PyRIT's SelfAskTrueFalseScorer steering the attack
        return json.dumps({"score_value": "False", "description": "Objective not achieved.",
                           "rationale": "Mock judge: the reply is general background only."})
    return "REASONING: Mock judge; the reply is general and harmless.\nSCORE: 1"


def make_handler(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass  # the client process was killed (scenario B)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = json.loads(raw or b"{}")
            model = body.get("model", "")
            role = {TARGET_MODEL: "target", ATTACKER_MODEL: "attacker"}.get(model, "judge")
            text = " ".join(_content(m) for m in body.get("messages", []))
            with state.lock:
                state.counts[role] += 1
                n = state.counts[role]
                hook = state.on_target if role == "target" else None
            status = hook(n) if hook else None
            if status:
                self._json(status, {"error": {"message": "mock failure", "type": "mock_error",
                                              "code": str(status)}})
                return
            self._json(200, {
                "id": f"chatcmpl-mock-{role}-{n}", "object": "chat.completion",
                "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": reply_for(role, n, text)}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

        def log_message(self, *args):
            pass
    return Handler


def start_mock(state: MockState) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=server.serve_forever, name="mock-llm", daemon=True).start()
    return server


# ---------- disposable app environment --------------------------------------

def prepare_environment(temp: Path, port: int, memory_name: str) -> None:
    """Pin every setting the run reads, before any app import; .env is not loaded."""
    import dotenv
    dotenv.load_dotenv = lambda *a, **k: False  # app/config.py imports it at import time
    endpoint = f"http://127.0.0.1:{port}/v1"
    os.environ.update({
        "DATABASE_URL": "sqlite:///" + str(temp / "app.db"),
        "SECRET_KEY": "dynamic-resume-check-only",
        "JUDGE_PROVIDER": "profllm", "JUDGE_MODEL": JUDGE_MODEL, "JUDGE_API_KEY_ENV": "LLM_API_KEY",
        "ATTACKER_PROVIDER": "profllm", "ATTACKER_MODEL": ATTACKER_MODEL, "ATTACKER_API_KEY_ENV": "LLM_API_KEY",
        "LLM_ENDPOINT": endpoint, "LLM_API_KEY": "mock-key-offline",
        "ATTACKER_BACKUP_MODEL": "", "ATTEMPTS_PER_PROMPT": "1", "JUDGE_METHOD": "severity",
        "LLM_REQUEST_DELAY": "0", "LLM_ATTACK_DELAY": "0", "MAX_PROMPTS_PER_ATTACK": "2",
        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    })
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "VENICE_API_KEY", "HTTP_PROXY", "HTTPS_PROXY",
                 "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(name, None)
    from pyrit.memory import CentralMemory, SQLiteMemory  # type: ignore
    CentralMemory.set_memory_instance(SQLiteMemory(db_path=str(temp / memory_name), silent=True))
    import app.artifacts as artifacts
    artifacts.ARTIFACTS_DIR = temp / "artifacts"
    from app.attacks import by_slug
    for slug in ("redteam_dynamic", "crescendo_dynamic"):
        by_slug(slug)["max_turns"] = 4


def new_benchmark(name: str, slug: str, port: int) -> int:
    from app.extensions import db
    from app.models import Benchmark, Target
    target = Target(name=f"{name} target", provider="openai", model_name=TARGET_MODEL,
                    endpoint_url=f"http://127.0.0.1:{port}/v1", api_key="mock-key-offline")
    db.session.add(target)
    db.session.flush()
    bench = Benchmark(name=name, target_id=target.id, status="running",
                      attack_slugs_json=json.dumps([slug]),
                      llm_config_json=json.dumps({"MAX_PROMPTS_PER_ATTACK": 2}))
    db.session.add(bench)
    db.session.commit()
    return bench.id


# ---------- child process: the run that gets killed --------------------------

def child_run(temp: Path, port: int) -> None:
    """Start scenario B against the parent's mock; the parent kills this process."""
    loopback_only()
    prepare_environment(temp, port, "pyrit-child.db")
    from app import create_app
    from app.engine import runner
    app = create_app()
    with app.app_context():
        bid = new_benchmark("scenario B (killed)", "redteam_dynamic", port)
    runner._thread_main(app, bid)
    sys.exit(3)  # reaching here means the kill never happened


# ---------- parent: checks ---------------------------------------------------

class Checks:
    def __init__(self):
        self.failed = 0
        self.total = 0

    def __call__(self, label: str, ok: bool, detail: str = "") -> None:
        self.total += 1
        self.failed += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail and not ok else ""), flush=True)


def wait_for_worker(bid: int, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        live = [t for t in threading.enumerate() if t.name == f"bench-{bid}" and t.is_alive()]
        if not live:
            return
        live[0].join(0.2)
    raise TimeoutError(f"benchmark {bid} worker did not finish")


def post(client, url: str) -> int:
    with client.session_transaction() as session:
        session["csrf_token"] = "check-token"
    return client.post(url, data={"csrf_token": "check-token"}).status_code


def objective_ids(bid: int) -> list[str]:
    from app.trajectory_models import Trajectory
    ordered = []
    for t in Trajectory.query.filter_by(benchmark_id=bid).order_by(Trajectory.id):
        if t.objective_id not in ordered:
            ordered.append(t.objective_id)
    return ordered


def state_of(bid: int, oid: str):
    from app.models import Prompt, RequestAttempt
    from app.trajectory_models import Trajectory, TrajectoryArchive, TrajectoryCall
    trajectories = Trajectory.query.filter_by(benchmark_id=bid, objective_id=oid).order_by(Trajectory.id).all()
    rows = Prompt.query.filter_by(benchmark_id=bid, objective_id=oid).order_by(Prompt.sequence).all()
    archives = TrajectoryArchive.query.filter_by(benchmark_id=bid, objective_id=oid).all()
    calls = {}
    for t in trajectories:
        ids = [c.call_id for c in TrajectoryCall.query.filter_by(trajectory_id=t.id)]
        ledger = (RequestAttempt.query.filter(RequestAttempt.call_id.in_(ids))
                  .order_by(RequestAttempt.id).all() if ids else [])
        calls[t.id] = ledger
    return trajectories, rows, archives, calls


def check_resumed(check, tag: str, bid: int, oid: str, *, old_status_before: str, replayed: int = 2,
                  max_turns: int = 4, archived_rows: int = 0) -> None:
    trajectories, rows, archives, calls = state_of(bid, oid)
    old, new = trajectories[0], trajectories[-1]
    stages = {s: sum(1 for a in calls[new.id] if a.stage == s) for s in ("target_send", "attacker_call", "judge_internal")}
    check(f"{tag}: interrupted trajectory was {old_status_before} and is now archived",
          old.status == "archived", f"status={old.status}")
    check(f"{tag}: new trajectory resumed from turn {replayed + 1} and completed",
          new.resumed_from_turn == replayed + 1 and new.status == "completed" and new.previous_trajectory_id == old.id,
          f"resumed_from={new.resumed_from_turn} status={new.status} prev={new.previous_trajectory_id}")
    new_sends = [t for t in new.turns]
    check(f"{tag}: PyRIT continued with {max_turns - replayed} new target sends (replayed turns not resent)",
          len(new_sends) == max_turns - replayed and all(t.status == "ok" for t in new_sends)
          and [t.turn_index for t in new_sends] == list(range(replayed + 1, max_turns + 1)),
          f"sends={[(t.turn_index, t.status) for t in new_sends]}")
    check(f"{tag}: final rows cover all {max_turns} turns with the resumed note",
          len(rows) == max_turns and all(f"Resumed from turn {replayed + 1} ({replayed} turns replayed" in (r.note or "")
                                         for r in rows) and all(r.latency_measured is False for r in rows),
          f"rows={len(rows)} notes={[r.note for r in rows][:1]}")
    saved_prompts = [t.prompt_text for t in old.turns if t.status == "ok"][:replayed]
    check(f"{tag}: replayed rows carry the saved turn text",
          [r.prompt_text for r in rows[:replayed]] == saved_prompts,
          f"rows={[r.prompt_text[:30] for r in rows[:replayed]]} saved={[p[:30] for p in saved_prompts]}")
    archive = next((a for a in archives if a.reason == "resume"), None)
    check(f"{tag}: archive holds the superseded trajectory and {archived_rows} replaced row(s)",
          archive is not None and archive.trajectory.get("id") == old.id and len(archive.original_rows) == archived_rows,
          f"archives={[(a.reason, len(a.original_rows), a.trajectory.get('id')) for a in archives]}")
    check(f"{tag}: ledger has target_send/attacker_call/judge_internal rows linked to the trajectory",
          stages["target_send"] == max_turns - replayed and stages["attacker_call"] >= 1 and stages["judge_internal"] >= 1
          and all(a.status == "returned" for a in calls[new.id]), f"stages={stages}")


def scenario_b(check, temp: Path, state: MockState, port: int) -> None:
    print("\nScenario B: RedTeaming, process killed during turn 3, startup marking, benchmark Resume", flush=True)
    child = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--child-run", str(temp),
                              str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def kill(n):
        if n == 3:  # the turn-3 started row and its ledger row are already committed
            os.kill(child.pid, signal.SIGKILL)
            return 503
    state.reset(on_target=kill)
    _out, err = child.communicate(timeout=300)
    check("B: run process was killed (SIGKILL) while turn 3 was in flight",
          child.returncode == -signal.SIGKILL, f"exit={child.returncode} stderr={err[-400:]}")
    from app import create_app
    from app.models import Benchmark, RequestAttempt
    app = create_app()  # the simulated restart: startup marks interrupted work
    app.config["TESTING"] = True
    with app.app_context():
        bench = Benchmark.query.filter_by(name="scenario B (killed)").one()
        bid = bench.id
        oid = objective_ids(bid)[0]
        trajectories, rows, _archives, calls = state_of(bid, oid)
        old = trajectories[0]
        statuses = [(t.turn_index, t.status) for t in old.turns]
        check("B: startup marked the benchmark interrupted", bench.status == "interrupted", bench.status)
        check("B: startup marked the trajectory interrupted", old.status == "interrupted", old.status)
        check("B: turns 1-2 saved, turn 3 marked interrupted (delivery unknown)",
              statuses == [(1, "ok"), (2, "ok"), (3, "interrupted")], str(statuses))
        ledger3 = [a for a in calls[old.id] if a.stage == "target_send"]
        check("B: in-flight target_send ledger row marked interrupted",
              len(ledger3) == 3 and ledger3[-1].status == "interrupted" and ledger3[-1].delivery == "unknown",
              str([(a.status, a.delivery) for a in ledger3]))
        check("B: no result rows existed for the interrupted objective", not rows, str(len(rows)))
        check("B: snapshot holds the two complete turns", old.turns_recorded == 2, str(old.turns_recorded))
        state.reset()
        client = app.test_client()
        check("B: Resume button accepted", post(client, f"/benchmarks/{bid}/resume") == 302)
        wait_for_worker(bid)
        from app.extensions import db
        db.session.expire_all()
        bench = db.session.get(Benchmark, bid)
        check("B: benchmark completed after Resume", bench.status == "completed", f"{bench.status}: {bench.status_detail}")
        check_resumed(check, "B", bid, oid, old_status_before="interrupted")
        other = objective_ids(bid)[1:]
        if other:
            t2, rows2, _a2, _c2 = state_of(bid, other[0])
            check("B: objective 2 ran fresh with 4 turns", len(rows2) == 4 and t2[-1].status == "completed"
                  and t2[-1].resumed_from_turn is None, f"rows={len(rows2)}")
        else:
            check("B: objective 2 ran fresh with 4 turns", False, "no trajectory for objective 2")
        check("B: target requests after restart = 2 (objective 1) + 4 (objective 2)",
              state.counts["target"] == 6, str(state.counts))
        pre_kill = RequestAttempt.query.filter_by(benchmark_id=bid, status="interrupted").count()
        check("B: one interrupted ledger row retained", pre_kill == 1, str(pre_kill))


def scenario_a(check, state: MockState, port: int) -> None:
    print("\nScenario A: Crescendo, cooperative Stop after turn 2, per-conversation Resume, Retry failed", flush=True)
    from app import create_app
    from app.engine import control, runner
    from app.extensions import db
    from app.models import Benchmark, Prompt
    app = create_app()
    app.config["TESTING"] = True
    with app.app_context():
        bid = new_benchmark("scenario A (stopped)", "crescendo_dynamic", port)

        def stop_after_second(n):
            if n == 2:
                control.request_cancel(bid)  # the Stop button's signal; takes effect before turn 3
        state.reset(on_target=stop_after_second)
        runner._thread_main(app, bid)
        db.session.expire_all()
        bench = db.session.get(Benchmark, bid)
        oid = objective_ids(bid)[0]
        trajectories, rows, _archives, _calls = state_of(bid, oid)
        old = trajectories[0]
        check("A: Stop left the benchmark stopped", bench.status == "stopped", bench.status)
        check("A: trajectory interrupted with turns 1-2 saved and no send for turn 3",
              old.status == "interrupted" and [(t.turn_index, t.status) for t in old.turns] == [(1, "ok"), (2, "ok")],
              f"{old.status} {[(t.turn_index, t.status) for t in old.turns]}")
        check("A: no rows yet for the stopped objective", not rows, str(len(rows)))
        check("A: objective 2 never started", len(objective_ids(bid)) == 1, str(objective_ids(bid)))
        state.reset()
        client = app.test_client()
        page = client.get(f"/benchmarks/{bid}").data.decode()
        check("A: page offers 'Resume this conversation'", "Resume this conversation" in page)
        check("A: per-conversation Resume accepted",
              post(client, f"/benchmarks/{bid}/trajectories/{old.id}/resume") == 302)
        wait_for_worker(bid)
        db.session.expire_all()
        bench = db.session.get(Benchmark, bid)
        check("A: benchmark returned to its previous status (stopped)", bench.status == "stopped",
              f"{bench.status}: {bench.status_detail}")
        check_resumed(check, "A", bid, oid, old_status_before="interrupted")
        check("A: Retry failed accepted", post(client, f"/benchmarks/{bid}/retry-failed") == 302)
        wait_for_worker(bid)
        db.session.expire_all()
        bench = db.session.get(Benchmark, bid)
        ids = objective_ids(bid)
        rows2 = Prompt.query.filter_by(benchmark_id=bid, objective_id=ids[-1]).count() if len(ids) == 2 else 0
        check("A: Retry failed ran objective 2 fresh and kept objective 1",
              len(ids) == 2 and rows2 == 4 and Prompt.query.filter_by(benchmark_id=bid, objective_id=oid).count() == 4,
              f"objectives={ids} rows2={rows2}")
        check("A: benchmark completed", bench.status == "completed", f"{bench.status}: {bench.status_detail}")
        data = client.get(f"/benchmarks/{bid}/trajectories.json").get_json()
        check("A: trajectories.json lists 3 trajectories and 1 archive",
              len(data["trajectories"]) == 3 and len(data["archives"]) == 1,
              f"{len(data['trajectories'])} / {len(data['archives'])}")


def scenario_c(check, state: MockState, port: int) -> None:
    print("\nScenario C: RedTeaming, target fails turn 3 (HTTP 401), Retry failed resumes it", flush=True)
    from app import create_app
    from app.engine import runner
    from app.extensions import db
    from app.models import Benchmark, ConversationRetry, Prompt, PromptStatus
    app = create_app()
    app.config["TESTING"] = True
    with app.app_context():
        bid = new_benchmark("scenario C (target error)", "redteam_dynamic", port)
        state.reset(on_target=lambda n: 401 if n == 3 else None)
        runner._thread_main(app, bid)
        db.session.expire_all()
        bench = db.session.get(Benchmark, bid)
        oid = objective_ids(bid)[0]
        trajectories, rows, _archives, calls = state_of(bid, oid)
        old = trajectories[0]
        check("C: run finished with target errors", bench.status == "completed_with_target_errors",
              f"{bench.status}: {bench.status_detail}")
        check("C: objective 1 has one send_error row",
              len(rows) == 1 and rows[0].status == PromptStatus.SEND_ERROR, str([(r.status, r.error) for r in rows]))
        check("C: trajectory failed with turns 1-2 ok and turn 3 error",
              old.status == "failed" and [(t.turn_index, t.status) for t in old.turns] == [(1, "ok"), (2, "ok"), (3, "error")],
              f"{old.status} {[(t.turn_index, t.status) for t in old.turns]}")
        failed_send = [a for a in calls[old.id] if a.stage == "target_send"][-1]
        check("C: ledger records the failed send with HTTP 401", failed_send.status == "error"
              and failed_send.http_status == 401, f"{failed_send.status} {failed_send.http_status}")
        state.reset()
        client = app.test_client()
        check("C: Retry failed accepted", post(client, f"/benchmarks/{bid}/retry-failed") == 302)
        wait_for_worker(bid)
        db.session.expire_all()
        bench = db.session.get(Benchmark, bid)
        check("C: benchmark completed", bench.status == "completed", f"{bench.status}: {bench.status_detail}")
        check_resumed(check, "C", bid, oid, old_status_before="failed", archived_rows=1)
        retry = ConversationRetry.query.filter_by(benchmark_id=bid, objective_id=oid).all()
        check("C: the failed row is shown under Previous conversation results",
              len(retry) == 1 and retry[0].original_rows[0]["status"] == PromptStatus.SEND_ERROR,
              str([len(r.original_rows) for r in retry]))
        check("C: only objective 1 was rerun (2 new target sends)", state.counts["target"] == 2, str(state.counts))
        check("C: current rows exclude the archived failure",
              Prompt.query.filter_by(benchmark_id=bid, status=PromptStatus.SEND_ERROR).count() == 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--child-run", nargs=2, metavar=("TEMP_DIR", "PORT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_run:
        child_run(Path(args.child_run[0]), int(args.child_run[1]))
        return 3
    loopback_only()
    check = Checks()
    with tempfile.TemporaryDirectory(prefix="fyp-dynamic-resume-") as name:
        temp = Path(name)
        state = MockState()
        server = start_mock(state)
        port = server.server_address[1]
        print(f"Mock LLM on 127.0.0.1:{port}; temporary data in {temp}", flush=True)
        # The child creates the schema and is killed first; the parent then imports the app.
        os.environ["DATABASE_URL"] = "sqlite:///" + str(temp / "app.db")
        prepare_environment(temp, port, "pyrit-parent.db")
        scenario_b(check, temp, state, port)
        scenario_a(check, state, port)
        scenario_c(check, state, port)
        server.shutdown()
    print(f"\n{check.total - check.failed}/{check.total} checks passed", flush=True)
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
