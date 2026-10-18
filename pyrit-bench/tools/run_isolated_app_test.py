#!/usr/bin/env python3
"""Run one approved offline app suite against disposable SQLite/artifact paths.

Run: .venv/bin/python -B tools/run_isolated_app_test.py test_scoring_recovery.py

The app prunes capture folders on startup. Redirect both storage locations
BEFORE importing the test, and block socket connections. This wrapper does not
change the app's production configuration or the separately deferred test-suite
cleanup work. Live provider tests are intentionally not accepted.
"""
import argparse
import os
from pathlib import Path
import runpy
import socket
import sys
import tempfile
from unittest.mock import patch

SUITES = ("test_request_history.py", "test_static_execution_history.py", "test_scoring_recovery.py", "test_run_model_record.py", "test_prompt_retry.py",
          "test_scoring_changes.py", "test_workshop_preset.py", "test_live_prompt_results.py",
          # Tasks 7 and 8 (October 2026): two-label scoring and multi-turn recovery suites.
          "test_two_label_scoring.py", "test_two_label_web.py",
          "test_conversation_recovery.py", "test_dynamic_resume.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=SUITES)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "test"), str(root)]
    with tempfile.TemporaryDirectory(prefix="fyp-app-test-") as temp:
        os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(temp) / "app.db")
        os.environ["ATTACKER_BACKUP_MODEL"] = ""
        # Tests may replace these with their own fake markers. Never let a
        # machine's real credentials become the defaults of an offline suite.
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "VENICE_API_KEY", "LLM_API_KEY"):
            os.environ[name] = "offline-test-only"
        if args.suite not in ("test_request_history.py", "test_run_model_record.py", "test_scoring_recovery.py", "test_static_execution_history.py",
                              "test_two_label_scoring.py", "test_two_label_web.py", "test_conversation_recovery.py", "test_dynamic_resume.py"):
            import app.artifacts as artifacts
            artifacts.ARTIFACTS_DIR = Path(temp) / "artifacts"
        # These suites install both disposable paths themselves before
        # their first app import. Let them set their fake model environment
        # first too: importing settings early would invalidate model snapshots.
        sys.argv = [str(root / "test" / args.suite)]
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Network disabled in offline app tests")), \
             patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Network disabled in offline app tests")):
            runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
