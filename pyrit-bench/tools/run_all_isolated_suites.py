#!/usr/bin/env python3
"""Run every approved offline app suite in parallel and print one summary table.

Run (from the project root):

    .venv/bin/python -B tools/run_all_isolated_suites.py            # all suites, one process each
    .venv/bin/python -B tools/run_all_isolated_suites.py --jobs 4   # limit concurrency
    .venv/bin/python -B tools/run_all_isolated_suites.py --with-resume-check

Each suite runs through tools/run_isolated_app_test.py (disposable storage, no
network). `--with-resume-check` also runs tools/dynamic_resume_check.py (loopback
mock server, temporary storage). Exit status is non-zero if any run fails. Logs
are written to a temporary folder that is printed at the end.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
sys.path.insert(0, str(ROOT / "tools"))
from run_isolated_app_test import SUITES  # noqa: E402

SUMMARY_RE = re.compile(r"(Ran \d+ tests? in [\d.]+s|\d+ passed, \d+ failed|^OK$|^FAILED.*|\d+/\d+ checks passed)", re.M)


def run_one(name: str, cmd: list[str], log_dir: Path) -> tuple[str, int, str]:
    log = log_dir / f"{name}.log"
    with log.open("w") as handle:
        code = subprocess.call(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    text = log.read_text(errors="replace")
    found = SUMMARY_RE.findall(text)
    summary = " · ".join(dict.fromkeys(m if isinstance(m, str) else m[0] for m in found)) or text.strip().splitlines()[-1:] and text.strip().splitlines()[-1]
    return name, code, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=len(SUITES) + 1)
    ap.add_argument("--with-resume-check", action="store_true")
    a = ap.parse_args(argv)
    log_dir = Path(tempfile.mkdtemp(prefix="fyp-suites-"))
    jobs = [(s, [str(PY), "-B", str(ROOT / "tools" / "run_isolated_app_test.py"), s]) for s in SUITES]
    if a.with_resume_check:
        jobs.append(("dynamic_resume_check", [str(PY), "-B", str(ROOT / "tools" / "dynamic_resume_check.py")]))
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        results = list(pool.map(lambda j: run_one(j[0], j[1], log_dir), jobs))
    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, code, summary in results:
        status = "PASS" if code == 0 else "FAIL"
        failed += code != 0
        print(f"{status}  {name.ljust(width)}  {summary}")
    print(f"\n{len(results) - failed} of {len(results)} runs passed; logs in {log_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
