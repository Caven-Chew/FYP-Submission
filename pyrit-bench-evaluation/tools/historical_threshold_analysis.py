#!/usr/bin/env python3
"""Read-only re-analysis of historical judge scores (RQ3 thresholds, RQ4 attempts).

Why this tool exists
--------------------
The dissertation reports a historical descriptive analysis (Sections 5.1-5.6:
1,471 scored responses collected 11 May - 16 August 2026) whose numbers were
copied from the interim report rather than recomputed. RQ3 (threshold choice)
and RQ4 (attempt budget) both need to be answered from records that already
exist, at zero model cost. This tool re-reads those records and reports what the
stored metadata can and cannot support.

What it does NOT do
-------------------
It makes no model, network or API call, starts no Flask application, imports no
application module, and writes nothing to the database. It runs on the Python
standard library only.

Read-only and metadata-only guarantees
--------------------------------------
* The database is opened through the SQLite URI ``file:<path>?mode=ro`` and the
  connection then sets ``PRAGMA query_only = 1``. Both are asserted at startup.
* Every statement is checked by :func:`assert_read_only_sql` before execution:
  only ``SELECT``, ``WITH ... SELECT`` and ``PRAGMA table_info`` /
  ``PRAGMA query_only`` are allowed.
* Every statement is checked by :func:`assert_metadata_only`, which rejects any
  reference to a free-text column (prompt text, converted prompt, response,
  judge verdict, error text, notes, LLM configuration JSON, endpoint URL, API
  key, and so on). The single exception is the emptiness predicate
  ``length(trim(coalesce(p.response, ''))) > 0``, which is needed to reproduce
  the "delivered" population; it yields a boolean and never retrieves text.
* ``attempt_group`` is treated as an opaque identifier. It is used for grouping
  and counting only; no attempt-group value is ever printed or written out.

Population rules (taken from app/models.py and app/web/rollups.py, read as
source, never imported)
-----------------------------------------------------------------------------
A row is *delivered* when ``status`` is ``ok`` or ``recovered`` and the response
is non-empty (``Prompt.counts_toward_stats``). A delivered row is *scored* when
``severity`` is 1-5 and ``severity_source`` is not ``synthetic``
(``rollups._scored``). The rubric is a property of the attack, not of the row:
``app/attacks.py`` gives ``scoring = "leak"`` for the system-prompt-extraction
family and ``harm`` (the default) for everything else. Harm and leak severities
share a 1-5 range and mean different things, so this tool never pools them.

Subcommands
-----------
``inventory``  Metadata census of the database: tables and columns, row counts
               by status, rubric, benchmark, target and attack slug, date range,
               how many rows are scored, and whether objective ids are populated.
``thresholds`` RQ3. Harm rubric only. Severity counts and positive rates at
               thresholds 2-5 with 95% Wilson intervals, per attack condition,
               per target, and per attack x target where n allows; condition
               rank order at each threshold and an explicit list of pairwise
               order changes between thresholds.
``attempts``   RQ4. Inventory of retained static multi-attempt groups.
``bounds``     Needs no database. Feasible-range solver over the published
               aggregates alone (Table 9 n / count>=3 / verified means, with the
               Table 10 pooled totals as optional caps).

Usage
-----
    .venv/bin/python -B tools/historical_threshold_analysis.py inventory \\
        --db instance/app.db --out reports/final_review/historical_reanalysis_01
    .venv/bin/python -B tools/historical_threshold_analysis.py thresholds \\
        --db instance/app.db --out reports/final_review/historical_reanalysis_01
    .venv/bin/python -B tools/historical_threshold_analysis.py attempts \\
        --db instance/app.db --out reports/final_review/historical_reanalysis_01
    .venv/bin/python -B tools/historical_threshold_analysis.py bounds \\
        --out reports/final_review/historical_reanalysis_01

Each subcommand writes ``<name>.json`` and ``<name>.md`` into the output
directory and refuses to overwrite an existing file. Exit 0 on success, 2 on a
usage, input or refuse-to-overwrite error.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from fractions import Fraction

TOOL_NAME = "historical_threshold_analysis.py"
TOOL_VERSION = "1.0.0"

# --------------------------------------------------------------------------
# Published aggregates. Transcribed from the dissertation text and verified
# against the interim report; see PROVENANCE below each block.
# --------------------------------------------------------------------------

#: Historical collection window stated in Section 5.1 (inclusive of both days).
HISTORICAL_START = "2026-05-11"
HISTORICAL_END = "2026-08-16"

#: Table 10 of reports/final_review/revised_text_for_review.txt: pooled severity
#: counts over the 1,471 historical scored responses.
REPORT_TABLE_10 = {1: 1182, 2: 23, 3: 37, 4: 18, 5: 211}
REPORT_POOLED_TOTAL = 1471

#: Table 11 of the same file: positives at each threshold over 1,471.
REPORT_TABLE_11 = {2: 289, 3: 266, 4: 229, 5: 211}

#: Table 9 of the same file (n and "scores at least 3"), joined to the mean
#: severity column of Table 9 of reports/interim/Interim_Report_PyRIT-Bench_v5.docx.
#: The means were read back out of word/document.xml with the standard library
#: and each one matched the value hard-coded in
#: reports/assets/report_images/generate_figures.py (fig7), including the order
#: in which the conditions are listed.
REPORT_TABLE_9 = [
    # (label, attack_slug, n, count_at_or_above_3, mean_severity_2dp)
    ("Forbidden Questions", "forbidden_questions", 177, 6, "1.14"),
    ("XSTest subset (benign)", "xstest", 202, 22, "1.28"),
    ("AdvBench", "advbench", 225, 19, "1.34"),
    ("HarmBench plain", "harmbench", 275, 25, "1.36"),
    ("HarmBench + leetspeak", "harmbench_leetspeak", 153, 38, "1.94"),
    ("HarmBench + Base64", "harmbench_base64", 138, 43, "2.23"),
    ("HarmBench + ROT13", "harmbench_rot13", 153, 91, "3.37"),
    ("HarmBench + codechameleon", "harmbench_codechameleon", 20, 18, "3.65"),
]

#: Section 5.3: the eight displayed conditions account for 1,343 responses and
#: 262 positives; the remaining 128 responses carry 4 positives.
REPORT_TABLE_9_TOTAL_N = 1343
REPORT_TABLE_9_TOTAL_POS = 262

THRESHOLDS = (2, 3, 4, 5)

#: Rows with fewer than this many scored responses are listed separately and
#: never ranked. 20 is the smallest Table 9 condition, kept as the floor.
DEFAULT_MIN_N = 20

# --------------------------------------------------------------------------
# Safety: read-only and metadata-only enforcement
# --------------------------------------------------------------------------

#: Columns that hold free text, secrets or configuration blobs. None of these
#: may appear in any statement this tool executes. Drawn from app/models.py
#: (Prompt, Benchmark, Target, ConversationRetry).
FREE_TEXT_COLUMNS = frozenset({
    "prompt_text", "response", "verdict", "error", "note", "notes",
    "status_detail", "llm_config_json", "attacker_roster_json",
    "attack_slugs_json", "completed_attacks_json", "extra_json",
    "endpoint_url", "api_key", "api_key_env_var", "architecture",
    "original_rows_json", "capture_ref",
})

#: The one permitted use of a free-text column: an emptiness test that returns a
#: boolean. It reproduces Prompt.counts_toward_stats without reading any text.
DELIVERED_SQL = "length(trim(coalesce(p.response, ''))) > 0"

#: SQL fragments exempted from the free-text check, stripped before scanning.
_ALLOWED_FRAGMENTS = (DELIVERED_SQL,)

_ALLOWED_STATEMENT_RE = re.compile(
    r"^\s*(select\b|with\b|pragma\s+(table_info|query_only)\b)", re.IGNORECASE)

_FORBIDDEN_KEYWORD_RE = re.compile(
    r"\b(insert|update|delete|drop|create|alter|replace|attach|detach|vacuum|"
    r"reindex|begin|commit|rollback|pragma\s+journal_mode|pragma\s+writable)\b",
    re.IGNORECASE)


class UnsafeQuery(ValueError):
    """Raised when a statement would write, or would touch free text."""


def assert_read_only_sql(sql: str) -> None:
    """Reject anything that is not a plain SELECT / WITH-SELECT / safe PRAGMA.

    The database connection is already read-only twice over; this is the third
    guard and the only one that is visible in a code review of this file.
    """
    if ";" in sql.strip().rstrip(";"):
        raise UnsafeQuery("multiple statements are not allowed")
    if not _ALLOWED_STATEMENT_RE.match(sql):
        raise UnsafeQuery(f"statement is not a permitted read: {sql.strip()[:60]!r}")
    hit = _FORBIDDEN_KEYWORD_RE.search(sql)
    if hit:
        raise UnsafeQuery(f"statement contains the write keyword {hit.group(0)!r}")


def assert_metadata_only(sql: str) -> None:
    """Reject any statement that references a free-text or secret column.

    The whitelisted emptiness predicate is removed first, so the only surviving
    mention of a free-text column is a real one.
    """
    scrubbed = sql
    for fragment in _ALLOWED_FRAGMENTS:
        scrubbed = scrubbed.replace(fragment, " 1 ")
    lowered = scrubbed.lower()
    for column in sorted(FREE_TEXT_COLUMNS):
        if re.search(rf"\b{re.escape(column)}\b", lowered):
            raise UnsafeQuery(
                f"statement references the free-text column {column!r}; "
                "this tool exports metadata only")


def open_readonly(db_path: str) -> sqlite3.Connection:
    """Open the application database read-only, and prove that it is read-only.

    Uses the SQLite URI read-only mode, then sets ``query_only`` on the
    connection. Raises RuntimeError if either guarantee cannot be established.
    """
    resolved = os.path.abspath(db_path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"database not found: {resolved}")
    uri = "file:" + resolved.replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.execute("PRAGMA query_only = 1")
    flag = con.execute("PRAGMA query_only").fetchone()
    if not flag or int(flag[0]) != 1:
        con.close()
        raise RuntimeError("could not establish PRAGMA query_only on the connection")
    return con


def query(con: sqlite3.Connection, sql: str, params=()) -> list:
    """Run one guarded read. Every database access in this tool goes through it."""
    assert_read_only_sql(sql)
    assert_metadata_only(sql)
    return con.execute(sql, params).fetchall()


# --------------------------------------------------------------------------
# Attack catalog (rubric and kind), read from source, never imported
# --------------------------------------------------------------------------

#: Fallback used when app/attacks.py cannot be parsed. Slugs that the catalog
#: has never contained (legacy Azure entries removed 2026-08-18, ad-hoc probes)
#: are deliberately absent: an unknown slug is reported, not guessed into the
#: static harm population.
_FALLBACK_CATALOG = {
    "harmbench": ("static", "harm"), "advbench": ("static", "harm"),
    "xstest": ("static", "harm"), "forbidden_questions": ("static", "harm"),
    "harmbench_base64": ("static", "harm"), "harmbench_leetspeak": ("static", "harm"),
    "harmbench_rot13": ("static", "harm"), "harmbench_poetry": ("static", "harm"),
    "harmbench_persuasion": ("static", "harm"), "harmbench_tense": ("static", "harm"),
    "harmbench_tense_poetry": ("static", "harm"), "harmbench_math": ("static", "harm"),
    "harmbench_lowresource": ("static", "harm"),
    "harmbench_persuasion_tense": ("static", "harm"),
    "harmbench_codechameleon": ("static", "harm"),
    "harmbench_variation_bon": ("static", "harm"),
    "harmbench_composition_sweep": ("static", "harm"),
    "harmbench_manyshot_8": ("static", "harm"),
    "harmbench_manyshot_32": ("static", "harm"),
    "harmbench_manyshot_poetry": ("static", "harm"),
    "sysprompt_extraction": ("static", "leak"),
    "sysprompt_extraction_persuasion": ("static", "leak"),
    "sysprompt_extraction_base64": ("static", "leak"),
    "redteam_dynamic": ("dynamic", "harm"), "crescendo_dynamic": ("dynamic", "harm"),
    "tap_dynamic": ("dynamic", "harm"), "pair_dynamic": ("dynamic", "harm"),
}


def load_attack_catalog(project_root: str) -> dict:
    """Return {slug: {kind, rubric, name, catalog_attempts, source}}.

    ``app/attacks.py`` is parsed with :mod:`ast` and its ``ATTACKS`` list is
    literal-evaluated. The module is never imported, so no application code runs
    and nothing touches the database. If the file is missing or has stopped
    being a plain literal, the embedded fallback table is used instead and the
    fact is recorded in ``source``.
    """
    path = os.path.join(project_root, "app", "attacks.py")
    entries = None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) == "ATTACKS" for t in node.targets):
                entries = ast.literal_eval(node.value)
                break
    except (OSError, SyntaxError, ValueError):
        entries = None

    catalog: dict = {}
    if entries:
        for entry in entries:
            slug = entry.get("slug")
            if not slug:
                continue
            catalog[slug] = {
                "kind": entry.get("kind", "unknown"),
                "rubric": entry.get("scoring", "harm"),
                "name": entry.get("name", slug),
                "catalog_attempts": entry.get("attempts"),
                "source": "app/attacks.py",
            }
    else:
        for slug, (kind, rubric) in _FALLBACK_CATALOG.items():
            catalog[slug] = {"kind": kind, "rubric": rubric, "name": slug,
                             "catalog_attempts": None,
                             "source": "embedded fallback table"}
    return catalog


def slugs_for(catalog: dict, kind: str, rubric: str) -> list:
    """Catalog slugs of one kind and one rubric, sorted. Unknown slugs excluded."""
    return sorted(s for s, meta in catalog.items()
                  if meta["kind"] == kind and meta["rubric"] == rubric)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def wilson_interval(successes: int, n: int, z: float = 1.96) -> dict:
    """95% Wilson score interval for a proportion, as percentages.

    Same formula as ``app/models.wilson_interval`` (reproduced here rather than
    imported, because importing app code would pull in Flask and the database
    layer). Returns {rate, low, high, n} in percent, rounded to 1 dp; a
    zero-width interval at 0 when n == 0.
    """
    if n <= 0:
        return {"rate": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return {
        "rate": round(100.0 * p, 1),
        "low": round(100.0 * max(0.0, center - half), 1),
        "high": round(100.0 * min(1.0, center + half), 1),
        "n": n,
    }


def threshold_row(counts: dict, threshold: int) -> dict:
    """Positives at ``threshold`` over a severity histogram {1..5: count}."""
    n = sum(counts.values())
    positives = sum(c for s, c in counts.items() if s >= threshold)
    out = {"threshold": threshold, "positives": positives, "n": n}
    out.update(wilson_interval(positives, n))
    return out


def rate_fraction(counts: dict, threshold: int) -> Fraction:
    """Exact positive rate at ``threshold``; Fraction(0) for an empty group."""
    n = sum(counts.values())
    if n == 0:
        return Fraction(0)
    return Fraction(sum(c for s, c in counts.items() if s >= threshold), n)


def rank_at(groups: dict, threshold: int) -> list:
    """Group keys ordered by descending positive rate, ties broken by key.

    ``groups`` maps a key to a severity histogram. The tie-break on the key
    keeps the order reproducible; genuine ties are visible in the reported rate.
    """
    return sorted(groups, key=lambda k: (-rate_fraction(groups[k], threshold), k))


def order_changes(groups: dict, low_t: int, high_t: int) -> dict:
    """Pairwise order changes between two thresholds.

    Returns ``reversals`` (A above B at ``low_t`` and strictly below at
    ``high_t``, or the reverse), ``new_ties`` and ``broken_ties``. Only strict
    reversals are reported as order changes; a pair that merely becomes tied is
    listed separately, because a tie is not a re-ranking.
    """
    keys = sorted(groups)
    reversals, new_ties, broken_ties = [], [], []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ra_low, rb_low = rate_fraction(groups[a], low_t), rate_fraction(groups[b], low_t)
            ra_high, rb_high = rate_fraction(groups[a], high_t), rate_fraction(groups[b], high_t)
            record = {
                "a": a, "b": b,
                f"a_rate_at_{low_t}": round(float(ra_low) * 100, 2),
                f"b_rate_at_{low_t}": round(float(rb_low) * 100, 2),
                f"a_rate_at_{high_t}": round(float(ra_high) * 100, 2),
                f"b_rate_at_{high_t}": round(float(rb_high) * 100, 2),
            }
            if (ra_low > rb_low and ra_high < rb_high) or (ra_low < rb_low and ra_high > rb_high):
                reversals.append(record)
            elif ra_low != rb_low and ra_high == rb_high:
                new_ties.append(record)
            elif ra_low == rb_low and ra_high != rb_high:
                broken_ties.append(record)
    return {"low_threshold": low_t, "high_threshold": high_t,
            "reversals": reversals, "new_ties": new_ties, "broken_ties": broken_ties}


# --------------------------------------------------------------------------
# Feasible-range solver for the published aggregates (bounds subcommand)
# --------------------------------------------------------------------------

def mean_sum_range(n: int, mean_2dp: str) -> tuple:
    """Integer severity-sum range consistent with a mean correctly rounded to 2 dp.

    A published mean ``m`` means the true mean lies in [m - 0.005, m + 0.005],
    so the severity sum S lies in [n(m - 0.005), n(m + 0.005)]. Exact rational
    arithmetic is used so the endpoints are not disturbed by binary floats.
    """
    m = Fraction(mean_2dp)
    half = Fraction(1, 200)
    low = Fraction(n) * (m - half)
    high = Fraction(n) * (m + half)
    s_min = -((-low.numerator) // low.denominator)          # ceil
    s_max = high.numerator // high.denominator              # floor
    return s_min, s_max


def feasible_counts(n: int, k3: int, mean_2dp: str, caps: dict | None = None) -> dict:
    """Feasible min/max responses at severity >= t for t in 2, 4, 5.

    Known for each Table 9 condition: the number of scored responses ``n``, the
    number scoring at least 3 (``k3``), and the mean severity rounded to 2 dp.
    Unknown: how the counts split inside {1,2} and inside {3,4,5}. Writing
    c1..c5 for those counts,

        c1 + c2 = n - k3,   c3 + c4 + c5 = k3,
        c1 + 2c2 + 3c3 + 4c4 + 5c5 = S  for some S in mean_sum_range(n, mean),

    which rearranges to  c2 + c4 + 2c5 = S - (n - k3) - 3k3. Every solution is
    enumerated exactly (the search space is at most k3^2 per candidate S), so
    the reported ranges are exact, not a relaxation.

    ``caps`` optionally bounds each c_s from above. Passing the Table 10 pooled
    totals encodes the assumption that the Table 9 rows are a subset of the
    1,471 pooled population: a single condition cannot hold more responses at
    severity s than the whole population does. This is applied per condition;
    a joint constraint across all eight conditions at once would be tighter and
    is not attempted here.

    Returns {feasible, sum_range, t2/t4/t5: {min, max}, solutions}.
    """
    caps = caps or {}
    limit = {s: min(caps.get(s, n), n) for s in range(1, 6)}
    lead = n - k3
    s_min, s_max = mean_sum_range(n, mean_2dp)

    best = {2: [None, None], 4: [None, None], 5: [None, None]}
    solutions = 0
    for total in range(s_min, s_max + 1):
        target = total - lead - 3 * k3            # = c2 + c4 + 2*c5
        for c5 in range(0, min(k3, limit[5]) + 1):
            for c4 in range(0, min(k3 - c5, limit[4]) + 1):
                c3 = k3 - c4 - c5
                if c3 > limit[3]:
                    continue
                c2 = target - c4 - 2 * c5
                if c2 < 0 or c2 > min(lead, limit[2]):
                    continue
                c1 = lead - c2
                if c1 < 0 or c1 > limit[1]:
                    continue
                solutions += 1
                for t, value in ((2, k3 + c2), (4, c4 + c5), (5, c5)):
                    lo, hi = best[t]
                    best[t][0] = value if lo is None else min(lo, value)
                    best[t][1] = value if hi is None else max(hi, value)

    feasible = solutions > 0
    out = {
        "n": n, "count_at_least_3": k3, "mean": mean_2dp,
        "sum_range": [s_min, s_max], "feasible": feasible, "solutions": solutions,
    }
    for t in (2, 4, 5):
        lo, hi = best[t]
        out[f"t{t}"] = {"min": lo, "max": hi}
    return out


def determined_orderings(bounds_by_label: dict, threshold: int) -> dict:
    """Which pairwise orderings are fixed by the feasible ranges at ``threshold``.

    A > B is determined when A's worst possible rate still exceeds B's best
    possible rate. Everything else is undetermined by the published aggregates.
    """
    labels = sorted(bounds_by_label)
    determined, undetermined = [], []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            ba, bb = bounds_by_label[a], bounds_by_label[b]
            key = f"t{threshold}"
            a_lo = Fraction(ba[key]["min"], ba["n"])
            a_hi = Fraction(ba[key]["max"], ba["n"])
            b_lo = Fraction(bb[key]["min"], bb["n"])
            b_hi = Fraction(bb[key]["max"], bb["n"])
            if a_lo > b_hi:
                determined.append({"higher": a, "lower": b})
            elif b_lo > a_hi:
                determined.append({"higher": b, "lower": a})
            else:
                undetermined.append({
                    "a": a, "b": b,
                    "a_range_pct": [round(float(a_lo) * 100, 2), round(float(a_hi) * 100, 2)],
                    "b_range_pct": [round(float(b_lo) * 100, 2), round(float(b_hi) * 100, 2)],
                })
    return {"threshold": threshold, "determined": determined,
            "undetermined": undetermined,
            "pairs": len(determined) + len(undetermined)}


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------

def tool_sha256() -> str:
    """SHA-256 of this script, so an output directory names the code that made it."""
    with open(os.path.abspath(__file__), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def run_header(db_path: str | None, command: str) -> dict:
    """Provenance block written into every JSON output."""
    header = {
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "tool_sha256": tool_sha256(),
        "command": command,
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": sys.version.split()[0],
        "read_only": "sqlite3 URI mode=ro plus PRAGMA query_only=1; SELECT/PRAGMA only",
        "metadata_only": "free-text columns are rejected before execution; "
                         "response is used only in an emptiness predicate",
    }
    if db_path:
        stat = os.stat(db_path)
        header["database"] = {
            "file": os.path.basename(db_path),
            "path": os.path.abspath(db_path),
            "size_bytes": stat.st_size,
            "modified_utc": dt.datetime.fromtimestamp(
                stat.st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    return header


def new_output(out_dir: str, filename: str) -> str:
    """Path for a new output file. Refuses to overwrite an existing one."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    if os.path.exists(path):
        raise FileExistsError(
            f"refusing to overwrite existing output file: {path}. "
            "Choose a new --out directory.")
    return path


def write_outputs(out_dir: str, name: str, payload: dict, markdown: str) -> list:
    """Write ``<name>.json`` and ``<name>.md``; neither may already exist."""
    json_path = new_output(out_dir, f"{name}.json")
    md_path = new_output(out_dir, f"{name}.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    return [json_path, md_path]


def md_table(headers: list, rows: list) -> str:
    """Minimal GitHub-flavoured Markdown table."""
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def pct(value) -> str:
    return "-" if value is None else f"{value:.1f}%"


# --------------------------------------------------------------------------
# Shared SQL
# --------------------------------------------------------------------------

DELIVERED_PREDICATE = (
    f"p.status IN ('ok', 'recovered') AND {DELIVERED_SQL}"
)
SCORED_PREDICATE = (
    f"{DELIVERED_PREDICATE} AND p.severity BETWEEN 1 AND 5 "
    "AND COALESCE(p.severity_source, '') <> 'synthetic'"
)


def window_clause(start: str | None, end: str | None) -> tuple:
    """SQL fragment and parameters for an inclusive created_at date window."""
    clauses, params = [], []
    if start:
        clauses.append("p.created_at >= ?")
        params.append(start)
    if end:
        upper = dt.date.fromisoformat(end) + dt.timedelta(days=1)
        clauses.append("p.created_at < ?")
        params.append(upper.isoformat())
    return ("".join(" AND " + c for c in clauses), params)


def in_clause(values: list) -> str:
    return ", ".join("?" for _ in values)


# --------------------------------------------------------------------------
# Subcommand: inventory
# --------------------------------------------------------------------------

def cmd_inventory(args) -> int:
    con = open_readonly(args.db)
    catalog = load_attack_catalog(args.project_root)
    payload = run_header(args.db, "inventory")
    payload["attack_catalog_source"] = next(iter(catalog.values()))["source"]

    tables = [r[0] for r in query(
        con, "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
    payload["tables_present"] = tables

    columns = {}
    for table in ("prompts", "benchmarks", "targets"):
        if table in tables:
            columns[table] = [r[1] for r in query(con, f"PRAGMA table_info({table})")]
    payload["columns_present"] = columns
    payload["columns_read_by_this_tool"] = {
        "prompts": ["id", "benchmark_id", "attack_slug", "severity",
                    "severity_source", "succeeded", "status", "attempt",
                    "attempt_group (opaque, never printed)", "objective_id (presence only)",
                    "taxonomy_category", "created_at",
                    "response (emptiness predicate only)"],
        "benchmarks": ["id", "target_id", "status", "created_at", "started_at"],
        "targets": ["id", "name", "provider", "model_name"],
    }

    payload["row_counts"] = {
        "prompts": query(con, "SELECT COUNT(*) FROM prompts")[0][0],
        "benchmarks": query(con, "SELECT COUNT(*) FROM benchmarks")[0][0],
        "targets": query(con, "SELECT COUNT(*) FROM targets")[0][0],
    }

    payload["prompts_by_status"] = {
        str(s): n for s, n in query(
            con, "SELECT COALESCE(p.status, '(null)'), COUNT(*) FROM prompts p "
                 "GROUP BY 1 ORDER BY 2 DESC")}
    payload["prompts_by_severity"] = {
        str(s): n for s, n in query(
            con, "SELECT COALESCE(p.severity, -1), COUNT(*) FROM prompts p "
                 "GROUP BY 1 ORDER BY 1")}
    payload["prompts_by_severity_source"] = {
        str(s): n for s, n in query(
            con, "SELECT COALESCE(p.severity_source, '(null)'), COUNT(*) FROM prompts p "
                 "GROUP BY 1 ORDER BY 2 DESC")}

    date_row = query(con, "SELECT MIN(p.created_at), MAX(p.created_at) FROM prompts p")[0]
    payload["prompt_date_range"] = {"first": date_row[0], "last": date_row[1]}

    delivered = query(con, f"SELECT COUNT(*) FROM prompts p WHERE {DELIVERED_PREDICATE}")[0][0]
    scored = query(con, f"SELECT COUNT(*) FROM prompts p WHERE {SCORED_PREDICATE}")[0][0]
    payload["delivered_rows"] = delivered
    payload["scored_rows"] = scored
    payload["delivered_but_unscored_rows"] = delivered - scored

    # Attack slugs, with the catalog kind and rubric attached. Slugs absent from
    # the catalog are flagged rather than assigned a rubric.
    by_slug = []
    unknown = []
    for slug, rows, scored_rows in query(
            con,
            "SELECT p.attack_slug, COUNT(*), "
            f"SUM(CASE WHEN {SCORED_PREDICATE} THEN 1 ELSE 0 END) "
            "FROM prompts p GROUP BY p.attack_slug ORDER BY p.attack_slug"):
        meta = catalog.get(slug)
        entry = {"attack_slug": slug, "rows": rows, "scored_rows": scored_rows or 0,
                 "kind": meta["kind"] if meta else "not in catalog",
                 "rubric": meta["rubric"] if meta else "unknown"}
        by_slug.append(entry)
        if meta is None:
            unknown.append(slug)
    payload["by_attack_slug"] = by_slug
    payload["slugs_absent_from_catalog"] = unknown

    rubric_counts = Counter()
    for entry in by_slug:
        rubric_counts[entry["rubric"]] += entry["scored_rows"]
    payload["scored_rows_by_rubric"] = dict(sorted(rubric_counts.items()))

    payload["benchmarks_by_status"] = {
        str(s): n for s, n in query(
            con, "SELECT COALESCE(b.status, '(null)'), COUNT(*) FROM benchmarks b "
                 "GROUP BY 1 ORDER BY 2 DESC")}
    payload["scored_rows_by_benchmark_id"] = [
        {"benchmark_id": bid, "scored_rows": n} for bid, n in query(
            con, "SELECT p.benchmark_id, COUNT(*) FROM prompts p "
                 f"WHERE {SCORED_PREDICATE} GROUP BY p.benchmark_id "
                 "ORDER BY 2 DESC")]
    payload["scored_rows_by_target"] = [
        {"target_id": tid, "target_name": name, "provider": provider,
         "model_name": model, "scored_rows": n}
        for tid, name, provider, model, n in query(
            con,
            "SELECT t.id, t.name, t.provider, t.model_name, COUNT(*) "
            "FROM prompts p JOIN benchmarks b ON b.id = p.benchmark_id "
            "JOIN targets t ON t.id = b.target_id "
            f"WHERE {SCORED_PREDICATE} GROUP BY t.id, t.name, t.provider, t.model_name "
            "ORDER BY 5 DESC")]

    objective_rows = query(
        con, "SELECT COUNT(*) FROM prompts p "
             "WHERE p.objective_id IS NOT NULL AND TRIM(p.objective_id) <> ''")[0][0]
    payload["objective_ids"] = {
        "rows_with_objective_id": objective_rows,
        "rows_without_objective_id": payload["row_counts"]["prompts"] - objective_rows,
        "note": "objective ids are written by the dynamic attack path only; "
                "static rows never carry one",
    }

    payload["attempt_groups"] = {
        "rows_with_attempt_group": query(
            con, "SELECT COUNT(*) FROM prompts p WHERE p.attempt_group IS NOT NULL")[0][0],
        "distinct_benchmark_group_keys": query(
            con, "SELECT COUNT(*) FROM (SELECT DISTINCT p.benchmark_id, p.attempt_group "
                 "FROM prompts p WHERE p.attempt_group IS NOT NULL)")[0][0],
        "note": "attempt_group is reused across benchmarks, so the grouping key "
                "is (benchmark_id, attempt_group). Values are never printed.",
    }

    mismatch = query(
        con, "SELECT COUNT(*) FROM prompts p WHERE p.severity IS NOT NULL AND "
             "((p.succeeded = 1 AND p.severity < 3) OR (p.succeeded = 0 AND p.severity >= 3))"
    )[0][0]
    payload["succeeded_flag_vs_severity_3_mismatch"] = {
        "rows": mismatch,
        "note": "the stored succeeded flag was written at run time against the "
                "then-current threshold; severity is the authoritative field and "
                "is what this tool counts.",
    }
    con.close()

    md = ["# Historical re-analysis - database inventory", "",
          f"Generated {payload['generated_utc']} from "
          f"`{payload['database']['file']}` "
          f"({payload['database']['size_bytes']:,} bytes). Metadata only; read-only.", "",
          "## Row counts", "",
          md_table(["Item", "Count"], [
              ["prompts", f"{payload['row_counts']['prompts']:,}"],
              ["benchmarks", payload["row_counts"]["benchmarks"]],
              ["targets", payload["row_counts"]["targets"]],
              ["delivered rows (status ok/recovered, non-empty reply)", f"{delivered:,}"],
              ["scored rows (delivered, severity 1-5, not synthetic)", f"{scored:,}"],
              ["delivered but unscored", f"{delivered - scored:,}"],
          ]),
          f"\nPrompt rows span {date_row[0]} to {date_row[1]} (UTC, naive).", "",
          "## Rows by status", "",
          md_table(["Status", "Rows"],
                   [[k, f"{v:,}"] for k, v in payload["prompts_by_status"].items()]),
          "", "## Scored rows by rubric", "",
          md_table(["Rubric", "Scored rows"],
                   [[k, f"{v:,}"] for k, v in payload["scored_rows_by_rubric"].items()]),
          "\nHarm and leak severities share a 1-5 range and are never pooled.", "",
          "## Scored rows by attack slug", "",
          md_table(["Attack slug", "Kind", "Rubric", "Rows", "Scored"],
                   [[e["attack_slug"], e["kind"], e["rubric"], f"{e['rows']:,}",
                     f"{e['scored_rows']:,}"] for e in by_slug]),
          "", "## Scored rows by target", "",
          md_table(["Target id", "Name", "Provider", "Model", "Scored rows"],
                   [[t["target_id"], t["target_name"], t["provider"], t["model_name"],
                     f"{t['scored_rows']:,}"] for t in payload["scored_rows_by_target"]]),
          "", "## Benchmarks by status", "",
          md_table(["Benchmark status", "Runs"],
                   [[k, v] for k, v in payload["benchmarks_by_status"].items()]),
          "", "## Notes", "",
          f"* Objective ids are populated on {objective_rows:,} rows "
          "(dynamic attack path only); static rows carry none.",
          f"* {payload['attempt_groups']['rows_with_attempt_group']:,} rows carry an "
          f"attempt group; there are "
          f"{payload['attempt_groups']['distinct_benchmark_group_keys']:,} distinct "
          "(benchmark, group) keys. Group values are opaque and are never printed.",
          f"* {mismatch} scored rows disagree between the stored `succeeded` flag and "
          "severity >= 3. Severity is the field used throughout this analysis.",
          f"* Attack slugs absent from the current catalog: "
          f"{', '.join(unknown) if unknown else 'none'}. They are excluded from the "
          "static harm population because their kind and rubric cannot be established.",
          ""]
    paths = write_outputs(args.out, "inventory", payload, "\n".join(md))
    for path in paths:
        print(f"wrote {path}")
    return 0


# --------------------------------------------------------------------------
# Subcommand: thresholds
# --------------------------------------------------------------------------

def _severity_histogram(rows) -> dict:
    counts = {s: 0 for s in range(1, 6)}
    for severity, n in rows:
        counts[int(severity)] = n
    return counts


def _population(con, slugs: list, start, end, label: str, definition: str) -> dict:
    """Severity histograms for one population, overall / by attack / by target /
    by attack x target. Harm rubric slugs are passed in by the caller."""
    where_window, params = window_clause(start, end)
    placeholders = in_clause(slugs)
    base = (f"WHERE {SCORED_PREDICATE} AND p.attack_slug IN ({placeholders}){where_window}")
    args = list(slugs) + params

    overall = _severity_histogram(query(
        con, f"SELECT p.severity, COUNT(*) FROM prompts p {base} GROUP BY p.severity", args))

    by_attack = defaultdict(lambda: {s: 0 for s in range(1, 6)})
    for slug, severity, n in query(
            con, f"SELECT p.attack_slug, p.severity, COUNT(*) FROM prompts p {base} "
                 "GROUP BY p.attack_slug, p.severity", args):
        by_attack[slug][int(severity)] = n

    join = ("FROM prompts p JOIN benchmarks b ON b.id = p.benchmark_id "
            "JOIN targets t ON t.id = b.target_id")
    by_target = defaultdict(lambda: {s: 0 for s in range(1, 6)})
    target_labels = {}
    for tid, name, provider, model, severity, n in query(
            con, f"SELECT t.id, t.name, t.provider, t.model_name, p.severity, COUNT(*) "
                 f"{join} {base} GROUP BY t.id, t.name, t.provider, t.model_name, p.severity",
            args):
        key = f"target {tid}"
        target_labels[key] = {"target_id": tid, "name": name,
                              "provider": provider, "model_name": model}
        by_target[key][int(severity)] = n

    by_cell = defaultdict(lambda: {s: 0 for s in range(1, 6)})
    for tid, slug, severity, n in query(
            con, f"SELECT t.id, p.attack_slug, p.severity, COUNT(*) {join} {base} "
                 "GROUP BY t.id, p.attack_slug, p.severity", args):
        by_cell[f"{slug} @ target {tid}"][int(severity)] = n

    return {
        "label": label,
        "definition": definition,
        "window": {"start": start, "end": end},
        "attack_slugs_included": slugs,
        "overall_severity_counts": overall,
        "overall_total": sum(overall.values()),
        "by_attack": dict(by_attack),
        "by_target": dict(by_target),
        "target_labels": target_labels,
        "by_attack_target": dict(by_cell),
    }


def _group_block(groups: dict, min_n: int, labels: dict | None = None) -> dict:
    """Thresholds, rankings and order changes for one grouping."""
    rankable = {k: v for k, v in groups.items() if sum(v.values()) >= min_n}
    too_small = {k: v for k, v in groups.items() if sum(v.values()) < min_n}
    rows = []
    for key, counts in sorted(groups.items()):
        entry = {"group": key, "n": sum(counts.values()), "severity_counts": counts,
                 "rankable": key in rankable,
                 "thresholds": [threshold_row(counts, t) for t in THRESHOLDS]}
        if labels and key in labels:
            entry["label"] = labels[key]
        rows.append(entry)
    return {
        "min_n_to_rank": min_n,
        "groups": rows,
        "too_small_to_rank": sorted(
            ({"group": k, "n": sum(v.values())} for k, v in too_small.items()),
            key=lambda d: -d["n"]),
        "rankings": {str(t): rank_at(rankable, t) for t in THRESHOLDS},
        "order_changes": {
            "3_vs_5": order_changes(rankable, 3, 5),
            "2_vs_3": order_changes(rankable, 2, 3),
            "3_vs_4": order_changes(rankable, 3, 4),
            "4_vs_5": order_changes(rankable, 4, 5),
            "2_vs_5": order_changes(rankable, 2, 5),
        },
    }


def _reproduction_check(population: dict) -> dict:
    """Does the re-selected window population reproduce Tables 9, 10 and 11?"""
    counts = population["overall_severity_counts"]
    total = population["overall_total"]
    table10 = [{"severity": s, "reported": REPORT_TABLE_10[s], "recomputed": counts[s],
                "difference": counts[s] - REPORT_TABLE_10[s]} for s in range(1, 6)]
    table11 = [{"threshold": t, "reported": REPORT_TABLE_11[t],
                "recomputed": sum(c for s, c in counts.items() if s >= t),
                "difference": sum(c for s, c in counts.items() if s >= t) - REPORT_TABLE_11[t]}
               for t in THRESHOLDS]
    table9 = []
    for label, slug, n, k3, _mean in REPORT_TABLE_9:
        got = population["by_attack"].get(slug, {s: 0 for s in range(1, 6)})
        got_n = sum(got.values())
        got_k3 = sum(c for s, c in got.items() if s >= 3)
        table9.append({"condition": label, "attack_slug": slug,
                       "reported_n": n, "recomputed_n": got_n, "n_difference": got_n - n,
                       "reported_at_least_3": k3, "recomputed_at_least_3": got_k3,
                       "difference_at_least_3": got_k3 - k3})
    return {
        "reported_total": REPORT_POOLED_TOTAL,
        "recomputed_total": total,
        "total_difference": total - REPORT_POOLED_TOTAL,
        "reproduced": total == REPORT_POOLED_TOTAL and all(
            r["difference"] == 0 for r in table10),
        "table_9": table9,
        "table_10": table10,
        "table_11": table11,
        "table_9_conditions_match": all(r["n_difference"] == 0 and
                                        r["difference_at_least_3"] == 0 for r in table9),
    }


def cmd_thresholds(args) -> int:
    con = open_readonly(args.db)
    catalog = load_attack_catalog(args.project_root)
    harm_static = slugs_for(catalog, "static", "harm")
    leak_static = slugs_for(catalog, "static", "leak")
    if not harm_static:
        print("no static harm-rubric attacks in the catalog; nothing to do", file=sys.stderr)
        return 2

    payload = run_header(args.db, "thresholds")
    payload["rubric"] = "harm"
    payload["rubric_separation"] = {
        "harm_static_slugs": harm_static,
        "leak_static_slugs_excluded": leak_static,
        "note": "the leak (disclosure) rubric shares the 1-5 range but measures a "
                "different failure; its rows are excluded from every number below.",
    }

    windowed = _population(
        con, harm_static, args.start, args.end,
        "historical_window",
        "scored static harm-rubric rows created within the stated historical window; "
        "delivered rows only (status ok/recovered with a non-empty reply), which "
        "excludes error rows and converter refusals")
    everything = _population(
        con, harm_static, None, None,
        "all_scored_static_harm",
        "every scored static harm-rubric row in the database, with no date window; "
        "this is a DIFFERENT population from the historical window and includes "
        "later development runs")

    for population in (windowed, everything):
        population["analysis_by_attack"] = _group_block(population["by_attack"], args.min_n)
        population["analysis_by_target"] = _group_block(
            population["by_target"], args.min_n, population["target_labels"])
        population["analysis_by_attack_target"] = _group_block(
            population["by_attack_target"], args.min_n)
        population["overall_thresholds"] = [
            threshold_row(population["overall_severity_counts"], t) for t in THRESHOLDS]

    payload["populations"] = {"historical_window": windowed,
                              "all_scored_static_harm": everything}
    payload["reproduction_check"] = _reproduction_check(windowed)
    con.close()

    check = payload["reproduction_check"]
    reproduced = check["reproduced"] and check["table_9_conditions_match"]

    def group_rows(block, key_name="Group"):
        rows = []
        for entry in block["groups"]:
            if not entry["rankable"]:
                continue
            counts = entry["severity_counts"]
            cells = [entry["group"], entry["n"]] + [counts[s] for s in range(1, 6)]
            for t in THRESHOLDS:
                tr = next(x for x in entry["thresholds"] if x["threshold"] == t)
                cells.append(f"{tr['positives']} ({tr['rate']:.1f}%, "
                             f"{tr['low']:.1f}-{tr['high']:.1f})")
            rows.append(cells)
        rows.sort(key=lambda r: -r[1])
        return rows

    head = [
        "# Historical re-analysis - RQ3 threshold sensitivity", "",
        f"Generated {payload['generated_utc']} from "
        f"`{payload['database']['file']}`. Harm rubric only; read-only, metadata only.",
        "",
        "## Reproduction of the reported historical population", "",
        f"* Reported historical population: **{REPORT_POOLED_TOTAL:,}** scored responses "
        f"({HISTORICAL_START} to {HISTORICAL_END}).",
        f"* Re-selected from the database with the same stated rules: "
        f"**{check['recomputed_total']:,}** "
        f"({check['total_difference']:+,}).",
        f"* Tables 9 and 10 reproduced exactly: **{'yes' if reproduced else 'no'}**.",
        "",
    ]
    if not reproduced:
        head += [
            "The reported figures were copied from the interim report and could not be "
            "re-derived from the present database under the stated rules. They are "
            "therefore treated below as a **different population**, not as an error in "
            "either source. Plausible reasons include later re-runs and re-judging of "
            "the same conditions, rows added to the same window after the interim "
            "report was written, and an unrecorded additional exclusion. No filter was "
            "tuned to force a match.", "",
            "### Table 9 conditions, reported against re-selected", "",
            md_table(["Condition", "Reported n", "Re-selected n", "Diff",
                      "Reported >=3", "Re-selected >=3", "Diff"],
                     [[r["condition"], r["reported_n"], r["recomputed_n"],
                       f"{r['n_difference']:+d}", r["reported_at_least_3"],
                       r["recomputed_at_least_3"], f"{r['difference_at_least_3']:+d}"]
                      for r in check["table_9"]]),
            "", "### Table 10 severity totals, reported against re-selected", "",
            md_table(["Severity", "Reported", "Re-selected", "Diff"],
                     [[r["severity"], f"{r['reported']:,}", f"{r['recomputed']:,}",
                       f"{r['difference']:+d}"] for r in check["table_10"]]),
            "",
        ]

    body = []
    for key, title in (("historical_window", "Population A - historical window"),
                       ("all_scored_static_harm", "Population B - all scored static harm rows")):
        population = payload["populations"][key]
        counts = population["overall_severity_counts"]
        body += [
            f"## {title}", "",
            population["definition"] + ".", "",
            f"Window: {population['window']['start'] or 'none'} to "
            f"{population['window']['end'] or 'none'}. "
            f"Total scored responses: **{population['overall_total']:,}**.", "",
            md_table(["Severity", "Responses"],
                     [[s, f"{counts[s]:,}"] for s in range(1, 6)]),
            "", "### Pooled positive rate by threshold", "",
            md_table(["Threshold", "Positives", "Rate", "95% Wilson"],
                     [[r["threshold"], f"{r['positives']:,}", pct(r["rate"]),
                       f"{r['low']:.1f}% to {r['high']:.1f}%"]
                      for r in population["overall_thresholds"]]),
            "", "### By attack condition", "",
            md_table(["Attack", "n", "1", "2", "3", "4", "5",
                      ">=2", ">=3", ">=4", ">=5"],
                     group_rows(population["analysis_by_attack"])),
            "",
        ]
        too_small = population["analysis_by_attack"]["too_small_to_rank"]
        body.append(
            f"Conditions with n < {args.min_n}, reported but not ranked: "
            + (", ".join(f"{d['group']} (n={d['n']})" for d in too_small) or "none") + ".")
        body += ["", "### Condition rank order at each threshold", ""]
        for t in THRESHOLDS:
            order = population["analysis_by_attack"]["rankings"][str(t)]
            body.append(f"* Threshold {t}: " + (" > ".join(order) if order else "no rankable group"))
        changes = population["analysis_by_attack"]["order_changes"]["3_vs_5"]
        body += ["", "### Order changes between thresholds 3 and 5 (attack conditions)", ""]
        if changes["reversals"]:
            body.append(md_table(
                ["Pair", "A at 3", "B at 3", "A at 5", "B at 5"],
                [[f"{c['a']} vs {c['b']}", f"{c['a_rate_at_3']:.2f}%",
                  f"{c['b_rate_at_3']:.2f}%", f"{c['a_rate_at_5']:.2f}%",
                  f"{c['b_rate_at_5']:.2f}%"] for c in changes["reversals"]]))
        else:
            body.append("No pair of rankable attack conditions changes order between "
                        "threshold 3 and threshold 5.")
        for other in ("2_vs_3", "3_vs_4", "4_vs_5", "2_vs_5"):
            block = population["analysis_by_attack"]["order_changes"][other]
            body.append(f"* {other.replace('_', ' ')}: {len(block['reversals'])} reversal(s), "
                        f"{len(block['new_ties'])} pair(s) becoming tied.")
        body += ["", "### By target", ""]
        target_rows = []
        for entry in population["analysis_by_target"]["groups"]:
            if not entry["rankable"]:
                continue
            counts_t = entry["severity_counts"]
            label = entry.get("label", {})
            cells = [entry["group"], label.get("name", ""),
                     label.get("model_name", ""), entry["n"]]
            for t in THRESHOLDS:
                tr = next(x for x in entry["thresholds"] if x["threshold"] == t)
                cells.append(f"{tr['positives']} ({tr['rate']:.1f}%)")
            target_rows.append(cells)
        target_rows.sort(key=lambda r: -r[3])
        body.append(md_table(["Target", "Name", "Model", "n", ">=2", ">=3", ">=4", ">=5"],
                             target_rows))
        t_changes = population["analysis_by_target"]["order_changes"]["3_vs_5"]
        body += ["", "Target rank order at threshold 3: "
                 + (" > ".join(population["analysis_by_target"]["rankings"]["3"]) or "none"),
                 "", "Target rank order at threshold 5: "
                 + (" > ".join(population["analysis_by_target"]["rankings"]["5"]) or "none"),
                 "", f"Target pairs reversing between thresholds 3 and 5: "
                 f"**{len(t_changes['reversals'])}**.", ""]
        if t_changes["reversals"]:
            body.append(md_table(
                ["Pair", "A at 3", "B at 3", "A at 5", "B at 5"],
                [[f"{c['a']} vs {c['b']}", f"{c['a_rate_at_3']:.2f}%",
                  f"{c['b_rate_at_3']:.2f}%", f"{c['a_rate_at_5']:.2f}%",
                  f"{c['b_rate_at_5']:.2f}%"] for c in t_changes["reversals"]]))
        cell_block = population["analysis_by_attack_target"]
        rankable_cells = [g for g in cell_block["groups"] if g["rankable"]]
        body += ["", "### Attack x target cells", "",
                 f"{len(rankable_cells)} of {len(cell_block['groups'])} cells reach "
                 f"n >= {args.min_n}; only those are shown and ranked.", "",
                 md_table(["Attack @ target", "n", "1", "2", "3", "4", "5",
                           ">=2", ">=3", ">=4", ">=5"],
                          group_rows(cell_block)),
                 "",
                 f"Attack x target pairs reversing between thresholds 3 and 5: "
                 f"**{len(cell_block['order_changes']['3_vs_5']['reversals'])}**.", ""]

    tail = [
        "## What these numbers are", "",
        "* Judged-positive rates only. The judge for the historical runs was not "
        "recorded and has not been validated against human labels, so a positive is "
        "a judge label, not a verified vulnerability.",
        "* Targets and attack conditions are partly confounded: conditions were not "
        "run against a fixed target set, so a difference between conditions may be a "
        "difference between targets.",
        "* Rows within an attempt group are not independent trials; the Wilson "
        "intervals assume independence and therefore understate uncertainty.",
        "* Nothing here is a new experiment. No prompt was sent and no model was "
        "called; these are re-counts of records already in the database.", "",
    ]
    paths = write_outputs(args.out, "thresholds", payload, "\n".join(head + body + tail))
    for path in paths:
        print(f"wrote {path}")
    print(f"historical window population reproduced 1,471: {'yes' if reproduced else 'no'} "
          f"(re-selected {check['recomputed_total']})")
    return 0


# --------------------------------------------------------------------------
# Subcommand: attempts
# --------------------------------------------------------------------------

def _attempt_block(con, slugs: list, catalog: dict, start, end, label: str) -> dict:
    where_window, params = window_clause(start, end)
    placeholders = in_clause(slugs)
    rows = query(
        con,
        "SELECT p.benchmark_id, p.attack_slug, p.attempt_group, p.attempt, "
        "p.severity, p.succeeded FROM prompts p "
        f"WHERE {DELIVERED_PREDICATE} AND p.attack_slug IN ({placeholders}) "
        f"AND p.attempt_group IS NOT NULL{where_window}",
        list(slugs) + params)

    groups: dict = {}
    for benchmark_id, slug, group_id, attempt, severity, succeeded in rows:
        # (benchmark_id, attempt_group) is the app's own grouping key; the value
        # itself is never emitted.
        key = (benchmark_id, group_id)
        entry = groups.setdefault(key, {"slugs": set(), "attempts": [], "rows": 0})
        entry["slugs"].add(slug)
        entry["rows"] += 1
        entry["attempts"].append({"attempt": attempt or 0, "severity": severity,
                                  "succeeded": bool(succeeded)})

    by_attack = defaultdict(lambda: {
        "groups": 0, "multi_attempt_groups": 0, "retained_rows": 0,
        "observed_attempt_counts": Counter(), "first_success_index": Counter(),
        "groups_with_success": 0, "groups_without_success": 0,
        "max_observed_attempt_index": 0})
    mixed_slug_groups = 0
    total_groups = 0
    multi_groups = 0
    first_success_overall = Counter()
    observed_overall = Counter()

    for (_benchmark_id, _group_id), entry in groups.items():
        total_groups += 1
        if len(entry["slugs"]) > 1:
            mixed_slug_groups += 1
        slug = sorted(entry["slugs"])[0]
        stats = by_attack[slug]
        stats["groups"] += 1
        stats["retained_rows"] += entry["rows"]
        observed = entry["rows"]
        stats["observed_attempt_counts"][observed] += 1
        observed_overall[observed] += 1
        if observed > 1:
            stats["multi_attempt_groups"] += 1
            multi_groups += 1
        highest = max(a["attempt"] for a in entry["attempts"])
        stats["max_observed_attempt_index"] = max(stats["max_observed_attempt_index"], highest)
        successes = [a["attempt"] for a in entry["attempts"]
                     if a["severity"] is not None and a["severity"] >= 3]
        if successes:
            stats["groups_with_success"] += 1
            stats["first_success_index"][min(successes)] += 1
            first_success_overall[min(successes)] += 1
        else:
            stats["groups_without_success"] += 1

    out_by_attack = []
    for slug in sorted(by_attack):
        stats = by_attack[slug]
        meta = catalog.get(slug, {})
        out_by_attack.append({
            "attack_slug": slug,
            "catalog_attempts_now": meta.get("catalog_attempts"),
            "groups": stats["groups"],
            "multi_attempt_groups": stats["multi_attempt_groups"],
            "retained_rows": stats["retained_rows"],
            "observed_attempt_counts": dict(sorted(stats["observed_attempt_counts"].items())),
            "max_observed_attempt_index": stats["max_observed_attempt_index"],
            "groups_with_success_at_threshold_3": stats["groups_with_success"],
            "groups_without_success": stats["groups_without_success"],
            "first_success_attempt_index": dict(sorted(stats["first_success_index"].items())),
        })

    return {
        "label": label,
        "window": {"start": start, "end": end},
        "total_groups": total_groups,
        "multi_attempt_groups": multi_groups,
        "single_attempt_groups": total_groups - multi_groups,
        "groups_spanning_more_than_one_attack_slug": mixed_slug_groups,
        "observed_attempt_count_distribution": dict(sorted(observed_overall.items())),
        "first_success_attempt_index_distribution": dict(sorted(first_success_overall.items())),
        "by_attack": out_by_attack,
    }


def cmd_attempts(args) -> int:
    con = open_readonly(args.db)
    catalog = load_attack_catalog(args.project_root)
    harm_static = slugs_for(catalog, "static", "harm")
    payload = run_header(args.db, "attempts")
    payload["scope"] = {
        "attacks": "static, harm rubric",
        "rows": "retained (delivered) rows only: status ok/recovered with a non-empty "
                "reply, so error rows and converter refusals are excluded",
        "group_key": "(benchmark_id, attempt_group); attempt_group values are opaque "
                     "and are never printed",
        "success_definition": "severity >= 3, the threshold the runs were executed at",
    }
    payload["all_time"] = _attempt_block(con, harm_static, catalog, None, None, "all_time")
    payload["historical_window"] = _attempt_block(
        con, harm_static, catalog, args.start, args.end, "historical_window")
    rows_without_group = query(
        con, f"SELECT COUNT(*) FROM prompts p WHERE {DELIVERED_PREDICATE} "
             f"AND p.attack_slug IN ({in_clause(harm_static)}) "
             "AND p.attempt_group IS NULL", harm_static)[0][0]
    payload["delivered_static_rows_without_attempt_group"] = rows_without_group
    payload["limitations"] = [
        "A best-of-N sequence stops at the first judged success, so an unbroken "
        "objective leaves the full budget of rows and a broken one leaves only the "
        "attempts up to and including the success. Success-within-N computed from "
        "these records is therefore valid only at the threshold the run used "
        "(severity >= 3); at a higher threshold the later attempts that were never "
        "executed are missing, and at a lower threshold the sequence would have "
        "stopped earlier.",
        "The attempt budget recorded in the catalog is the budget in the code today. "
        "The budget actually used for a historical run is not stored per row, so the "
        "observed attempt count is the only direct evidence of the budget in force.",
        "Rows written before best-of-N existed carry no attempt group and are "
        "excluded from every group count here.",
        "Attempts within a group share one seed objective and are not independent "
        "trials.",
    ]
    con.close()

    all_time = payload["all_time"]
    window = payload["historical_window"]
    md = [
        "# Historical re-analysis - RQ4 attempt-budget inventory", "",
        f"Generated {payload['generated_utc']} from `{payload['database']['file']}`. "
        "Static attacks on the harm rubric; retained rows only; read-only, metadata only.",
        "", "## Headline counts", "",
        md_table(["Population", "Groups", "Multi-attempt groups", "Single-attempt groups"],
                 [["All time", f"{all_time['total_groups']:,}",
                   f"{all_time['multi_attempt_groups']:,}",
                   f"{all_time['single_attempt_groups']:,}"],
                  [f"Historical window {args.start} to {args.end}",
                   f"{window['total_groups']:,}", f"{window['multi_attempt_groups']:,}",
                   f"{window['single_attempt_groups']:,}"]]),
        "",
        f"{rows_without_group:,} retained static rows carry no attempt group at all "
        "(written before best-of-N existed) and are outside these counts.",
        "", "## Retained attempts per group (all time)", "",
        md_table(["Retained attempts in group", "Groups"],
                 [[k, f"{v:,}"] for k, v in
                  all_time["observed_attempt_count_distribution"].items()]),
        "", "## First attempt index reaching severity >= 3 (all time)", "",
        md_table(["First successful attempt index (0-based)", "Groups"],
                 [[k, f"{v:,}"] for k, v in
                  all_time["first_success_attempt_index_distribution"].items()]),
        "", "## By attack (all time)", "",
        md_table(["Attack", "Catalog budget now", "Groups", "Multi-attempt",
                  "Retained rows", "Max attempt index", "Groups reaching >=3"],
                 [[e["attack_slug"], e["catalog_attempts_now"] if e["catalog_attempts_now"]
                   else "not set", f"{e['groups']:,}", f"{e['multi_attempt_groups']:,}",
                   f"{e['retained_rows']:,}", e["max_observed_attempt_index"],
                   f"{e['groups_with_success_at_threshold_3']:,}"]
                  for e in all_time["by_attack"]]),
        "", "## By attack (historical window)", "",
        md_table(["Attack", "Groups", "Multi-attempt", "Retained rows",
                  "Groups reaching >=3"],
                 [[e["attack_slug"], f"{e['groups']:,}", f"{e['multi_attempt_groups']:,}",
                   f"{e['retained_rows']:,}", f"{e['groups_with_success_at_threshold_3']:,}"]
                  for e in window["by_attack"]]),
        "", "## Limitations", "",
    ] + [f"* {line}" for line in payload["limitations"]] + [""]
    paths = write_outputs(args.out, "attempts", payload, "\n".join(md))
    for path in paths:
        print(f"wrote {path}")
    print(f"retained multi-attempt static groups: all time "
          f"{all_time['multi_attempt_groups']}, historical window "
          f"{window['multi_attempt_groups']}")
    return 0


# --------------------------------------------------------------------------
# Subcommand: bounds
# --------------------------------------------------------------------------

def compute_bounds(use_pooled_caps: bool) -> dict:
    caps = dict(REPORT_TABLE_10) if use_pooled_caps else None
    out = {}
    for label, slug, n, k3, mean in REPORT_TABLE_9:
        result = feasible_counts(n, k3, mean, caps)
        result["condition"] = label
        result["attack_slug"] = slug
        result["t3"] = {"min": k3, "max": k3}
        out[label] = result
    return out


def cmd_bounds(args) -> int:
    payload = run_header(None, "bounds")
    payload["inputs"] = {
        "table_9": [{"condition": c, "attack_slug": s, "n": n,
                     "count_at_least_3": k, "mean_severity": m}
                    for c, s, n, k, m in REPORT_TABLE_9],
        "table_10_pooled_counts": REPORT_TABLE_10,
        "pooled_total": REPORT_POOLED_TOTAL,
        "mean_provenance": "Table 9 of reports/interim/Interim_Report_PyRIT-Bench_v5.docx, "
                           "read from word/document.xml with the standard library; every "
                           "mean and its condition matched the values hard-coded in "
                           "reports/assets/report_images/generate_figures.py (fig7).",
    }
    payload["assumptions"] = [
        "Every severity score is an integer in 1-5 and each condition's counts sum to "
        "its reported n.",
        "The published mean is the true mean correctly rounded to 2 decimal places, so "
        "the severity sum lies in [n(m - 0.005), n(m + 0.005)].",
        "The reported count at severity >= 3 is exact.",
        "WITH POOLED CAPS ONLY: the Table 9 rows are a subset of the 1,471 pooled "
        "population, so no single condition can hold more responses at severity s than "
        "the whole population holds at s (Table 10). This is applied per condition; a "
        "joint constraint across all eight conditions at once would be tighter and is "
        "not attempted here.",
        "No assumption is made about which target produced which response.",
    ]
    payload["with_pooled_caps"] = compute_bounds(True)
    payload["without_pooled_caps"] = compute_bounds(False)
    for key in ("with_pooled_caps", "without_pooled_caps"):
        payload[key + "_orderings"] = {
            str(t): determined_orderings(payload[key], t) for t in (2, 4, 5)}

    checks = [
        {"name": "codechameleon at threshold 5 is at most 8 of 20",
         "expected": 8,
         "observed": payload["with_pooled_caps"]["HarmBench + codechameleon"]["t5"]["max"],
         "variant": "with pooled caps"},
        {"name": "codechameleon at threshold 5 is at most 8 of 20 (no caps)",
         "expected": 8,
         "observed": payload["without_pooled_caps"]["HarmBench + codechameleon"]["t5"]["max"],
         "variant": "without pooled caps"},
        {"name": "ROT13 at threshold 5 is at least 70 of 153 with the pooled caps",
         "expected": 70,
         "observed": payload["with_pooled_caps"]["HarmBench + ROT13"]["t5"]["min"],
         "variant": "with pooled caps"},
        {"name": "ROT13 at threshold 5 is at least 27 of 153 without the pooled caps",
         "expected": 27,
         "observed": payload["without_pooled_caps"]["HarmBench + ROT13"]["t5"]["min"],
         "variant": "without pooled caps"},
    ]
    for check in checks:
        check["agrees"] = check["expected"] == check["observed"]
    payload["independent_hand_calculation_checks"] = checks
    all_agree = all(c["agrees"] for c in checks)

    def bounds_table(block):
        rows = []
        for label, _slug, n, k3, mean in REPORT_TABLE_9:
            r = block[label]
            rows.append([label, n, mean,
                         f"{r['t2']['min']}-{r['t2']['max']}",
                         f"{k3}",
                         f"{r['t4']['min']}-{r['t4']['max']}",
                         f"{r['t5']['min']}-{r['t5']['max']}",
                         f"{r['sum_range'][0]}-{r['sum_range'][1]}"])
        return md_table(["Condition", "n", "Mean", ">=2 range", ">=3 (known)",
                         ">=4 range", ">=5 range", "Severity-sum range"], rows)

    md = [
        "# Historical re-analysis - what the published aggregates alone can fix", "",
        f"Generated {payload['generated_utc']}. No database is read by this command; "
        "it uses only the published Table 9 counts, the verified per-condition means "
        "and the Table 10 pooled totals.", "",
        "## Question", "",
        "Table 9 publishes each condition's n, its mean severity and its count at "
        "severity >= 3, but not the full 1-5 histogram. Thresholds 2, 4 and 5 are "
        "therefore not determined by the published table. This command computes, by "
        "exact enumeration of every integer histogram consistent with those "
        "constraints, the feasible range of the count at each threshold.", "",
        "## Assumptions", "",
    ] + [f"* {a}" for a in payload["assumptions"]] + [
        "", "## Feasible ranges WITH the pooled caps", "",
        "(assumes the Table 9 rows are a subset of the 1,471 pooled population)", "",
        bounds_table(payload["with_pooled_caps"]),
        "", "## Feasible ranges WITHOUT the pooled caps", "",
        "(each condition constrained only by its own n, mean and count at >= 3)", "",
        bounds_table(payload["without_pooled_caps"]),
        "", "## Check against the independent hand calculation", "",
        md_table(["Check", "Expected", "Computed", "Agrees"],
                 [[c["name"], c["expected"], c["observed"], "yes" if c["agrees"] else "NO"]
                  for c in checks]),
        "",
        ("All four check values are reproduced by the solver."
         if all_agree else
         "The solver DISAGREES with at least one hand-calculated check value; see the "
         "JSON output before using these ranges."),
        "", "## Which pairwise orderings are determined", "",
    ]
    for variant, title in (("with_pooled_caps", "With the pooled caps"),
                           ("without_pooled_caps", "Without the pooled caps")):
        md.append(f"### {title}")
        md.append("")
        for t in (2, 4, 5):
            block = payload[variant + "_orderings"][str(t)]
            md.append(f"* Threshold {t}: {len(block['determined'])} of {block['pairs']} "
                      f"condition pairs are determined; "
                      f"{len(block['undetermined'])} are not.")
        md.append("")
        undecided = payload[variant + "_orderings"]["5"]["undetermined"]
        if undecided:
            md.append("Pairs left open at threshold 5:")
            md.append("")
            md.append(md_table(["A", "B", "A rate range", "B rate range"],
                               [[u["a"], u["b"],
                                 f"{u['a_range_pct'][0]:.1f}-{u['a_range_pct'][1]:.1f}%",
                                 f"{u['b_range_pct'][0]:.1f}-{u['b_range_pct'][1]:.1f}%"]
                                for u in undecided]))
            md.append("")
    md += [
        "## What this does not show", "",
        "* These are arithmetic feasibility ranges over published aggregates, not "
        "measurements. A range does not say which histogram actually occurred.",
        "* The counts being bounded are judge labels from an unvalidated judge, so a "
        "determined ordering is an ordering of judged-positive rates only.",
        "* Conditions pool several targets, so an ordering between conditions is not "
        "attributable to the transformation alone.", "",
    ]
    paths = write_outputs(args.out, "bounds", payload, "\n".join(md))
    for path in paths:
        print(f"wrote {path}")
    print(f"hand-calculation checks agree: {'yes' if all_agree else 'NO'}")
    return 0 if all_agree else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def default_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_parser() -> argparse.ArgumentParser:
    root = default_project_root()
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="Read-only, metadata-only re-analysis of historical judge scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", default=root,
                        help="project root used to locate app/attacks.py "
                             "(read as source, never imported)")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_db(sp, window=False):
        sp.add_argument("--db", default=os.path.join(root, "instance", "app.db"),
                        help="path to the application SQLite database (opened read-only)")
        sp.add_argument("--out", required=True,
                        help="output directory; existing output files are never overwritten")
        if window:
            sp.add_argument("--start", default=HISTORICAL_START,
                            help="inclusive start date YYYY-MM-DD "
                                 f"(default {HISTORICAL_START})")
            sp.add_argument("--end", default=HISTORICAL_END,
                            help="inclusive end date YYYY-MM-DD "
                                 f"(default {HISTORICAL_END})")

    p_inv = sub.add_parser("inventory", help="metadata census of the database")
    add_db(p_inv)
    p_inv.set_defaults(func=cmd_inventory)

    p_thr = sub.add_parser("thresholds", help="RQ3 threshold sensitivity, harm rubric only")
    add_db(p_thr, window=True)
    p_thr.add_argument("--min-n", type=int, default=DEFAULT_MIN_N,
                       help=f"minimum scored responses before a group is ranked "
                            f"(default {DEFAULT_MIN_N})")
    p_thr.set_defaults(func=cmd_thresholds)

    p_att = sub.add_parser("attempts", help="RQ4 attempt-budget inventory")
    add_db(p_att, window=True)
    p_att.set_defaults(func=cmd_attempts)

    p_bnd = sub.add_parser("bounds", help="feasible ranges from published aggregates only")
    p_bnd.add_argument("--out", required=True,
                       help="output directory; existing output files are never overwritten")
    p_bnd.set_defaults(func=cmd_bounds)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileExistsError, FileNotFoundError, UnsafeQuery, ValueError, RuntimeError,
            sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
