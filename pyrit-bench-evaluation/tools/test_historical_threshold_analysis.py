"""Offline checks for historical_threshold_analysis.py.

Every test uses a temporary SQLite fixture built from synthetic numbers, or pure
arithmetic. The real application database is never opened: the fixture path is
always inside a TemporaryDirectory, and one test asserts that the tool's own
connection refuses a write.

Run with:

    .venv/bin/python -B -m unittest discover -s tools -p 'test_historical_threshold_analysis.py' -v
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import historical_threshold_analysis as hta  # noqa: E402


# --------------------------------------------------------------------------
# Fixture: a miniature copy of the application schema, synthetic rows only
# --------------------------------------------------------------------------

FIXTURE_ATTACKS = '''\
ATTACKS = [
    {"slug": "syn_plain", "name": "Synthetic plain", "kind": "static"},
    {"slug": "syn_encoded", "name": "Synthetic encoded", "kind": "static",
     "attempts": 3},
    {"slug": "syn_tiny", "name": "Synthetic tiny", "kind": "static"},
    {"slug": "syn_leak", "name": "Synthetic leak", "kind": "static",
     "scoring": "leak"},
    {"slug": "syn_dynamic", "name": "Synthetic dynamic", "kind": "dynamic"},
]
'''


def build_fixture(directory: str) -> str:
    """Create a synthetic project root with app/attacks.py and instance/app.db.

    Severity layout (all rows delivered unless stated):

      syn_plain    20 rows: 16 at severity 1, 2 at 2, 1 at 4, 1 at 5
      syn_encoded  20 rows: 8 at 1, 2 at 2, 2 at 3, 0 at 4, 8 at 5
      syn_tiny      3 rows: 3 at severity 1          (below the ranking floor)
      syn_leak     20 rows: 20 at severity 5         (different rubric)
      syn_dynamic  10 rows: 10 at severity 5         (not a static attack)

    Plus 4 rows that must never be counted: a send_error, a converter_refused,
    a delivered row with no severity, and a delivered row outside the window.
    Plus 5 further syn_encoded rows forming two best-of-N groups (2 and 3
    retained attempts), which do count towards the scored population.
    """
    root = Path(directory)
    (root / "app").mkdir(parents=True, exist_ok=True)
    (root / "app" / "attacks.py").write_text(FIXTURE_ATTACKS, encoding="utf-8")
    (root / "instance").mkdir(parents=True, exist_ok=True)
    db_path = root / "instance" / "app.db"

    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE targets (
            id INTEGER PRIMARY KEY, name VARCHAR(120), provider VARCHAR(40),
            model_name VARCHAR(120), endpoint_url VARCHAR(255), api_key TEXT,
            extra_json TEXT, notes TEXT, architecture TEXT, created_at DATETIME);
        CREATE TABLE benchmarks (
            id INTEGER PRIMARY KEY, name VARCHAR(180), target_id INTEGER,
            status VARCHAR(20), status_detail TEXT, attack_slugs_json TEXT,
            llm_config_json TEXT, attacker_roster_json TEXT,
            completed_attacks_json TEXT, started_at DATETIME,
            finished_at DATETIME, total_prompts INTEGER, total_successes INTEGER,
            avg_latency_ms FLOAT, created_at DATETIME);
        CREATE TABLE prompts (
            id INTEGER PRIMARY KEY, benchmark_id INTEGER, attack_slug VARCHAR(80),
            sequence INTEGER, prompt_text TEXT, converter_used VARCHAR(60),
            response TEXT, latency_ms INTEGER, refused BOOLEAN, succeeded BOOLEAN,
            severity INTEGER, error TEXT, created_at DATETIME,
            taxonomy_category VARCHAR(60), objective_id VARCHAR(40),
            attempt INTEGER, attempt_group VARCHAR(120), capture_ref VARCHAR(80),
            status VARCHAR(24), note TEXT, verdict TEXT,
            severity_source VARCHAR(12), latency_measured BOOLEAN);
        """)
    con.execute("INSERT INTO targets (id, name, provider, model_name, endpoint_url, "
                "api_key, notes) VALUES (1, 'Synthetic A', 'openai', 'syn-model-a', "
                "'https://example.invalid/a', 'NOT-A-REAL-KEY', 'note text')")
    con.execute("INSERT INTO targets (id, name, provider, model_name) "
                "VALUES (2, 'Synthetic B', 'ui_browser', 'syn-model-b')")
    con.execute("INSERT INTO benchmarks (id, name, target_id, status, created_at) "
                "VALUES (1, 'run one', 1, 'completed', '2026-06-01 10:00:00')")
    con.execute("INSERT INTO benchmarks (id, name, target_id, status, created_at) "
                "VALUES (2, 'run two', 2, 'failed', '2026-06-02 10:00:00')")

    rows = []
    next_id = [1]

    def add(slug, severity, benchmark=1, status="ok",
            response="SYNTHETIC-RESPONSE-BODY",
            created="2026-06-10 12:00:00", attempt=0, group=None, source="judge"):
        rows.append((next_id[0], benchmark, slug, "SYNTHETIC-PROMPT-TEXT", response, severity,
                     created, attempt, group, status, source,
                     1 if (severity or 0) >= 3 else 0))
        next_id[0] += 1

    for _ in range(16):
        add("syn_plain", 1)
    for _ in range(2):
        add("syn_plain", 2)
    add("syn_plain", 4)
    add("syn_plain", 5)

    for _ in range(8):
        add("syn_encoded", 1, benchmark=2)
    for _ in range(2):
        add("syn_encoded", 2, benchmark=2)
    for _ in range(2):
        add("syn_encoded", 3, benchmark=2)
    for _ in range(8):
        add("syn_encoded", 5, benchmark=2)

    for _ in range(3):
        add("syn_tiny", 1)
    for _ in range(20):
        add("syn_leak", 5)
    for _ in range(10):
        add("syn_dynamic", 5)

    # Rows that must be excluded by the population rules.
    add("syn_plain", 5, status="send_error", response="")
    add("syn_plain", 5, status="converter_refused", response="")
    add("syn_plain", None)
    add("syn_plain", 5, created="2026-12-25 09:00:00")

    # Two best-of-N groups: one broken on the second attempt, one never broken.
    for attempt in (0, 1):
        add("syn_encoded", 1 if attempt == 0 else 4, attempt=attempt, group="grp-a")
    for attempt in (0, 1, 2):
        add("syn_encoded", 1, attempt=attempt, group="grp-b")

    con.executemany(
        "INSERT INTO prompts (id, benchmark_id, attack_slug, prompt_text, response, "
        "severity, created_at, attempt, attempt_group, status, severity_source, "
        "succeeded) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return str(db_path)


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fyp-hta-")
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name
        self.db = build_fixture(self.root)
        self.out = os.path.join(self.root, "out")

    def run_command(self, *argv, expected=0):
        code = hta.main(list(argv))
        self.assertEqual(code, expected, f"unexpected exit code for {argv}")
        return code

    def load(self, name):
        with open(os.path.join(self.out, f"{name}.json"), encoding="utf-8") as handle:
            return json.load(handle)


# --------------------------------------------------------------------------
# Threshold arithmetic
# --------------------------------------------------------------------------

class ThresholdArithmeticTests(unittest.TestCase):
    def test_threshold_row_counts_at_or_above(self):
        counts = {1: 10, 2: 3, 3: 2, 4: 1, 5: 4}
        self.assertEqual(hta.threshold_row(counts, 2)["positives"], 10)
        self.assertEqual(hta.threshold_row(counts, 3)["positives"], 7)
        self.assertEqual(hta.threshold_row(counts, 4)["positives"], 5)
        self.assertEqual(hta.threshold_row(counts, 5)["positives"], 4)
        self.assertEqual(hta.threshold_row(counts, 3)["n"], 20)

    def test_reported_table_11_is_reproduced_from_table_10(self):
        for threshold, expected in hta.REPORT_TABLE_11.items():
            self.assertEqual(
                hta.threshold_row(hta.REPORT_TABLE_10, threshold)["positives"], expected)
        self.assertEqual(sum(hta.REPORT_TABLE_10.values()), hta.REPORT_POOLED_TOTAL)

    def test_rate_fraction_is_exact_and_empty_group_is_zero(self):
        from fractions import Fraction
        self.assertEqual(hta.rate_fraction({1: 2, 2: 0, 3: 1, 4: 0, 5: 0}, 3),
                         Fraction(1, 3))
        self.assertEqual(hta.rate_fraction({s: 0 for s in range(1, 6)}, 3), Fraction(0))


class WilsonTests(unittest.TestCase):
    def test_known_published_interval(self):
        # Table 9, Forbidden Questions: 6 of 177 reported as 3.4% (1.6% to 7.2%).
        result = hta.wilson_interval(6, 177)
        self.assertEqual(result["rate"], 3.4)
        self.assertEqual(result["low"], 1.6)
        self.assertEqual(result["high"], 7.2)

    def test_every_published_table_9_interval(self):
        published = {
            "forbidden_questions": (6, 177, 3.4, 1.6, 7.2),
            "xstest": (22, 202, 10.9, 7.3, 15.9),
            "advbench": (19, 225, 8.4, 5.5, 12.8),
            "harmbench": (25, 275, 9.1, 6.2, 13.1),
            "harmbench_leetspeak": (38, 153, 24.8, 18.7, 32.2),
            "harmbench_base64": (43, 138, 31.2, 24.0, 39.3),
            "harmbench_rot13": (91, 153, 59.5, 51.6, 66.9),
            "harmbench_codechameleon": (18, 20, 90.0, 69.9, 97.2),
        }
        for slug, (k, n, rate, low, high) in published.items():
            with self.subTest(slug=slug):
                result = hta.wilson_interval(k, n)
                self.assertEqual(result["rate"], rate)
                self.assertEqual(result["low"], low)
                self.assertEqual(result["high"], high)

    def test_zero_sample_has_no_interval(self):
        self.assertEqual(hta.wilson_interval(0, 0),
                         {"rate": 0.0, "low": 0.0, "high": 0.0, "n": 0})

    def test_interval_stays_inside_zero_to_one_hundred(self):
        for k, n in ((0, 5), (5, 5), (1, 3)):
            result = hta.wilson_interval(k, n)
            self.assertGreaterEqual(result["low"], 0.0)
            self.assertLessEqual(result["high"], 100.0)


# --------------------------------------------------------------------------
# Order-change detection
# --------------------------------------------------------------------------

class OrderChangeTests(unittest.TestCase):
    def setUp(self):
        # A leads at threshold 3 (60% vs 50%) but B leads at threshold 5
        # (50% vs 10%): exactly the reversal the report needs surfaced.
        self.groups = {
            "A": {1: 30, 2: 10, 3: 50, 4: 0, 5: 10},
            "B": {1: 40, 2: 10, 3: 0, 4: 0, 5: 50},
        }

    def test_rank_order_differs_between_thresholds(self):
        self.assertEqual(hta.rank_at(self.groups, 3), ["A", "B"])
        self.assertEqual(hta.rank_at(self.groups, 5), ["B", "A"])

    def test_reversal_is_reported_once_with_both_rates(self):
        changes = hta.order_changes(self.groups, 3, 5)
        self.assertEqual(len(changes["reversals"]), 1)
        record = changes["reversals"][0]
        self.assertEqual({record["a"], record["b"]}, {"A", "B"})
        self.assertEqual(record["a_rate_at_3"], 60.0)
        self.assertEqual(record["b_rate_at_3"], 50.0)
        self.assertEqual(record["a_rate_at_5"], 10.0)
        self.assertEqual(record["b_rate_at_5"], 50.0)

    def test_stable_order_reports_no_reversal(self):
        stable = {
            "A": {1: 50, 2: 0, 3: 0, 4: 0, 5: 50},
            "B": {1: 90, 2: 0, 3: 0, 4: 0, 5: 10},
        }
        changes = hta.order_changes(stable, 3, 5)
        self.assertEqual(changes["reversals"], [])
        self.assertEqual(hta.rank_at(stable, 3), ["A", "B"])
        self.assertEqual(hta.rank_at(stable, 5), ["A", "B"])

    def test_pair_becoming_tied_is_not_counted_as_a_reversal(self):
        groups = {
            "A": {1: 50, 2: 0, 3: 10, 4: 0, 5: 40},
            "B": {1: 60, 2: 0, 3: 0, 4: 0, 5: 40},
        }
        changes = hta.order_changes(groups, 3, 5)
        self.assertEqual(changes["reversals"], [])
        self.assertEqual(len(changes["new_ties"]), 1)


# --------------------------------------------------------------------------
# Guards: read-only and metadata-only
# --------------------------------------------------------------------------

class GuardTests(FixtureCase):
    def test_write_attempt_on_the_tool_connection_fails(self):
        con = hta.open_readonly(self.db)
        self.addCleanup(con.close)
        with self.assertRaises(sqlite3.OperationalError):
            con.execute("CREATE TABLE should_not_exist (a INTEGER)")
        with self.assertRaises(sqlite3.OperationalError):
            con.execute("UPDATE prompts SET severity = 5")
        with self.assertRaises(sqlite3.OperationalError):
            con.execute("DELETE FROM prompts")

    def test_query_only_pragma_is_set(self):
        con = hta.open_readonly(self.db)
        self.addCleanup(con.close)
        self.assertEqual(con.execute("PRAGMA query_only").fetchone()[0], 1)

    def test_database_file_is_unchanged_after_a_full_run(self):
        before = os.stat(self.db)
        self.run_command("--project-root", self.root, "inventory",
                         "--db", self.db, "--out", self.out)
        self.run_command("--project-root", self.root, "thresholds",
                         "--db", self.db, "--out", self.out)
        after = os.stat(self.db)
        self.assertEqual((before.st_size, before.st_mtime), (after.st_size, after.st_mtime))

    def test_write_statements_are_rejected_before_execution(self):
        for sql in ("DELETE FROM prompts",
                    "UPDATE prompts SET severity = 1",
                    "INSERT INTO prompts (id) VALUES (1)",
                    "DROP TABLE prompts",
                    "VACUUM",
                    "SELECT 1; DROP TABLE prompts"):
            with self.subTest(sql=sql):
                with self.assertRaises(hta.UnsafeQuery):
                    hta.assert_read_only_sql(sql)

    def test_free_text_columns_are_rejected_before_execution(self):
        for sql in ("SELECT p.response FROM prompts p",
                    "SELECT p.verdict FROM prompts p",
                    "SELECT p.prompt_text FROM prompts p",
                    "SELECT p.error FROM prompts p",
                    "SELECT t.api_key FROM targets t",
                    "SELECT t.endpoint_url FROM targets t",
                    "SELECT b.llm_config_json FROM benchmarks b",
                    "SELECT p.note FROM prompts p"):
            with self.subTest(sql=sql):
                with self.assertRaises(hta.UnsafeQuery):
                    hta.assert_metadata_only(sql)

    def test_the_emptiness_predicate_is_the_only_permitted_use_of_response(self):
        hta.assert_metadata_only(
            f"SELECT COUNT(*) FROM prompts p WHERE {hta.DELIVERED_SQL}")

    def test_metadata_columns_are_permitted(self):
        hta.assert_metadata_only(
            "SELECT p.id, p.attack_slug, p.severity, p.attempt, p.status, "
            "p.created_at FROM prompts p")
        hta.assert_metadata_only("SELECT t.id, t.name, t.provider, t.model_name FROM targets t")

    def test_no_attempt_group_value_appears_in_any_output(self):
        self.run_command("--project-root", self.root, "attempts",
                         "--db", self.db, "--out", self.out)
        for name in ("attempts.json", "attempts.md"):
            text = Path(self.out, name).read_text(encoding="utf-8")
            self.assertNotIn("grp-a", text)
            self.assertNotIn("grp-b", text)

    def test_no_free_text_appears_in_any_output(self):
        self.run_command("--project-root", self.root, "inventory",
                         "--db", self.db, "--out", self.out)
        self.run_command("--project-root", self.root, "thresholds",
                         "--db", self.db, "--out", self.out)
        for path in Path(self.out).iterdir():
            text = path.read_text(encoding="utf-8")
            for secret in ("SYNTHETIC-PROMPT-TEXT", "SYNTHETIC-RESPONSE-BODY",
                           "NOT-A-REAL-KEY", "example.invalid", "note text"):
                self.assertNotIn(secret, text, f"{secret!r} leaked into {path.name}")


class OverwriteRefusalTests(FixtureCase):
    def test_second_run_into_the_same_directory_is_refused(self):
        self.run_command("--project-root", self.root, "inventory",
                         "--db", self.db, "--out", self.out)
        self.run_command("--project-root", self.root, "inventory",
                         "--db", self.db, "--out", self.out, expected=2)

    def test_existing_file_is_not_modified_by_the_refused_run(self):
        self.run_command("--project-root", self.root, "bounds", "--out", self.out)
        marker = Path(self.out, "bounds.json")
        original = marker.read_text(encoding="utf-8")
        self.run_command("--project-root", self.root, "bounds",
                         "--out", self.out, expected=2)
        self.assertEqual(marker.read_text(encoding="utf-8"), original)

    def test_new_output_raises_file_exists(self):
        os.makedirs(self.out, exist_ok=True)
        Path(self.out, "taken.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            hta.new_output(self.out, "taken.json")


# --------------------------------------------------------------------------
# Catalog parsing and rubric separation
# --------------------------------------------------------------------------

class CatalogTests(FixtureCase):
    def test_catalog_is_read_from_source_without_importing_it(self):
        catalog = hta.load_attack_catalog(self.root)
        self.assertEqual(catalog["syn_leak"]["rubric"], "leak")
        self.assertEqual(catalog["syn_plain"]["rubric"], "harm")
        self.assertEqual(catalog["syn_dynamic"]["kind"], "dynamic")
        self.assertEqual(catalog["syn_encoded"]["catalog_attempts"], 3)
        self.assertNotIn("app.attacks", sys.modules)

    def test_missing_catalog_falls_back_without_raising(self):
        empty = tempfile.TemporaryDirectory(prefix="fyp-hta-empty-")
        self.addCleanup(empty.cleanup)
        catalog = hta.load_attack_catalog(empty.name)
        self.assertEqual(catalog["harmbench_rot13"]["rubric"], "harm")
        self.assertEqual(catalog["sysprompt_extraction"]["rubric"], "leak")
        self.assertEqual(catalog["harmbench_rot13"]["source"], "embedded fallback table")

    def test_static_harm_selection_excludes_leak_and_dynamic(self):
        catalog = hta.load_attack_catalog(self.root)
        self.assertEqual(hta.slugs_for(catalog, "static", "harm"),
                         ["syn_encoded", "syn_plain", "syn_tiny"])
        self.assertEqual(hta.slugs_for(catalog, "static", "leak"), ["syn_leak"])


class RubricSeparationTests(FixtureCase):
    def test_leak_and_dynamic_rows_are_excluded_from_the_harm_analysis(self):
        self.run_command("--project-root", self.root, "thresholds",
                         "--db", self.db, "--out", self.out,
                         "--start", "2026-06-01", "--end", "2026-06-30")
        payload = self.load("thresholds")
        self.assertEqual(payload["rubric"], "harm")
        self.assertEqual(payload["rubric_separation"]["leak_static_slugs_excluded"],
                         ["syn_leak"])
        window = payload["populations"]["historical_window"]
        self.assertNotIn("syn_leak", window["by_attack"])
        self.assertNotIn("syn_dynamic", window["by_attack"])
        # 20 plain + 20 encoded + 3 tiny + the 5 best-of-N rows; the 30
        # leak/dynamic rows and the four excluded rows must not appear.
        self.assertEqual(window["overall_total"], 48)

    def test_excluded_statuses_and_dates_do_not_enter_the_population(self):
        self.run_command("--project-root", self.root, "thresholds",
                         "--db", self.db, "--out", self.out,
                         "--start", "2026-06-01", "--end", "2026-06-30")
        payload = self.load("thresholds")
        window = payload["populations"]["historical_window"]
        plain = window["by_attack"]["syn_plain"]
        self.assertEqual(sum(plain.values()), 20)
        self.assertEqual(plain["5"], 1)
        # The out-of-window row is picked up by the unwindowed population.
        everything = payload["populations"]["all_scored_static_harm"]
        self.assertEqual(sum(everything["by_attack"]["syn_plain"].values()), 21)

    def test_small_condition_is_reported_but_not_ranked(self):
        self.run_command("--project-root", self.root, "thresholds",
                         "--db", self.db, "--out", self.out,
                         "--start", "2026-06-01", "--end", "2026-06-30")
        payload = self.load("thresholds")
        block = payload["populations"]["historical_window"]["analysis_by_attack"]
        too_small = [d["group"] for d in block["too_small_to_rank"]]
        self.assertIn("syn_tiny", too_small)
        for ranking in block["rankings"].values():
            self.assertNotIn("syn_tiny", ranking)

    def test_reproduction_check_reports_a_difference_rather_than_forcing_a_match(self):
        self.run_command("--project-root", self.root, "thresholds",
                         "--db", self.db, "--out", self.out,
                         "--start", "2026-06-01", "--end", "2026-06-30")
        check = self.load("thresholds")["reproduction_check"]
        self.assertFalse(check["reproduced"])
        self.assertEqual(check["reported_total"], hta.REPORT_POOLED_TOTAL)
        self.assertEqual(check["recomputed_total"], 48)
        self.assertEqual(check["total_difference"], 48 - hta.REPORT_POOLED_TOTAL)


class AttemptsTests(FixtureCase):
    def test_group_counts_and_first_success_index(self):
        self.run_command("--project-root", self.root, "attempts",
                         "--db", self.db, "--out", self.out)
        payload = self.load("attempts")
        all_time = payload["all_time"]
        self.assertEqual(all_time["total_groups"], 2)
        self.assertEqual(all_time["multi_attempt_groups"], 2)
        self.assertEqual(all_time["observed_attempt_count_distribution"], {"2": 1, "3": 1})
        # grp-a first reaches severity >= 3 on attempt index 1; grp-b never does.
        self.assertEqual(all_time["first_success_attempt_index_distribution"], {"1": 1})
        encoded = next(e for e in all_time["by_attack"] if e["attack_slug"] == "syn_encoded")
        self.assertEqual(encoded["groups_with_success_at_threshold_3"], 1)
        self.assertEqual(encoded["groups_without_success"], 1)
        self.assertEqual(encoded["catalog_attempts_now"], 3)

    def test_rows_without_a_group_are_counted_separately(self):
        self.run_command("--project-root", self.root, "attempts",
                         "--db", self.db, "--out", self.out)
        payload = self.load("attempts")
        self.assertGreater(payload["delivered_static_rows_without_attempt_group"], 0)


class InventoryTests(FixtureCase):
    def test_inventory_reports_scored_rows_and_rubric_split(self):
        self.run_command("--project-root", self.root, "inventory",
                         "--db", self.db, "--out", self.out)
        payload = self.load("inventory")
        self.assertEqual(payload["row_counts"]["prompts"], 82)
        self.assertEqual(payload["scored_rows_by_rubric"]["leak"], 20)
        self.assertEqual(payload["scored_rows_by_rubric"]["harm"], 59)
        self.assertEqual(payload["objective_ids"]["rows_with_objective_id"], 0)
        self.assertNotIn("response", payload["columns_read_by_this_tool"]["targets"])


# --------------------------------------------------------------------------
# Bounds solver
# --------------------------------------------------------------------------

class BoundsSolverTests(unittest.TestCase):
    def test_mean_sum_range_is_exact(self):
        # 20 responses with a mean published as 3.65: the sum must be 73.
        self.assertEqual(hta.mean_sum_range(20, "3.65"), (73, 73))
        # 153 responses with a mean published as 3.37: 514.845 to 516.375.
        self.assertEqual(hta.mean_sum_range(153, "3.37"), (515, 516))

    def test_codechameleon_at_threshold_5_is_at_most_8_of_20(self):
        with_caps = hta.feasible_counts(20, 18, "3.65", hta.REPORT_TABLE_10)
        without = hta.feasible_counts(20, 18, "3.65", None)
        self.assertTrue(with_caps["feasible"])
        self.assertEqual(with_caps["t5"]["max"], 8)
        self.assertEqual(without["t5"]["max"], 8)

    def test_rot13_at_threshold_5_is_at_least_70_with_pooled_caps(self):
        result = hta.feasible_counts(153, 91, "3.37", hta.REPORT_TABLE_10)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["t5"]["min"], 70)

    def test_rot13_at_threshold_5_is_at_least_27_without_pooled_caps(self):
        result = hta.feasible_counts(153, 91, "3.37", None)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["t5"]["min"], 27)

    def test_pooled_caps_never_widen_a_range(self):
        for label, _slug, n, k3, mean in hta.REPORT_TABLE_9:
            with self.subTest(condition=label):
                capped = hta.feasible_counts(n, k3, mean, hta.REPORT_TABLE_10)
                free = hta.feasible_counts(n, k3, mean, None)
                self.assertTrue(capped["feasible"])
                self.assertTrue(free["feasible"])
                for t in (2, 4, 5):
                    self.assertGreaterEqual(capped[f"t{t}"]["min"], free[f"t{t}"]["min"])
                    self.assertLessEqual(capped[f"t{t}"]["max"], free[f"t{t}"]["max"])

    def test_every_solution_respects_the_published_aggregates(self):
        # Brute-force cross-check on a small case: enumerate all histograms and
        # confirm the solver's range matches an independent enumeration.
        n, k3, mean = 20, 18, "3.65"
        s_min, s_max = hta.mean_sum_range(n, mean)
        found = []
        for c1 in range(n + 1):
            for c2 in range(n + 1 - c1):
                for c3 in range(n + 1 - c1 - c2):
                    for c4 in range(n + 1 - c1 - c2 - c3):
                        c5 = n - c1 - c2 - c3 - c4
                        if c3 + c4 + c5 != k3:
                            continue
                        total = c1 + 2 * c2 + 3 * c3 + 4 * c4 + 5 * c5
                        if s_min <= total <= s_max:
                            found.append((c1, c2, c3, c4, c5))
        self.assertTrue(found)
        result = hta.feasible_counts(n, k3, mean, None)
        self.assertEqual(result["t5"]["min"], min(f[4] for f in found))
        self.assertEqual(result["t5"]["max"], max(f[4] for f in found))
        self.assertEqual(result["t4"]["min"], min(f[3] + f[4] for f in found))
        self.assertEqual(result["t4"]["max"], max(f[3] + f[4] for f in found))
        self.assertEqual(result["t2"]["min"], min(n - f[0] for f in found))
        self.assertEqual(result["t2"]["max"], max(n - f[0] for f in found))

    def test_infeasible_input_is_reported_not_guessed(self):
        # A mean of 5.00 cannot hold with 10 of 20 responses below severity 3.
        result = hta.feasible_counts(20, 10, "5.00", None)
        self.assertFalse(result["feasible"])
        self.assertIsNone(result["t5"]["min"])


class BoundsCommandTests(FixtureCase):
    def test_bounds_command_needs_no_database_and_passes_its_own_checks(self):
        self.run_command("bounds", "--out", self.out)
        payload = self.load("bounds")
        self.assertTrue(all(c["agrees"]
                            for c in payload["independent_hand_calculation_checks"]))
        self.assertEqual(len(payload["independent_hand_calculation_checks"]), 4)

    def test_orderings_are_reported_with_and_without_the_pooled_assumption(self):
        self.run_command("bounds", "--out", self.out)
        payload = self.load("bounds")
        for variant in ("with_pooled_caps_orderings", "without_pooled_caps_orderings"):
            for threshold in ("2", "4", "5"):
                block = payload[variant][threshold]
                self.assertEqual(block["pairs"], 28)  # 8 conditions choose 2
                self.assertEqual(len(block["determined"]) + len(block["undetermined"]), 28)


if __name__ == "__main__":
    unittest.main()
