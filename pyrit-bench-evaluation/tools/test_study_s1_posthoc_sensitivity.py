"""Synthetic tests for study_s1_posthoc_sensitivity.py (no network, no real data, no model calls).

Run: python3 -B -m unittest tools/test_study_s1_posthoc_sensitivity.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import study_s1_external_reference as ext  # noqa: E402
import study_s1_posthoc_sensitivity as tool  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# 12 main rows over 3 objectives (4 rows each); reference-positive rows: 1, 2, 5, 9 (4 of 12).
MEDIANS = [1.0, 0.5, 0.0, 0.25, 0.75, 0.0, 0.0, 0.0, 1.0, 0.0, 0.25, 0.0]


def make_prep(root: Path) -> Path:
    prep = root / "prep"
    rows, refs = [], []
    for k, med in enumerate(MEDIANS, start=1):
        rid, obj = f"SRH-{k:04d}", f"SRH-OBJ-00{(k - 1) // 4 + 1}"
        rows.append({"id": rid, "objective_id": obj, "model": "m1", "method": "AIM" if k % 2 else "base64",
                     "surface": "plain" if k % 2 else "encoded", "split": "main", "in_j2_subsample": False})
        refs.append({"id": rid, "objective_id": obj, "median_0_1": med})
    rows.append({"id": "SRH-0099", "objective_id": "SRH-OBJ-009", "model": "m1", "method": "AIM",
                 "surface": "plain", "split": "development", "in_j2_subsample": False})
    refs.append({"id": "SRH-0099", "objective_id": "SRH-OBJ-009", "median_0_1": 1.0})
    write_jsonl(prep / "population.jsonl", rows)
    write_jsonl(prep / "private" / "external_reference.jsonl", refs)
    return prep


def ids() -> list[str]:
    return [f"SRH-{k:04d}" for k in range(1, len(MEDIANS) + 1)]


def two_label_rows(wrong=(), abstain=(), invalid=(), missing=()) -> list[dict]:
    out = []
    for i, med in zip(ids(), MEDIANS):
        if i in missing:
            continue
        if i in invalid:
            out.append({"id": i, "valid": False})
            continue
        if i in abstain:
            out.append({"id": i, "objective_pred": "Unclear", "fulfilment_pred": "No"})
            continue
        pos = (med >= 0.5) != (i in wrong)
        out.append({"id": i, "objective_pred": "Harmful", "fulfilment_pred": "Partial" if pos else "No"})
    return out


def severity_rows(wrong=()) -> list[dict]:
    return [{"id": i, "severity": (4 if (med >= 0.5) != (i in wrong) else 1)} for i, med in zip(ids(), MEDIANS)]


class StatisticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.prep = make_prep(self.tmp)

    def load(self, configs: dict[str, list[dict]], kinds=None) -> dict:
        paths = {}
        for name, rows in configs.items():
            paths[name] = self.tmp / f"{name}.jsonl"
            write_jsonl(paths[name], rows)
        return tool.load(self.prep, "main", paths, kinds or {})

    def test_development_rows_excluded_and_labels_reused(self):
        d = self.load({"S": severity_rows()}, {"S": "severity"})
        self.assertEqual(len(d["ids"]), 12)
        self.assertEqual(sum(d["ref"].values()), 4)          # median 0.5 is positive (primary cut)
        self.assertEqual(d["objectives"], 3)
        # the state comes from the imported analysis_main_03 mapping
        self.assertEqual(d["states"]["S"]["SRH-0001"], ext.prediction_state({"id": "SRH-0001", "severity": 4}, "severity"))

    def test_always_negative(self):
        d = self.load({"S": severity_rows()}, {"S": "severity"})
        a = tool.always_negative(d["ids"], d["ref"])
        self.assertEqual((a["matches"], a["n"], a["agreement"]), (8, 12, 0.6667))

    def test_balanced_accuracy_counts_invalid_and_abstain_as_incorrect(self):
        # SRH-0001 (positive) invalid, SRH-0003 (negative) abstains, SRH-0004 (negative) wrong, SRH-0005 missing (positive)
        d = self.load({"J": two_label_rows(wrong=("SRH-0004",), abstain=("SRH-0003",), invalid=("SRH-0001",),
                                           missing=("SRH-0005",))})
        b = tool.balanced_accuracy(d["states"]["J"], d["ref"], d["ids"])
        self.assertEqual((b["true_positive"], b["true_negative"]), (2, 6))
        self.assertEqual(b["sensitivity"], 0.5)               # 2 / 4 positives
        self.assertEqual(b["specificity"], 0.75)              # 6 / 8 negatives
        self.assertEqual(b["balanced_accuracy"], 0.625)
        u = b["usable_only"]
        self.assertEqual((u["reference_positive"], u["reference_negative"]), (2, 7))
        self.assertEqual(u["sensitivity"], 1.0)
        self.assertEqual(u["specificity"], round(6 / 7, 4))

    def test_both_usable(self):
        d = self.load({"A": two_label_rows(invalid=("SRH-0001", "SRH-0002"), abstain=("SRH-0003",)),
                       "B": severity_rows(wrong=("SRH-0004", "SRH-0001"))}, {"B": "severity"})
        r = tool.both_usable(d["states"]["A"], d["states"]["B"], d["ref"], d["ids"], "A", "B")
        self.assertEqual(r["rows"], 9)
        self.assertEqual(r["A"]["matches"], 9)
        self.assertEqual(r["B"]["matches"], 8)                # only SRH-0004 wrong among the 9
        self.assertEqual(r["difference"], round(1 / 9, 4))

    def test_paired_bootstrap_identical_judges_is_zero(self):
        c = {i: (k % 3 != 0) for k, i in enumerate(ids())}
        cl = {i: f"O{k // 4}" for k, i in enumerate(ids())}
        r = tool.paired_cluster_bootstrap(c, dict(c), cl, 500, "s")
        self.assertEqual(r["estimate"], 0.0)
        self.assertEqual(r["interval_95"], [0.0, 0.0])
        self.assertTrue(r["interval_includes_zero"])

    def test_paired_bootstrap_estimate_sign_and_determinism(self):
        a = {i: True for i in ids()}
        b = {i: (k >= 4) for k, i in enumerate(ids())}        # B wrong on the whole first objective
        cl = {i: f"O{k // 4}" for k, i in enumerate(ids())}
        r1 = tool.paired_cluster_bootstrap(a, b, cl, 1000, "seed-x")
        r2 = tool.paired_cluster_bootstrap(a, b, cl, 1000, "seed-x")
        self.assertEqual(r1, r2)
        self.assertEqual((r1["a_correct"], r1["b_correct"], r1["estimate"]), (12, 8, 0.3333))
        lo, hi = r1["interval_95"]
        self.assertGreaterEqual(lo, 0.0)                      # a resample can miss objective 1 entirely
        self.assertLessEqual(hi, 1.0)
        self.assertLessEqual(lo, r1["estimate"])
        self.assertGreaterEqual(hi, r1["estimate"])
        self.assertEqual(r1["clusters"], 3)

    def test_paired_bootstrap_resamples_whole_clusters(self):
        # every cluster has the same within-cluster difference, so every resample gives the same value
        a = {i: (k % 4 < 2) for k, i in enumerate(ids())}
        b = {i: (k % 4 < 1) for k, i in enumerate(ids())}
        cl = {i: f"O{k // 4}" for k, i in enumerate(ids())}
        r = tool.paired_cluster_bootstrap(a, b, cl, 300, "z")
        self.assertEqual(r["interval_95"], [0.25, 0.25])

    def test_unknown_configuration_rejected(self):
        write_jsonl(self.tmp / "a.jsonl", two_label_rows())
        with self.assertRaises(ValueError):
            tool.analyse(self.prep, "main", {"A": self.tmp / "a.jsonl"}, {}, ["Z"], [], [], 10, "s")


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.prep = make_prep(self.tmp)
        write_jsonl(self.tmp / "J0.jsonl", severity_rows(wrong=("SRH-0003", "SRH-0007")))
        write_jsonl(self.tmp / "J1.jsonl", two_label_rows(wrong=("SRH-0002",), invalid=("SRH-0010",)))

    def argv(self, out: str, extra=()) -> list[str]:
        return ["run", "--prep-dir", str(self.prep), "--split", "main",
                "--predictions", f"J0={self.tmp / 'J0.jsonl'}", f"J1={self.tmp / 'J1.jsonl'}",
                "--balanced", "J0", "J1", "--both-usable", "J1:J0", "--differences", "J1:J0",
                "--bootstrap", "400", "--out-dir", str(self.tmp / out), *extra]

    def test_end_to_end_outputs_and_reproducibility(self):
        self.assertEqual(tool.main(self.argv("o1", ["--mc-check", "3"])), 0)
        self.assertEqual(tool.main(self.argv("o2", ["--mc-check", "3"])), 0)
        r1 = json.loads((self.tmp / "o1" / "results.json").read_text())
        r2 = json.loads((self.tmp / "o2" / "results.json").read_text())
        self.assertEqual(r1["results"], r2["results"])
        self.assertEqual(r1["results"]["agreement"]["J0"]["matches"], 10)
        self.assertEqual(r1["results"]["agreement"]["J1"]["matches"], 10)
        self.assertEqual(r1["results"]["differences"]["J1:J0"]["estimate"], 0.0)
        self.assertIn("monte_carlo_check", r1["results"]["differences"]["J1:J0"])
        self.assertEqual(r1["inputs"]["J0"]["kind"], "severity")   # default kind, as in the analysis tool
        self.assertEqual(r1["inputs"]["J0"]["sha256"], ext.sha256_file(self.tmp / "J0.jsonl"))
        readme = (self.tmp / "o1" / "README.md").read_text()
        self.assertIn("POST-REVIEW SENSITIVITY ANALYSIS", readme)
        self.assertIn("study_s1_posthoc_sensitivity.py run", readme)
        self.assertNotIn("scratchpad_comparison", r1)               # opt-in only
        self.assertEqual(tool.main(self.argv("o5", ["--compare-scratchpad", "--mc-check", "2"])), 0)
        r5 = json.loads((self.tmp / "o5" / "results.json").read_text())
        items = {r["item"]: r for r in r5["scratchpad_comparison"]}
        self.assertFalse(items["always_negative.matches"]["match_3dp"])  # synthetic data: 8, not 751
        self.assertIn("scratchpad_within_monte_carlo_range", items["differences.J1:J0.lower"])
        self.assertEqual(tool.main(self.argv("o6", ["--mc-check", "2", "--mc-resamples", "100", "200"])), 0)
        r6 = json.loads((self.tmp / "o6" / "results.json").read_text())
        self.assertEqual([m["resamples"] for m in r6["results"]["differences"]["J1:J0"]["monte_carlo_check"]], [100, 200])

    def test_refuses_existing_out_dir(self):
        (self.tmp / "exists").mkdir()
        self.assertEqual(tool.main(self.argv("exists")), 2)

    def test_check_against_analysis_json(self):
        rc = ext.main(["run", "--prep-dir", str(self.prep), "--split", "main", "--predictions",
                       f"J0={self.tmp / 'J0.jsonl'}", f"J1={self.tmp / 'J1.jsonl'}", "--contrasts", "J1:J0",
                       "--bootstrap", "50", "--out-dir", str(self.tmp / "ana")])
        self.assertEqual(rc, 0)
        self.assertEqual(tool.main(self.argv("o3", ["--check-against", str(self.tmp / "ana" / "analysis.json")])), 0)
        r = json.loads((self.tmp / "o3" / "results.json").read_text())
        self.assertTrue(r["check_against"]["all_pass"])
        # a different prediction file must fail the cross-check
        write_jsonl(self.tmp / "J1.jsonl", two_label_rows())
        self.assertEqual(tool.main(self.argv("o4", ["--check-against", str(self.tmp / "ana" / "analysis.json")])), 0)
        r = json.loads((self.tmp / "o4" / "results.json").read_text())
        self.assertFalse(r["check_against"]["all_pass"])


if __name__ == "__main__":
    unittest.main()
