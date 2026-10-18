"""Synthetic tests for study_s1_external_reference.py (no network, no real data)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import study_s1_external_reference as tool  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class Fixture:
    """18 main rows over 3 objectives, 3 methods and 2 models; medians chosen so that 7 are reference-positive."""

    def __init__(self, tmp: Path):
        self.prep = tmp / "prep"
        rows, refs = [], []
        medians = [0.0, 1.0, 0.5, 0.0, 0.25, 1.0, 0.0, 0.75, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.25]
        methods = ["base64", "AIM", "poems"]
        k = 0
        for o in range(3):
            for m in methods:
                for model in ("m1", "m2"):
                    k += 1
                    rid = f"SRH-{k:04d}"
                    rows.append({"id": rid, "objective_id": f"SRH-OBJ-00{o + 1}", "model": model, "method": m,
                                 "surface": "encoded" if m == "base64" else "plain", "split": "main",
                                 "in_j2_subsample": k <= 6, "in_repeat_set": False, "source_row_index": k})
                    refs.append({"id": rid, "objective_id": f"SRH-OBJ-00{o + 1}", "labels_0_1": [medians[k - 1]] * 5,
                                 "median_0_1": medians[k - 1], "median_bin": str(medians[k - 1]), "n_labels": 5})
        # one development row, must be ignored in the main split
        rows.append({"id": "SRH-0099", "objective_id": "SRH-OBJ-009", "model": "m1", "method": "AIM", "surface": "plain",
                     "split": "development", "in_j2_subsample": False, "in_repeat_set": False, "source_row_index": 99})
        refs.append({"id": "SRH-0099", "objective_id": "SRH-OBJ-009", "labels_0_1": [1.0] * 5, "median_0_1": 1.0,
                     "median_bin": "1.0", "n_labels": 5})
        write_jsonl(self.prep / "population.jsonl", rows)
        write_jsonl(self.prep / "private" / "external_reference.jsonl", refs)
        self.rows, self.medians = rows[:-1], medians

    def perfect_severity(self, path: Path) -> None:
        write_jsonl(path, [{"id": r["id"], "severity": int(round(self.medians[i] * 4 + 1)), "valid": True}
                           for i, r in enumerate(self.rows)])

    def two_label(self, path: Path, wrong_ids=(), abstain_ids=(), missing_ids=()) -> None:
        out = []
        for i, r in enumerate(self.rows):
            if r["id"] in missing_ids:
                continue
            pos = self.medians[i] >= 0.5
            if r["id"] in wrong_ids:
                pos = not pos
            if r["id"] in abstain_ids:
                out.append({"id": r["id"], "objective_pred": "Harmful", "fulfilment_pred": "Cannot assess"})
            else:
                out.append({"id": r["id"], "objective_pred": "Harmful",
                            "fulfilment_pred": "Complete" if pos and self.medians[i] == 1.0 else ("Partial" if pos else "No")})
        write_jsonl(path, out)


class ExternalReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fx = Fixture(self.tmp)

    def run_tool(self, out: str, preds: list[str], extra: list[str] | None = None) -> dict:
        argv = ["run", "--prep-dir", str(self.fx.prep), "--split", "main", "--predictions", *preds,
                "--out-dir", str(self.tmp / out), "--bootstrap", "200"] + (extra or [])
        self.assertEqual(tool.main(argv), 0)
        return json.loads((self.tmp / out / "analysis.json").read_text())

    def test_perfect_severity_judge(self):
        self.fx.perfect_severity(self.tmp / "j0.jsonl")
        a = self.run_tool("a", [f"J0={self.tmp / 'j0.jsonl'}"])
        c = a["configurations"]["J0"]
        self.assertEqual(a["rows"], 18)
        self.assertEqual(a["reference_positive_primary"], 7)
        self.assertEqual(c["cuts"]["primary"]["agreement_over_population"], 1.0)
        self.assertEqual(c["cuts"]["strict"]["agreement_over_population"], 1.0)
        self.assertEqual(c["spearman_severity_vs_median"], 1.0)
        self.assertEqual(c["cuts"]["primary"]["coverage"], 1.0)
        self.assertEqual(sum(sum(v.values()) for v in c["confusion_rating_rows_severity_cols"].values()), 18)

    def test_two_label_errors_abstentions_and_missing(self):
        wrong, abstain, missing = ("SRH-0001",), ("SRH-0002",), ("SRH-0003",)
        self.fx.two_label(self.tmp / "j1.jsonl", wrong, abstain, missing)
        a = self.run_tool("b", [f"J1={self.tmp / 'j1.jsonl'}"])
        c = a["configurations"]["J1"]
        self.assertEqual(c["status_counts"], {"valid": 16, "abstain": 1, "missing": 1})
        p = c["cuts"]["primary"]
        self.assertEqual(p["matches"], 15)          # 18 - wrong - abstain - missing
        self.assertEqual(p["valid_outputs"], 16)
        self.assertEqual(p["agreement_over_population"], 0.8333)
        self.assertEqual(p["agreement_over_valid_outputs"], 0.9375)
        self.assertIsNotNone(p["agreement_cluster_bootstrap"])
        self.assertIn("encoded", p["by_surface"])

    def test_contrast_mcnemar_and_common_rows(self):
        self.fx.perfect_severity(self.tmp / "j0.jsonl")
        self.fx.two_label(self.tmp / "j1.jsonl", wrong_ids=("SRH-0001", "SRH-0004", "SRH-0005"))
        a = self.run_tool("c", [f"J0={self.tmp / 'j0.jsonl'}", f"J1={self.tmp / 'j1.jsonl'}"], ["--contrasts", "J1:J0"])
        r = a["contrasts"][0]
        self.assertEqual((r["primary"], r["baseline"], r["common_rows"]), ("J1", "J0", 18))
        self.assertEqual(r["baseline_only_correct"], 3)
        self.assertEqual(r["primary_only_correct"], 0)
        self.assertEqual(r["discordant"], 3)
        self.assertAlmostEqual(r["p_value"], 0.25)

    def test_subsample_restriction(self):
        self.fx.two_label(self.tmp / "j2.jsonl")
        a = self.run_tool("d", [f"J2={self.tmp / 'j2.jsonl'}"], ["--subset", "J2=j2_subsample"])
        self.assertEqual(a["configurations"]["J2"]["n"], 6)
        self.assertEqual(a["configurations"]["J2"]["population"], "j2_subsample")
        self.assertNotIn("J2:partial_or_complete", a["rq3"]["definitions"])

    def test_refusal_floor_mapping(self):
        rows = [{"id": r["id"], "surface": "refusal" if self.fx.medians[i] < 0.5 else "non_refusal",
                 "predicted_fulfilment": "No" if self.fx.medians[i] < 0.5 else "Partial_or_Complete"}
                for i, r in enumerate(self.fx.rows)]
        write_jsonl(self.tmp / "b0.jsonl", rows)
        a = self.run_tool("e", [f"B0={self.tmp / 'b0.jsonl'}"])
        self.assertEqual(a["configurations"]["B0"]["kind"], "refusal_floor")
        self.assertEqual(a["configurations"]["B0"]["cuts"]["primary"]["agreement_over_population"], 1.0)
        self.assertIn("B0:non_refusal", a["rq3"]["definitions"])

    def test_rq3_orderings_and_reversals(self):
        self.fx.perfect_severity(self.tmp / "j0.jsonl")
        a = self.run_tool("f", [f"J0={self.tmp / 'j0.jsonl'}"])
        d = a["rq3"]["definitions"]["J0:severity_ge3"]
        self.assertEqual(d["kendall_tau_b_vs_human_primary"], 1.0)
        self.assertEqual(d["ordering"], a["rq3"]["human_ordering_primary_cut"])
        rev = a["rq3"]["J0:pairs_reversed_between_ge3_and_ge5"]
        self.assertEqual(rev["of_pairs"], 3)
        self.assertIsInstance(rev["count"], int)

    def test_kendall_and_spearman_helpers(self):
        self.assertEqual(tool.kendall_tau_b([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertEqual(tool.kendall_tau_b([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertEqual(tool.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertIsNone(tool.spearman([1, 1, 1], [1, 2, 3]))

    def test_determinism_and_overwrite_refusal(self):
        self.fx.two_label(self.tmp / "j1.jsonl", wrong_ids=("SRH-0006",))
        a = self.run_tool("g1", [f"J1={self.tmp / 'j1.jsonl'}"])
        b = self.run_tool("g2", [f"J1={self.tmp / 'j1.jsonl'}"])
        self.assertEqual(a["configurations"]["J1"]["cuts"]["primary"]["agreement_cluster_bootstrap"],
                         b["configurations"]["J1"]["cuts"]["primary"]["agreement_cluster_bootstrap"])
        rc = tool.main(["run", "--prep-dir", str(self.fx.prep), "--predictions", f"J1={self.tmp / 'j1.jsonl'}",
                        "--out-dir", str(self.tmp / "g1")])
        self.assertEqual(rc, 2)

    def test_repeat_rows_are_rejected(self):
        write_jsonl(self.tmp / "rep.jsonl", [{"id": "SRH-0001", "severity": 3, "repeat_index": 1}])
        rc = tool.main(["run", "--prep-dir", str(self.fx.prep), "--predictions", f"J0={self.tmp / 'rep.jsonl'}",
                        "--out-dir", str(self.tmp / "h")])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
