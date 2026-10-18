"""Offline tests for the Study S1-R2 repeat-stability tool (synthetic data; no model calls)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_s1_repeat_stability as stab  # noqa: E402


def two(obj, ful, valid=True, rid="R1"):
    return {"id": rid, "objective_pred": obj, "fulfilment_pred": ful, "valid": valid}


class RepeatStabilityTests(unittest.TestCase):
    def test_two_label_exact_cut_and_validity(self):
        primary = [two("Harmful", "Partial", rid="A"), two("Harmful", "No", rid="B"), two("Harmful", "Complete", rid="C"),
                   two("Harmful", "No", rid="D"), two("Benign", "No", rid="E")]
        repeat = [two("Harmful", "Partial", rid="A"), two("Harmful", "Partial", rid="B"), two("Harmful", "Partial", rid="C"),
                  two(None, None, valid=False, rid="D"), two("Benign", "No", rid="E")]
        r = stab.stability(primary, repeat, "two_label")
        self.assertEqual((r["rows"], r["both_valid"], r["validity_changed"]), (5, 4, 1))
        self.assertEqual(r["exact_agreement"], 2)          # A, E
        self.assertEqual(r["primary_cut_agreement"], 3)    # A, C (both positive), E
        self.assertEqual(r["primary_cut_agreement_rate"], 0.75)

    def test_severity_and_missing_primary(self):
        sev = lambda rid, s: {"id": rid, "severity": s, "valid": True}
        r = stab.stability([sev("A", 3), sev("B", 1)], [sev("A", 4), sev("B", 1)], "severity")
        self.assertEqual((r["exact_agreement"], r["primary_cut_agreement"]), (1, 2))
        with self.assertRaises(ValueError):
            stab.stability([sev("A", 3)], [sev("A", 3), sev("Z", 1)], "severity")

    def test_cli_writes_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, q = root / "p.jsonl", root / "q.jsonl"
            p.write_text(json.dumps(two("Harmful", "No")) + "\n", encoding="utf-8")
            q.write_text(json.dumps(two("Harmful", "No")) + "\n", encoding="utf-8")
            out = root / "out.json"
            args = ["--pair", f"J1={p},{q}", "--kinds", "J1=two_label", "--out", str(out)]
            self.assertEqual(stab.main(args), 0)
            self.assertEqual(json.loads(out.read_text())["configurations"]["J1"]["exact_agreement_rate"], 1.0)
            self.assertEqual(stab.main(args), 2)


if __name__ == "__main__":
    unittest.main()
