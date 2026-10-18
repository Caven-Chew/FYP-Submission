"""Offline fixture tests for the Study S1-R2 semantic-review tally."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_s1_semantic_tally as tool  # noqa: E402

ROW = """### {n}. {id} ({group})

- J1 decisions shown here

| Finding (supported / issue found / unresolved) | Evidence lines (e.g. L003) | Note |
|---|---|---|
| {finding} | {evidence} | {note} |
"""


class TallyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.prep = root / "prep"
        (self.prep / "private").mkdir(parents=True)
        (self.prep / "private" / "external_reference.jsonl").write_text(
            '{"id": "A", "median_0_1": 0.0}\n{"id": "B", "median_0_1": 1.0}\n', encoding="utf-8")
        self.preds = root / "J1.jsonl"
        self.preds.write_text(
            '{"id": "A", "valid": true, "objective_pred": "Harmful", "fulfilment_pred": "Partial"}\n'
            '{"id": "B", "valid": true, "objective_pred": "Harmful", "fulfilment_pred": "Complete"}\n', encoding="utf-8")
        self.sample = root / "sample.json"
        self.sample.write_text(json.dumps({
            "predictions_sha256": hashlib.sha256(self.preds.read_bytes()).hexdigest(),
            "sample": {"disagreement": {"ids": ["A"]}, "agreement": {"ids": ["B"]}}}), encoding="utf-8")
        self.dir = root / "packet"
        (self.dir / "responses_numbered").mkdir(parents=True)
        for item in ("A", "B"):
            (self.dir / "responses_numbered" / f"{item}.txt").write_text("L001 | text\nL002 | more\n", encoding="utf-8")
        self.packet = self.dir / "J1_Semantic_Review.md"
        self.rows = [dict(n=1, id="A", group="disagreement", finding="issue found", evidence="L001", note="Partial is wrong."),
                     dict(n=2, id="B", group="agreement", finding="supported", evidence="L001, L002", note="")]
        self.reviewer = "Reviewer: Tester · Review completed at (date and time): 2 Oct 2026 00:41 SGT"

    def tearDown(self):
        self.tmp.cleanup()

    def write(self):
        body = "# Packet\n\n" + self.reviewer + "\n\n" + "\n".join(ROW.format(**r) for r in self.rows)
        self.packet.write_text(body, encoding="utf-8")

    def run_tally(self):
        self.write()
        return tool.tally(self.packet, self.sample, self.prep, self.preds)

    def test_counts_directions_and_verbatim_notes(self):
        result = self.run_tally()
        self.assertEqual(result["reviewer"]["name"], "Tester")
        self.assertEqual(result["counts"]["all"], {"supported": 1, "issue found": 1, "unresolved": 0})
        self.assertEqual(result["rows"][0]["direction"], "J1 positive, reference negative")
        self.assertEqual(result["rows"][1]["direction"], "both positive")
        self.assertEqual(result["rows"][0]["note"], "Partial is wrong.")
        self.assertEqual(result["rows"][1]["evidence_lines"], ["L001", "L002"])

    def test_unfilled_reviewer_line_is_refused(self):
        self.reviewer = "Reviewer: ______ · Review completed at (date and time): ______"
        with self.assertRaisesRegex(ValueError, "reviewer line"):
            self.run_tally()

    def test_unknown_finding_is_refused(self):
        self.rows[1]["finding"] = "looks fine"
        with self.assertRaisesRegex(ValueError, "not one of"):
            self.run_tally()

    def test_evidence_line_must_exist_in_response(self):
        self.rows[0]["evidence"] = "L009"
        with self.assertRaisesRegex(ValueError, "not in the response"):
            self.run_tally()

    def test_issue_needs_a_note(self):
        self.rows[0]["note"] = ""
        with self.assertRaisesRegex(ValueError, "note is required"):
            self.run_tally()

    def test_rows_must_match_sample_order(self):
        self.rows.reverse()
        with self.assertRaisesRegex(ValueError, "do not match"):
            self.run_tally()

    def test_existing_output_is_not_overwritten(self):
        self.write()
        out = self.dir / "tally.json"
        out.write_text("{}", encoding="utf-8")
        argv = ["--packet", str(self.packet), "--sample", str(self.sample), "--prep-dir", str(self.prep),
                "--predictions", str(self.preds), "--out", str(out)]
        self.assertEqual(tool.main(argv), 2)
        self.assertEqual(out.read_text(encoding="utf-8"), "{}")


if __name__ == "__main__":
    unittest.main()
