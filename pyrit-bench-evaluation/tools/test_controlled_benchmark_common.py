"""Offline saved-result pointer checks; no live capture or judging."""
import json
from pathlib import Path
import tempfile
import unittest
import controlled_benchmark_common as common

class PointerReaderTests(unittest.TestCase):
    def test_final_pointer_and_unfinished_judge(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / "plan.json").write_text(json.dumps({"judges": {"J0": {}}}))
            with self.assertRaises(ValueError):
                common.final_normalised_dirs(base)
            (base / "final").mkdir()
            expected = base / "normalised"
            (base / "final/J0.json").write_text(json.dumps({"normalised_dir": str(expected)}))
            self.assertEqual(common.final_normalised_dirs(base), {"J0": expected})

    def test_missing_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                common.load_plan(Path(temp))

if __name__ == "__main__":
    unittest.main()
