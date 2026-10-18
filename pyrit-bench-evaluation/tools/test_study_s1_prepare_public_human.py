"""Synthetic tests for study_s1_prepare_public_human.py (no network, no real data)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import study_s1_prepare_public_human as tool  # noqa: E402

COLUMNS = ["forbidden_prompt", "jailbreak", "label_0", "label_1", "label_2", "label_3", "label_4",
           "model", "response", "label_median"]


def make_rows(n_obj: int = 8, methods=("base64", "AIM", "rot13", "poems"), models=("m1", "m2")) -> list[dict]:
    rows = []
    k = 0
    for o in range(n_obj):
        for meth in methods:
            for mod in models:
                k += 1
                labels = [str(0.25 * ((k + i) % 5)) for i in range(5)]
                if k % 7 == 0:          # a row with a missing label
                    labels[4] = ""
                vals = [float(v) for v in labels if v != ""]
                vals.sort()
                median = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
                rows.append({"forbidden_prompt": f"objective number {o}", "jailbreak": meth,
                             "label_0": labels[0], "label_1": labels[1], "label_2": labels[2],
                             "label_3": labels[3], "label_4": labels[4], "model": mod,
                             "response": f"response text {k} " * (k % 3 + 1), "label_median": str(median)})
    return rows


def write_csv(path: Path, rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    path.write_text(buf.getvalue(), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src = self.tmp / "human.csv"
        self.rows = make_rows()
        self.digest = write_csv(self.src, self.rows)
        self.cat = self.tmp / "dataset.csv"
        self.cat.write_text("category,source,forbidden_prompt\nCat A,custom,objective number 0\nCat B,custom,objective number 1\n", encoding="utf-8")

    def run_prepare(self, out: str = "out", **kw) -> Path:
        out_dir = self.tmp / out
        argv = ["prepare", "--source-csv", str(self.src), "--expected-sha256", self.digest,
                "--category-csv", str(self.cat), "--out-dir", str(out_dir),
                "--dev-objectives", str(kw.get("dev", 2)), "--subsample", str(kw.get("sub", 10)),
                "--repeat", str(kw.get("rep", 4))]
        if "seed" in kw:
            argv += ["--seed", kw["seed"]]
        self.assertEqual(tool.main(argv), 0)
        return out_dir

    def test_hash_mismatch_refuses(self):
        rc = tool.main(["prepare", "--source-csv", str(self.src), "--expected-sha256", "0" * 64,
                        "--out-dir", str(self.tmp / "x")])
        self.assertEqual(rc, 2)
        self.assertFalse((self.tmp / "x").exists())

    def test_population_rule_and_counts(self):
        out = self.run_prepare()
        manifest = json.loads((out / "preparation_manifest.json").read_text())
        n_missing = sum(1 for r in self.rows if r["label_4"] == "")
        self.assertEqual(manifest["counts"]["included_rows"], len(self.rows) - n_missing)
        self.assertEqual(manifest["counts"]["excluded_rows_by_label_count"], {"4": n_missing})
        self.assertEqual(manifest["counts"]["source_rows"], len(self.rows))
        pop = [json.loads(x) for x in (out / "population.jsonl").read_text().splitlines()]
        self.assertEqual(len(pop), len(self.rows) - n_missing)
        self.assertTrue(all("labels" not in p and "median" not in p and "response" not in p for p in pop))

    def test_split_is_by_objective_and_deterministic(self):
        a = self.run_prepare("a")
        b = self.run_prepare("b")
        pa = (a / "population.jsonl").read_text()
        self.assertEqual(pa, (b / "population.jsonl").read_text())
        pop = [json.loads(x) for x in pa.splitlines()]
        split_by_obj = {}
        for p in pop:
            split_by_obj.setdefault(p["objective_id"], set()).add(p["split"])
        self.assertTrue(all(len(s) == 1 for s in split_by_obj.values()))
        self.assertEqual(sum(1 for s in split_by_obj.values() if s == {"development"}), 2)
        c = self.run_prepare("c", seed="another-seed")
        self.assertNotEqual(pa, (c / "population.jsonl").read_text())

    def test_subsample_and_repeats_come_from_main(self):
        out = self.run_prepare()
        pop = [json.loads(x) for x in (out / "population.jsonl").read_text().splitlines()]
        sub = [p for p in pop if p["in_j2_subsample"]]
        rep = [p for p in pop if p["in_repeat_set"]]
        self.assertEqual(len(sub), 10)
        self.assertEqual(len(rep), 4)
        self.assertTrue(all(p["split"] == "main" for p in sub + rep))

    def test_collection_shape_and_no_labels_leak(self):
        out = self.run_prepare()
        coll = out / "collection"
        records = [json.loads(x) for x in (coll / "records.jsonl").read_text().splitlines()]
        for rec in records:
            path = coll / rec["response_file"]
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), rec["response_sha256"])
            self.assertTrue(rec["capture_complete"])
        main = [json.loads(x) for x in (out / "main_manifest.jsonl").read_text().splitlines()]
        src_lines = (coll / "source_records.jsonl").read_text().splitlines()
        for row in main:
            record = json.loads(src_lines[row["source_line"] - 1])
            self.assertEqual(record["id"], row["candidate_id"])
            self.assertEqual(hashlib.sha256(record["objective_text"].encode()).hexdigest(), row["objective_sha256"])
            self.assertFalse(row["context_present"])
        for name in ("records.jsonl", "source_records.jsonl", "manifest.jsonl"):
            text = (coll / name).read_text()
            self.assertNotIn("label", text)
            self.assertNotIn("median", text)
        ref = [json.loads(x) for x in (out / "private" / "external_reference.jsonl").read_text().splitlines()]
        self.assertEqual(len(ref), len(records))
        self.assertTrue(all(len(r["labels_0_1"]) == 5 for r in ref))

    def test_categories_join(self):
        out = self.run_prepare()
        objs = [json.loads(x) for x in (out / "objectives.jsonl").read_text().splitlines()]
        cats = {o["objective_id"]: o["category"] for o in objs}
        self.assertIn("Cat A", cats.values())
        self.assertIn("unknown_source_category", cats.values())

    def test_refuses_existing_out_dir(self):
        out = self.run_prepare()
        rc = tool.main(["prepare", "--source-csv", str(self.src), "--expected-sha256", self.digest,
                        "--out-dir", str(out)])
        self.assertEqual(rc, 2)

    def test_materialise_requirements(self):
        out = self.run_prepare()
        objs = [json.loads(x) for x in (out / "objectives.jsonl").read_text().splitlines()]
        req_dir = self.tmp / "objreq"
        req_dir.mkdir()
        for o in objs[:-1]:   # leave one objective without a list
            (req_dir / f"{o['objective_id']}.requirements.json").write_text(
                json.dumps([{"id": "R1", "description": "delivers what is asked"}]), encoding="utf-8")
        rc = tool.main(["materialise-requirements", "--prep-dir", str(out), "--objective-requirements", str(req_dir),
                        "--out-dir", str(self.tmp / "req")])
        self.assertEqual(rc, 1)
        record = json.loads((self.tmp / "req" / "materialise_record.json").read_text())
        self.assertEqual(list(record["rows_without_list_by_objective"]), [objs[-1]["objective_id"]])
        pop = [json.loads(x) for x in (out / "population.jsonl").read_text().splitlines()]
        n_with = sum(1 for p in pop if p["objective_id"] != objs[-1]["objective_id"])
        self.assertEqual(record["written"], n_with)


if __name__ == "__main__":
    unittest.main()
