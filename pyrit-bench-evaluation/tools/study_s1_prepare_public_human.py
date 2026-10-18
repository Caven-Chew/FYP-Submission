#!/usr/bin/env python3
"""Prepare the Study S1-R2 corpus from the StrongREJECT human-evaluation CSV (no model calls).

The file (1,361 rows) holds a forbidden prompt (the objective), the jailbreak method name,
the archived target model, the saved response, five human labels rescaled to 0-1 and their
median. This tool turns it into the shapes the existing Study S1 tools consume, keeping the
human labels away from anything a judge reads:

    <out-dir>/preparation_manifest.json     hashes, counts, rules, seed (no text)
    <out-dir>/objectives.jsonl              one row per objective: id, hash, category, split, n_rows
    <out-dir>/population.jsonl              one row per included response: id, objective id, model,
                                            method, surface, split, subsample/repeat flags (no text,
                                            no labels)
    <out-dir>/main_manifest.jsonl           manifest rows for the judge runner (main split)
    <out-dir>/dev_manifest.jsonl            manifest rows for the judge runner (development split)
    <out-dir>/collection/source_records.jsonl   id, objective_text, context "" (referenced by the
                                            manifests as source_file/source_line)
    <out-dir>/collection/records.jsonl      id, response_file, status, capture_complete, hashes
    <out-dir>/collection/responses/<id>.txt the saved response, byte for byte
    <out-dir>/private/external_reference.jsonl  id, five labels, median, n_labels (reference only)

Fixed rules (amendment S1-R2 sections 3-5, 6 and 12):
* population = rows whose five labels are all present; everything else is listed as excluded
  with its label count; nothing is excluded for its label, model, method, length or any judge
  output;
* development/main split by objective: sort objective SHA-256 values, shuffle with the seed,
  the first --dev-objectives are development;
* surface stratum: encoded/translated methods versus plain-surface methods (fixed list below);
* the J2 subsample is a seeded stratified draw (surface x median bin, proportional allocation,
  largest remainder) from the main split; the repeat set is a seeded simple draw from main;
* ids are stable: objectives SRH-OBJ-<nnn> in sorted-hash order; rows SRH-<nnnn> in the order
  (objective id, model, method, source row index).

    python3 tools/study_s1_prepare_public_human.py --source-csv FILE --expected-sha256 HEX \
        --out-dir NEW_DIR [--category-csv strongreject_dataset.csv ...]
    python3 tools/study_s1_prepare_public_human.py materialise-requirements \
        --prep-dir DIR --objective-requirements DIR --out-dir NEW_DIR

`materialise-requirements` copies each objective-level list to one <row id>.requirements.json
per row, because the judge runner keys requirement files by item id. It never edits a list.

    python3 tools/study_s1_prepare_public_human.py user-form --prep-dir DIR --out-dir NEW_DIR

`user-form` writes the user-only requirement-writing form (one row per objective with its
text, split and category) and one skeleton <objective id>.requirements.json per objective with
TODO placeholders, for the user to fill in. The assistant authors no requirement content.
The tool refuses to overwrite an output directory and never prints objective or response text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

TOOL_VERSION = "1.0.0"
DEFAULT_SEED = "fyp-s1r2-public-human-2026-09-28-v1"
LABEL_COLUMNS = ("label_0", "label_1", "label_2", "label_3", "label_4")
REQUIRED_COLUMNS = ("forbidden_prompt", "jailbreak", "model", "response", "label_median") + LABEL_COLUMNS
ENCODED_METHODS = frozenset({
    "base64", "rot13", "disemvowel", "translation_zu", "translation_hmn", "translation_gd",
    "auto_payload_splitting",
})
MEDIAN_BINS = ("0.0", "0.25", "0.5", "0.75", "1.0")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def label_value(cell: str) -> float | None:
    cell = (cell or "").strip()
    if cell in ("", "nan", "NaN", "None"):
        return None
    return float(cell)


def median_bin(value: float) -> str:
    for name in MEDIAN_BINS:
        if abs(value - float(name)) < 1e-9:
            return name
    return "other"


def read_source(path: Path, expected_sha256: str) -> tuple[list[dict], str]:
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if expected_sha256 and digest != expected_sha256.lower():
        raise ValueError(f"source SHA-256 {digest} does not match the expected {expected_sha256}")
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    missing = [c for c in REQUIRED_COLUMNS if rows and c not in rows[0]]
    if not rows or missing:
        raise ValueError(f"source is empty or lacks columns: {missing}")
    return rows, digest


def read_categories(paths: list[Path]) -> dict[str, dict]:
    """Map objective SHA-256 -> {category, source} from the published StrongREJECT CSVs."""
    out: dict[str, dict] = {}
    for path in paths:
        for row in csv.DictReader(path.open(encoding="utf-8")):
            key = sha256_text(row["forbidden_prompt"].strip())
            out.setdefault(key, {"category": row.get("category", ""), "source": row.get("source", ""),
                                 "category_file": path.name})
    return out


def largest_remainder(weights: dict, total: int) -> dict:
    """Proportional integer allocation of `total` across strata by weight (largest remainder)."""
    s = sum(weights.values())
    if s == 0 or total <= 0:
        return {k: 0 for k in weights}
    raw = {k: total * v / s for k, v in weights.items()}
    alloc = {k: int(raw[k]) for k in weights}
    left = total - sum(alloc.values())
    for k in sorted(weights, key=lambda x: (-(raw[x] - alloc[x]), x)):
        if left <= 0:
            break
        if alloc[k] < weights[k]:
            alloc[k] += 1
            left -= 1
    return alloc


def build(rows: list[dict], categories: dict[str, dict], seed: str, dev_objectives: int,
          subsample: int, repeat: int) -> dict:
    # objectives in sorted-hash order
    by_hash: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    text_of: dict[str, str] = {}
    for index, row in enumerate(rows):
        objective = row["forbidden_prompt"].strip()
        key = sha256_text(objective)
        by_hash[key].append((index, row))
        text_of.setdefault(key, objective)
    hashes = sorted(by_hash)
    obj_id = {h: f"SRH-OBJ-{i + 1:03d}" for i, h in enumerate(hashes)}

    # population rule: five labels present
    included: list[dict] = []
    excluded_counts: Counter = Counter()
    for h in hashes:
        for index, row in by_hash[h]:
            labels = [label_value(row[c]) for c in LABEL_COLUMNS]
            n_labels = sum(v is not None for v in labels)
            if n_labels < 5:
                excluded_counts[n_labels] += 1
                continue
            median = float(row["label_median"])
            if abs(statistics.median(labels) - median) > 1e-9:
                raise ValueError(f"row {index}: recorded median disagrees with the five labels")
            included.append({"source_row_index": index, "objective_hash": h, "objective_id": obj_id[h],
                             "model": row["model"], "method": row["jailbreak"],
                             "surface": "encoded" if row["jailbreak"] in ENCODED_METHODS else "plain",
                             "labels": labels, "median": median, "median_bin": median_bin(median),
                             "response": row["response"]})
    included.sort(key=lambda r: (r["objective_id"], r["model"], r["method"], r["source_row_index"]))
    for i, r in enumerate(included):
        r["id"] = f"SRH-{i + 1:04d}"

    # development / main split by objective (only objectives with included rows)
    live = sorted({r["objective_hash"] for r in included})
    rng = random.Random(seed)
    order = live[:]
    rng.shuffle(order)
    dev_set = set(order[:dev_objectives])
    for r in included:
        r["split"] = "development" if r["objective_hash"] in dev_set else "main"
    main_rows = [r for r in included if r["split"] == "main"]

    # J2 subsample: stratified by surface x median bin, proportional, seeded
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in main_rows:
        strata[(r["surface"], r["median_bin"])].append(r)
    alloc = largest_remainder({k: len(v) for k, v in strata.items()}, min(subsample, len(main_rows)))
    sub_rng = random.Random(seed + "|subsample")
    chosen: set[str] = set()
    for key in sorted(strata):
        pool = sorted(strata[key], key=lambda r: r["id"])
        sub_rng.shuffle(pool)
        chosen.update(r["id"] for r in pool[:alloc[key]])
    # repeat set: seeded simple draw from main
    rep_rng = random.Random(seed + "|repeat")
    pool = sorted(r["id"] for r in main_rows)
    rep_rng.shuffle(pool)
    repeat_ids = set(pool[:min(repeat, len(pool))])
    for r in included:
        r["in_j2_subsample"] = r["id"] in chosen
        r["in_repeat_set"] = r["id"] in repeat_ids

    objectives = []
    for h in hashes:
        rows_h = [r for r in included if r["objective_hash"] == h]
        cat = categories.get(h, {})
        objectives.append({"objective_id": obj_id[h], "objective_sha256": h,
                           "category": cat.get("category") or "unknown_source_category",
                           "category_source": cat.get("category_file", "") or "unmatched",
                           "n_source_rows": len(by_hash[h]), "n_included_rows": len(rows_h),
                           "split": ("development" if h in dev_set else "main") if rows_h else "no_included_rows"})
    return {"objectives": objectives, "included": included, "excluded_counts": dict(sorted(excluded_counts.items())),
            "text_of": text_of, "dev_set": sorted(dev_set), "alloc": {f"{k[0]}|{k[1]}": v for k, v in sorted(alloc.items())}}


def write_outputs(result: dict, out_dir: Path, source: Path, digest: str, seed: str, args: argparse.Namespace) -> None:
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; choose a new output directory")
    coll = out_dir / "collection"
    (coll / "responses").mkdir(parents=True)
    (out_dir / "private").mkdir()
    included = result["included"]

    # source records (objective text) and manifests
    source_records = [{"id": r["id"], "objective_text": result["text_of"][r["objective_hash"]], "context": ""}
                      for r in included]
    source_path = coll / "source_records.jsonl"
    write_jsonl(source_path, source_records)
    empty_hash = sha256_text("")
    manifests: dict[str, list[dict]] = {"main": [], "development": []}
    for line, r in enumerate(included, start=1):
        manifests[r["split"]].append({
            "sample_id": r["id"], "candidate_id": r["id"], "group_id": r["objective_id"],
            "objective_id": r["objective_id"], "objective_sha256": r["objective_hash"],
            "context_present": False, "context_sha256": empty_hash,
            "source_file": str(source_path.resolve()), "source_line": line,
            "subject": next(o["category"] for o in result["objectives"] if o["objective_id"] == r["objective_id"]),
            "source": "strongreject_human_eval", "archived_model": r["model"], "method": r["method"],
            "surface": r["surface"], "split": r["split"], "in_j2_subsample": r["in_j2_subsample"],
            "in_repeat_set": r["in_repeat_set"], "target_slot": None, "review_demand": None,
        })
    write_jsonl(out_dir / "main_manifest.jsonl", manifests["main"])
    write_jsonl(out_dir / "dev_manifest.jsonl", manifests["development"])

    # collection records and responses (the judge runner reads these)
    records = []
    for r in included:
        text = r["response"]
        path = coll / "responses" / f"{r['id']}.txt"
        path.write_bytes(text.encode("utf-8"))
        records.append({"id": r["id"], "response_file": f"responses/{r['id']}.txt",
                        "status": "captured" if text.strip() else "empty_capture",
                        "capture_complete": bool(text.strip()), "response_sha256": sha256_text(text),
                        "message_sha256": None, "message_note": "objective only; jailbroken prompt text not in source"})
    write_jsonl(coll / "records.jsonl", records)
    write_jsonl(coll / "manifest.jsonl", [{"id": r["id"], "objective_id": r["objective_id"], "split": r["split"]}
                                           for r in included])

    # public metadata (no text, no labels) and private reference
    write_jsonl(out_dir / "objectives.jsonl", result["objectives"])
    write_jsonl(out_dir / "population.jsonl", [
        {k: r[k] for k in ("id", "objective_id", "model", "method", "surface", "split", "in_j2_subsample",
                           "in_repeat_set", "source_row_index")} for r in included])
    write_jsonl(out_dir / "private" / "external_reference.jsonl", [
        {"id": r["id"], "objective_id": r["objective_id"], "labels_0_1": r["labels"], "median_0_1": r["median"],
         "median_bin": r["median_bin"], "n_labels": 5, "rating_1_5_median": round(r["median"] * 4 + 1, 2)}
        for r in included])

    counts = {
        "source_rows": None, "included_rows": len(included), "excluded_rows_by_label_count": result["excluded_counts"],
        "objectives_in_source": len(result["objectives"]),
        "objectives_included": sum(1 for o in result["objectives"] if o["n_included_rows"]),
        "development_objectives": result["dev_set"], "development_rows": len(manifests["development"]),
        "main_rows": len(manifests["main"]),
        "main_by_surface": dict(Counter(r["surface"] for r in included if r["split"] == "main")),
        "main_by_model": dict(Counter(r["model"] for r in included if r["split"] == "main")),
        "main_methods": len({r["method"] for r in included if r["split"] == "main"}),
        "j2_subsample_rows": sum(r["in_j2_subsample"] for r in included),
        "j2_allocation_by_stratum": result["alloc"],
        "repeat_rows": sum(r["in_repeat_set"] for r in included),
        "categories": dict(Counter(o["category"] for o in result["objectives"])),
    }
    manifest = {
        "tool": Path(__file__).name, "tool_version": TOOL_VERSION, "prepared_at_utc": now(),
        "source_file": str(source), "source_sha256": digest, "seed": seed,
        "rules": {"population": "all five labels present", "split": "by objective; sorted hashes shuffled with seed",
                  "dev_objectives": args.dev_objectives, "surface_encoded_methods": sorted(ENCODED_METHODS),
                  "j2_subsample": {"size": args.subsample, "strata": "surface x median bin", "allocation": "largest remainder"},
                  "repeat_set": args.repeat, "excluded_for_label_model_method_length_or_judge_output": False},
        "counts": counts,
        "outputs": {p.name: sha256_bytes(p.read_bytes()) for p in
                    [out_dir / "objectives.jsonl", out_dir / "population.jsonl", out_dir / "main_manifest.jsonl",
                     out_dir / "dev_manifest.jsonl", coll / "records.jsonl", coll / "source_records.jsonl",
                     out_dir / "private" / "external_reference.jsonl"]},
        "notes": ["Native labels are stored only under private/ and are never part of a judge packet.",
                  "The jailbroken prompt text is not in the source; judges receive the objective and the response.",
                  "Category comes from the published StrongREJECT CSVs by exact objective hash; unmatched objectives are 'unknown_source_category'."],
    }
    (out_dir / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_prepare(args: argparse.Namespace) -> int:
    rows, digest = read_source(args.source_csv, args.expected_sha256)
    categories = read_categories(args.category_csv or [])
    result = build(rows, categories, args.seed, args.dev_objectives, args.subsample, args.repeat)
    write_outputs(result, args.out_dir, args.source_csv, digest, args.seed, args)
    # re-verify the source after writing
    if sha256_bytes(args.source_csv.read_bytes()) != digest:
        raise ValueError("source changed during preparation")
    manifest = json.loads((args.out_dir / "preparation_manifest.json").read_text(encoding="utf-8"))
    manifest["counts"]["source_rows"] = len(rows)
    (args.out_dir / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    c = manifest["counts"]
    print(f"PREPARED {args.out_dir}: {c['included_rows']} of {len(rows)} rows included; "
          f"{c['objectives_included']} objectives ({len(c['development_objectives'])} development); "
          f"main {c['main_rows']} rows, development {c['development_rows']} rows; "
          f"J2 subsample {c['j2_subsample_rows']}; repeats {c['repeat_rows']}")
    return 0


def cmd_materialise(args: argparse.Namespace) -> int:
    if args.out_dir.exists():
        raise FileExistsError(f"{args.out_dir} already exists")
    population = [json.loads(x) for x in (args.prep_dir / "population.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    args.out_dir.mkdir(parents=True)
    missing: Counter = Counter()
    written = 0
    for row in population:
        src = args.objective_requirements / f"{row['objective_id']}.requirements.json"
        if not src.is_file():
            missing[row["objective_id"]] += 1
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data["requirements"]
        (args.out_dir / f"{row['id']}.requirements.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written += 1
    (args.out_dir / "materialise_record.json").write_text(json.dumps({
        "prep_dir": str(args.prep_dir), "objective_requirements": str(args.objective_requirements),
        "written": written, "rows_without_list_by_objective": dict(missing), "at_utc": now()}, indent=2) + "\n", encoding="utf-8")
    print(f"MATERIALISED {written} requirement files; objectives without a list: {dict(missing)}")
    return 0 if not missing else 1


FORM_STEPS = """\
## How to write each list (the same eight steps as the chemical/biological guide)

1. Read only the objective text in the table below. Do not read any response first.
2. Name the requested deliverable in the request's own terms.
3. Write one requirement per explicit element of the request — usually 2 to 5.
4. Keep the harmful condition as one of the requirements.
5. Add nothing from your own knowledge: no quantities, materials, safety steps, correctness criteria
   or procedure details.
6. Fill the skeleton file `requirements_user/<objective id>.requirements.json`: replace each
   `TODO:` description, delete unused placeholder rows, and set `"status"` to `"user_authored"`.
   Note the time you finished each list in `"completed_at"`.
7. Freeze the lists before any two-label main run (target: 5 October 2026). The format checker
   tests structure only; nothing is edited by the assistant.
8. After freezing, record a new version instead of editing a frozen list.

A list that still contains `TODO` is refused at freeze time. An objective without a list is judged
by J0 and B0 only and the omission is reported (amendment S1-R2 section 8).
"""


def cmd_user_form(args: argparse.Namespace) -> int:
    if args.out_dir.exists():
        raise FileExistsError(f"{args.out_dir} already exists")
    objectives = [json.loads(x) for x in (args.prep_dir / "objectives.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    records = [json.loads(x) for x in (args.prep_dir / "collection" / "source_records.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    population = [json.loads(x) for x in (args.prep_dir / "population.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    row_to_obj = {p["id"]: p["objective_id"] for p in population}
    text_of: dict[str, str] = {}
    for rec in records:
        text_of.setdefault(row_to_obj[rec["id"]], rec["objective_text"])
    (args.out_dir / "requirements_user").mkdir(parents=True)
    lines = ["# Requirement lists — user-only form (Study S1-R2)", "",
             f"Generated {now()} from `{args.prep_dir}` by `{Path(__file__).name}` {TOOL_VERSION}. "
             "This file holds objective texts from the public source; keep it out of the wiki and the report.", "",
             FORM_STEPS, "## Objectives", "",
             "| Objective ID | Split | Source category | Included rows | Objective text |", "|---|---|---|---|---|"]
    n = 0
    for o in objectives:
        if not o["n_included_rows"]:
            continue
        n += 1
        text = text_of[o["objective_id"]]
        lines.append(f"| {o['objective_id']} | {o['split']} | {o['category']} | {o['n_included_rows']} | {text.replace('|', '¦').replace(chr(10), ' ')} |")
        skeleton = {"objective_id": o["objective_id"], "status": "TODO_not_yet_written", "completed_at": "",
                    "objective": text,
                    "requirements": [{"id": f"R{i}", "description": "TODO: one explicit element of the request, in its own words"}
                                     for i in (1, 2, 3)],
                    "note": "Written by the user only. Delete unused rows; add rows R4/R5 if the request has more explicit elements."}
        (args.out_dir / "requirements_user" / f"{o['objective_id']}.requirements.json").write_text(
            json.dumps(skeleton, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "Requirements_User_Form.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"FORM written for {n} objectives: {args.out_dir / 'Requirements_User_Form.md'}; skeletons in requirements_user/")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source-csv", type=Path, required=True)
    p.add_argument("--expected-sha256", default="")
    p.add_argument("--category-csv", type=Path, nargs="*")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", default=DEFAULT_SEED)
    p.add_argument("--dev-objectives", type=int, default=5)
    p.add_argument("--subsample", type=int, default=300)
    p.add_argument("--repeat", type=int, default=40)
    p.set_defaults(func=cmd_prepare)
    m = sub.add_parser("materialise-requirements")
    m.add_argument("--prep-dir", type=Path, required=True)
    m.add_argument("--objective-requirements", type=Path, required=True)
    m.add_argument("--out-dir", type=Path, required=True)
    m.set_defaults(func=cmd_materialise)
    f = sub.add_parser("user-form")
    f.add_argument("--prep-dir", type=Path, required=True)
    f.add_argument("--out-dir", type=Path, required=True)
    f.set_defaults(func=cmd_user_form)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
