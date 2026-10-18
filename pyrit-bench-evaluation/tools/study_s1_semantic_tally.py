#!/usr/bin/env python3
"""Check and tally a completed Study S1-R2 semantic-review packet (amendment S1-R2 section 12); no model calls.

Reads the reviewer's filled J1_Semantic_Review.md and checks it against the prespecified sample:
the same rows in the same order, one finding per row (supported / issue found / unresolved),
evidence line IDs that exist in the row's numbered response (or "none") and a note for every
"issue found" or "unresolved". It then adds each row's primary-cut direction (J1 positive or
negative versus the five-labeller reference) and counts findings by sample group and direction.
The reviewer's findings and notes are copied verbatim; nothing changes predictions or the
reference. Writes a new JSON file; refuses to overwrite.

    python3 tools/study_s1_semantic_tally.py --packet DIR/J1_Semantic_Review.md --sample DIR/sample.json \
        --prep-dir PREP --predictions J1.jsonl --out NEW.json
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_s1_external_reference import CUTS, index_by_id, load_jsonl, prediction_state  # noqa: E402

FINDINGS = ("supported", "issue found", "unresolved")
HEADING = re.compile(r"^### (\d+)\. (\S+) \((disagreement|agreement)\)\s*$")
LINE_ID = re.compile(r"^L\d{3,}$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_packet(text: str) -> tuple[dict, list[dict]]:
    reviewer = re.search(r"^Reviewer: (.*?) · Review completed at \(date and time\): (.*)$", text, re.M)
    if not reviewer or "____" in reviewer.group(0):
        raise ValueError("reviewer line is not filled in")
    rows, current = [], None
    for number, line in enumerate(text.splitlines(), 1):
        match = HEADING.match(line)
        if match:
            if current:
                raise ValueError(f"row {current['n']} has no finding line")
            current = {"n": int(match.group(1)), "id": match.group(2), "group": match.group(3)}
        elif current and line.startswith("|") and not line.startswith(("| Finding", "|---")):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 3:
                raise ValueError(f"line {number}: expected 3 table cells, found {len(cells)}")
            current.update(finding=cells[0], evidence=cells[1], note=cells[2], packet_line=number)
            rows.append(current)
            current = None
    if current:
        raise ValueError(f"row {current['n']} has no finding line")
    return {"name": reviewer.group(1).strip(), "completed_at_as_written": reviewer.group(2).strip()}, rows


def check_row(row: dict, responses_dir: Path) -> list[str]:
    problems = []
    if row["finding"] not in FINDINGS:
        problems.append(f"finding {row['finding']!r} is not one of {FINDINGS}")
    if row["evidence"].lower() == "none":
        row["evidence_lines"] = []
    else:
        ids = [part.strip() for part in row["evidence"].split(",")]
        known = {line.split(" | ", 1)[0] for line in
                 (responses_dir / f"{row['id']}.txt").read_text(encoding="utf-8").splitlines()}
        bad = [i for i in ids if not LINE_ID.match(i) or i not in known]
        if bad or not ids:
            problems.append(f"evidence {row['evidence']!r} names lines not in the response: {bad}")
        row["evidence_lines"] = ids
    if row["finding"] in ("issue found", "unresolved") and not row["note"]:
        problems.append("a note is required for 'issue found' or 'unresolved'")
    return problems


def tally(packet: Path, sample_file: Path, prep_dir: Path, predictions: Path) -> dict:
    sample = json.loads(sample_file.read_text(encoding="utf-8"))
    if sample.get("predictions_sha256") != sha256(predictions):
        raise ValueError("predictions differ from the ones the sample was drawn from")
    expected = [(i, "disagreement") for i in sample["sample"]["disagreement"]["ids"]] + \
               [(i, "agreement") for i in sample["sample"]["agreement"]["ids"]]
    reviewer, rows = parse_packet(packet.read_text(encoding="utf-8"))
    if [(r["id"], r["group"]) for r in rows] != expected or [r["n"] for r in rows] != list(range(1, len(expected) + 1)):
        raise ValueError("packet rows do not match the prespecified sample and order")
    problems = [f"row {r['n']} {r['id']}: {p}" for r in rows for p in check_row(r, packet.parent / "responses_numbered")]
    if problems:
        raise ValueError("; ".join(problems))
    reference = index_by_id(load_jsonl(prep_dir / "private" / "external_reference.jsonl"), "reference")
    preds = index_by_id(load_jsonl(predictions), "predictions")
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        state = prediction_state(preds.get(row["id"]), "two_label")
        human = float(reference[row["id"]]["median_0_1"]) >= CUTS["primary"]
        if state["status"] != "valid" or (state["primary"] != human) != (row["group"] == "disagreement"):
            raise ValueError(f"row {row['id']}: prediction no longer matches its sample group")
        row["direction"] = {(True, False): "J1 positive, reference negative",
                            (False, True): "J1 negative, reference positive",
                            (True, True): "both positive", (False, False): "both negative"}[(state["primary"], human)]
        for key in (row["group"], row["direction"], "all"):
            counts.setdefault(key, dict.fromkeys(FINDINGS, 0))[row["finding"]] += 1
    return {"tool": "study_s1_semantic_tally", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "packet": str(packet), "packet_sha256": sha256(packet), "sample": str(sample_file),
            "sample_sha256": sha256(sample_file), "predictions": str(predictions),
            "predictions_sha256": sha256(predictions), "reviewer": reviewer, "rows_reviewed": len(rows),
            "counts": counts, "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--prep-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise FileExistsError(f"refusing to overwrite {args.out}")
        result = tally(args.packet, args.sample, args.prep_dir, args.predictions)
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for key, value in result["counts"].items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
