#!/usr/bin/env python3
"""Repeat stability for Study S1-R2 (amendment section 12): primary run versus its repeat; no model calls.

For each configuration, compares the normalised primary predictions (repeat index 0) with the
normalised repeat predictions (repeat index 1) on the repeated rows, using the analysis tool's own
mapping. Reports: rows compared; both-valid rows; exact agreement of the prediction (severity value,
or objective label + fulfilment outcome); agreement of the primary-cut decision; and validity
changes (valid in one call, not in the other). Writes a new JSON file; refuses to overwrite.

    python3 tools/study_s1_repeat_stability.py --pair J0=primary.jsonl,repeat.jsonl \
        --pair J1=primary.jsonl,repeat.jsonl --kinds J0=severity J1=two_label --out NEW.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_s1_external_reference import load_jsonl, prediction_state  # noqa: E402


def exact(row: dict | None, kind: str):
    if row is None or row.get("valid", True) is not True:
        return None
    return row.get("severity") if kind == "severity" else (row.get("objective_pred"), row.get("fulfilment_pred"))


def stability(primary: list[dict], repeat: list[dict], kind: str) -> dict:
    first = {r["id"]: r for r in primary}
    second = {r["id"]: r for r in repeat}
    ids = sorted(second)
    missing = [i for i in ids if i not in first]
    if missing:
        raise ValueError(f"{len(missing)} repeated rows have no primary prediction")
    both_valid = exact_same = cut_same = changed_validity = 0
    for i in ids:
        a, b = prediction_state(first[i], kind), prediction_state(second[i], kind)
        if (a["status"] == "valid") != (b["status"] == "valid"):
            changed_validity += 1
        if a["status"] == "valid" and b["status"] == "valid":
            both_valid += 1
            exact_same += exact(first[i], kind) == exact(second[i], kind)
            cut_same += a["primary"] == b["primary"]
    rate = lambda k: round(k / both_valid, 4) if both_valid else None
    return {"rows": len(ids), "both_valid": both_valid, "validity_changed": changed_validity,
            "exact_agreement": exact_same, "exact_agreement_rate": rate(exact_same),
            "primary_cut_agreement": cut_same, "primary_cut_agreement_rate": rate(cut_same)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", action="append", required=True, help="CONFIG=primary.jsonl,repeat.jsonl")
    parser.add_argument("--kinds", nargs="+", required=True, help="CONFIG=severity|two_label")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise FileExistsError(f"refusing to overwrite {args.out}")
        kinds = dict(item.split("=", 1) for item in args.kinds)
        result = {"tool": "study_s1_repeat_stability", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "configurations": {}}
        for item in args.pair:
            cid, paths = item.split("=", 1)
            primary, repeat = (Path(p) for p in paths.split(","))
            if kinds.get(cid) not in ("severity", "two_label"):
                raise ValueError(f"{cid}: give --kinds {cid}=severity|two_label")
            result["configurations"][cid] = {"primary": str(primary), "repeat": str(repeat), "kind": kinds[cid],
                                             **stability(load_jsonl(primary), load_jsonl(repeat), kinds[cid])}
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, indent=2) + "\n")
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for cid, c in result["configurations"].items():
        print(f"{cid}: {c['rows']} repeated rows; both valid {c['both_valid']}; exact {c['exact_agreement_rate']}; "
              f"primary cut {c['primary_cut_agreement_rate']}; validity changed {c['validity_changed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
