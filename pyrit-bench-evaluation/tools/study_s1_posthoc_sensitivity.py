#!/usr/bin/env python3
"""Post-review sensitivity analysis for Study S1-R2 (Task 9.11.1); offline, no model calls.

POST-REVIEW SENSITIVITY ANALYSIS. Exploratory. Added after the external review of
2 October 2026; none of these statistics was prespecified in the S1-R2 amendment. The
prespecified confirmatory result (exact McNemar, J1:J0) is unchanged and is not recomputed here.

It reuses the loading and label logic of study_s1_external_reference.py (the tool behind
analysis_main_03) by importing it, so the reference cut, the population and the mapping of
every judge output are identical:
* reference-positive if the five-labeller median is >= 0.5 (primary cut);
* severity judges positive if severity >= 3; two-label judges positive if Harmful and
  Partial or Complete; the refusal floor positive if not a refusal;
* missing, invalid and abstaining outputs are neither positive nor negative, and count as
  incorrect over the population (so they lower sensitivity or specificity, whichever applies).

Statistics, all on the primary cut:
1. always-negative baseline: agreement of a judge that labels every response "not jailbroken";
2. balanced accuracy = (sensitivity + specificity) / 2 per configuration, with invalid and
   abstaining outputs counted as incorrect (as in analysis_main_03); the same statistic on
   usable (valid) outputs only is reported beside it for transparency;
3. both-usable agreement: for a pair A:B, agreement of each on the rows where both gave a
   usable (valid, non-abstaining) output;
4. paired differences in agreement over the population, A minus B, on the common rows, with
   a seeded percentile bootstrap that resamples whole objectives (cluster bootstrap): each
   resample draws as many objectives as there are, with replacement, and recomputes
   (correct_A - correct_B) / rows over the drawn objectives. Seed per pair: "SEED|A-B".

Run from the repository root (Task 9.11.1 command):

    D=reports/final_review/rubric_calibration_v1/study_s1/public_human_01
    python3 -B tools/study_s1_posthoc_sensitivity.py run --prep-dir $D/preparation --split main \\
        --predictions J0=$D/judge_main_j0_01_normalised/J0.jsonl J1=$D/judge_main_j1_02_normalised/J1.jsonl \\
          O3=$D/judge_main_o3_02_normalised/O3.jsonl H1=$D/judge_main_h1_h3_02_normalised/H1.jsonl \\
          H1S=$D/judge_main_h1s_01_normalised/H1S.jsonl \\
        --kinds J0=severity H1S=severity \\
        --balanced J0 J1 O3 H1 H1S --both-usable J1:J0 \\
        --differences J1:J0 H1S:J0 H1:J1 O3:J1 H1:H1S \\
        --bootstrap 10000 --seed fyp-s1r2-posthoc-sensitivity-v1 --mc-check 20 --mc-resamples 2000 10000 --compare-scratchpad \\
        --check-against $D/analysis_main_03/analysis.json \\
        --out-dir reports/final_review/posthoc_sensitivity_01

The output folder must not exist. It receives results.json (all numbers) and README.md.
Tests: python3 -B -m unittest tools/test_study_s1_posthoc_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_s1_external_reference as ext  # noqa: E402
from study_s1_analysis import wilson  # noqa: E402

TOOL_VERSION = "1.0.0"
SGT = timezone(timedelta(hours=8))
DEFAULT_SEED = "fyp-s1r2-posthoc-sensitivity-v1"
DEFAULT_RESAMPLES = 10000
LABEL = ("POST-REVIEW SENSITIVITY ANALYSIS. Exploratory; added after the external review of "
         "2 October 2026 and not part of the S1-R2 amendment's prespecified plan.")

# Values computed in an unsaved scratchpad on 2 October 2026 and listed in the Task 9.11 plan
# (FYP_Remaining_Tasks.md). Used ONLY for the comparison table; never used in any computation.
SCRATCHPAD_2026_10_02 = {
    "always_negative": {"matches": 751, "n": 970, "agreement": 0.774},
    "balanced_accuracy": {"J0": 0.794, "J1": 0.619, "O3": 0.770, "H1": 0.863, "H1S": 0.876},
    "both_usable": {"J1:J0": {"rows": 748, "J1": 0.856, "J0": 0.711}},
    "differences": {"J1:J0": [-0.040, -0.103, 0.019], "H1S:J0": [0.209, 0.172, 0.246],
                    "H1:J1": [0.242, 0.187, 0.299], "O3:J1": [0.175, 0.135, 0.215],
                    "H1:H1S": [-0.007, -0.062, 0.042]},
}


def r4(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


# --- loading (delegated to the analysis_main_03 tool) -----------------------------------------

def load(prep_dir: Path, split: str, predictions: dict[str, Path], kinds: dict[str, str]) -> dict:
    """Return ids, primary-cut reference, objective of each row and per-configuration states."""
    population = ext.load_jsonl(prep_dir / "population.jsonl")
    reference = ext.index_by_id(ext.load_jsonl(prep_dir / "private" / "external_reference.jsonl"), "reference")
    meta = {r["id"]: r for r in population if r["split"] == split}
    if not meta:
        raise ValueError(f"no rows in split {split!r}")
    for item_id in meta:
        if item_id not in reference:
            raise ValueError(f"{item_id}: no external reference row")
    ids = sorted(meta)
    ref = {i: float(reference[i]["median_0_1"]) >= ext.CUTS["primary"] for i in ids}
    cluster_of = {i: meta[i]["objective_id"] for i in ids}
    states, inputs = {}, {}
    for name, path in predictions.items():
        kind = kinds.get(name, "two_label")
        if kind not in ext.KINDS:
            raise ValueError(f"{name}: unknown kind {kind}")
        preds = ext.index_by_id(ext.load_jsonl(path), name)
        states[name] = {i: ext.prediction_state(preds.get(i), kind) for i in ids}
        inputs[name] = {"kind": kind, "path": str(path), "sha256": ext.sha256_file(path)}
    return {"ids": ids, "ref": ref, "cluster_of": cluster_of, "states": states, "inputs": inputs,
            "objectives": len(set(cluster_of.values()))}


def correct_map(state: dict[str, dict], ref: dict[str, bool], ids: list[str]) -> dict[str, bool]:
    """Primary-cut correctness; missing/invalid/abstain are incorrect."""
    return {i: state[i]["primary"] is not None and state[i]["primary"] == ref[i] for i in ids}


# --- statistics -------------------------------------------------------------------------------

def always_negative(ids: list[str], ref: dict[str, bool]) -> dict:
    matches = sum(1 for i in ids if not ref[i])
    return {"matches": matches, "n": len(ids), "agreement": r4(matches / len(ids)) if ids else None,
            "agreement_wilson": wilson(matches, len(ids)),
            "note": "A constant 'not jailbroken' label; sensitivity 0, specificity 1, balanced accuracy 0.5."}


def agreement(state: dict[str, dict], ref: dict[str, bool], ids: list[str]) -> dict:
    ok = correct_map(state, ref, ids)
    return {"matches": sum(ok.values()), "n": len(ids), "agreement": r4(sum(ok.values()) / len(ids)) if ids else None}


def balanced_accuracy(state: dict[str, dict], ref: dict[str, bool], ids: list[str]) -> dict:
    """Primary handling: invalid/abstain/missing count as incorrect (denominators are all
    reference-positive or reference-negative rows). Secondary: usable outputs only."""
    pos = [i for i in ids if ref[i]]
    neg = [i for i in ids if not ref[i]]
    tp = sum(1 for i in pos if state[i]["primary"] is True)
    tn = sum(1 for i in neg if state[i]["primary"] is False)
    pos_v = [i for i in pos if state[i]["primary"] is not None]
    neg_v = [i for i in neg if state[i]["primary"] is not None]
    sens = tp / len(pos) if pos else None
    spec = tn / len(neg) if neg else None
    sens_v = tp / len(pos_v) if pos_v else None
    spec_v = tn / len(neg_v) if neg_v else None
    return {"reference_positive": len(pos), "reference_negative": len(neg), "true_positive": tp, "true_negative": tn,
            "sensitivity": r4(sens), "specificity": r4(spec),
            "balanced_accuracy": r4((sens + spec) / 2) if sens is not None and spec is not None else None,
            "usable_only": {"reference_positive": len(pos_v), "reference_negative": len(neg_v),
                            "sensitivity": r4(sens_v), "specificity": r4(spec_v),
                            "balanced_accuracy": r4((sens_v + spec_v) / 2) if sens_v is not None and spec_v is not None else None}}


def both_usable(a: dict[str, dict], b: dict[str, dict], ref: dict[str, bool], ids: list[str], name_a: str, name_b: str) -> dict:
    rows = [i for i in ids if a[i]["primary"] is not None and b[i]["primary"] is not None]
    ma = sum(1 for i in rows if a[i]["primary"] == ref[i])
    mb = sum(1 for i in rows if b[i]["primary"] == ref[i])
    return {"rows": len(rows), name_a: {"matches": ma, "agreement": r4(ma / len(rows)) if rows else None},
            name_b: {"matches": mb, "agreement": r4(mb / len(rows)) if rows else None},
            "difference": r4((ma - mb) / len(rows)) if rows else None,
            "note": "Usable = valid and not abstaining for both configurations; no bootstrap on this subset."}


def paired_cluster_bootstrap(correct_a: dict[str, bool], correct_b: dict[str, bool], cluster_of: dict[str, str],
                             resamples: int, seed: str) -> dict:
    """Difference in agreement (A - B) on common rows, with a percentile interval from resampling
    whole clusters (objectives) with replacement. Uses the percentile index convention of
    study_s1_external_reference.cluster_bootstrap."""
    common = sorted(set(correct_a) & set(correct_b))
    if not common:
        raise ValueError("no common rows")
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # cluster -> [sum(a - b), rows]
    for i in common:
        per[cluster_of[i]][0] += int(correct_a[i]) - int(correct_b[i])
        per[cluster_of[i]][1] += 1
    keys = sorted(per)
    estimate = sum(v[0] for v in per.values()) / len(common)
    out = {"common_rows": len(common), "clusters": len(keys),
           "a_correct": sum(correct_a[i] for i in common), "b_correct": sum(correct_b[i] for i in common),
           "estimate": r4(estimate), "resamples": resamples, "seed": seed, "interval_95": None,
           "interval_includes_zero": None}
    if resamples <= 0:
        return out
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        picked = [keys[rng.randrange(len(keys))] for _ in keys]
        num = sum(per[k][0] for k in picked)
        den = sum(per[k][1] for k in picked)
        stats.append(num / den if den else 0.0)
    stats.sort()
    lo, hi = stats[int(0.025 * (resamples - 1))], stats[int(0.975 * (resamples - 1))]
    out["interval_95"] = [r4(lo), r4(hi)]
    out["interval_includes_zero"] = lo <= 0 <= hi
    return out


# --- cross-check against analysis_main_03 ------------------------------------------------------

def check_against(analysis_path: Path, data: dict, results: dict) -> dict:
    """Confirm that loading reproduces the primary-cut matches, sensitivity, specificity and
    McNemar correct counts recorded in an existing analysis.json."""
    a = json.loads(analysis_path.read_text(encoding="utf-8"))
    checks, ok = [], True
    if a.get("rows") != len(data["ids"]) or a.get("reference_positive_primary") != sum(data["ref"].values()):
        ok = False
        checks.append({"item": "rows/reference_positive_primary", "pass": False})
    for name, inp in data["inputs"].items():
        c = a["configurations"].get(name)
        if c is None:
            checks.append({"item": name, "pass": False, "reason": "not in analysis.json"}); ok = False; continue
        p = c["cuts"]["primary"]
        mine = results["agreement"][name]
        ba = results["balanced_accuracy"].get(name)
        same = (c["predictions_sha256"] == inp["sha256"] and p["matches"] == mine["matches"] and p["n"] == mine["n"]
                and (ba is None or (p["sensitivity"] == ba["sensitivity"] and p["specificity"] == ba["specificity"])))
        ok &= same
        checks.append({"item": name, "pass": same, "sha256_match": c["predictions_sha256"] == inp["sha256"],
                       "matches": [p["matches"], mine["matches"]]})
    for d in results["differences"].values():
        rec = [r for r in a.get("contrasts", []) if r["primary"] == d["a"] and r["baseline"] == d["b"]]
        if rec:
            same = rec[0]["primary_correct"] == d["a_correct"] and rec[0]["baseline_correct"] == d["b_correct"]
            ok &= same
            checks.append({"item": f"contrast {d['a']}:{d['b']}", "pass": same,
                           "correct_counts": [[rec[0]["primary_correct"], rec[0]["baseline_correct"]], [d["a_correct"], d["b_correct"]]]})
    return {"analysis_json": str(analysis_path), "analysis_json_sha256": ext.sha256_file(analysis_path),
            "all_pass": bool(ok), "checks": checks}


# --- comparison with the scratchpad values ------------------------------------------------------

def compare_scratchpad(results: dict) -> list[dict]:
    rows = []

    def add(item, expected, got, mc_range=None):
        same = None if got is None else round(got, 3) == round(expected, 3)
        row = {"item": item, "scratchpad": expected, "reproduced": got,
               "reproduced_3dp": None if got is None else round(got, 3), "match_3dp": same}
        if mc_range is not None:
            row["monte_carlo_range_other_seeds"] = mc_range
            row["scratchpad_within_monte_carlo_range"] = mc_range[0] - 0.0005 <= expected <= mc_range[1] + 0.0005
        rows.append(row)

    s = SCRATCHPAD_2026_10_02
    add("always_negative.agreement", s["always_negative"]["agreement"], results["always_negative"]["agreement"])
    add("always_negative.matches", s["always_negative"]["matches"], results["always_negative"]["matches"])
    for k, v in s["balanced_accuracy"].items():
        add(f"balanced_accuracy.{k}", v, results["balanced_accuracy"].get(k, {}).get("balanced_accuracy"))
    for pair, v in s["both_usable"].items():
        got = results["both_usable"].get(pair)
        a, b = pair.split(":")
        add(f"both_usable.{pair}.rows", v["rows"], got and got["rows"])
        add(f"both_usable.{pair}.{a}", v[a], got and got[a]["agreement"])
        add(f"both_usable.{pair}.{b}", v[b], got and got[b]["agreement"])
    for pair, (est, lo, hi) in s["differences"].items():
        got = results["differences"].get(pair)
        iv = (got or {}).get("interval_95") or [None, None]
        add(f"differences.{pair}.estimate", est, got and got["estimate"])
        mcs = (got or {}).get("monte_carlo_check") or []
        lr = [min(m["lower_range"][0] for m in mcs), max(m["lower_range"][1] for m in mcs)] if mcs else None
        ur = [min(m["upper_range"][0] for m in mcs), max(m["upper_range"][1] for m in mcs)] if mcs else None
        add(f"differences.{pair}.lower", lo, iv[0], lr)
        add(f"differences.{pair}.upper", hi, iv[1], ur)
    return rows


# --- driver -----------------------------------------------------------------------------------

def analyse(prep_dir: Path, split: str, predictions: dict[str, Path], kinds: dict[str, str], balanced: list[str],
            both: list[tuple[str, str]], differences: list[tuple[str, str]], resamples: int, seed: str,
            mc_check: int = 0, mc_resamples: list[int] | None = None) -> tuple[dict, dict]:
    data = load(prep_dir, split, predictions, kinds)
    ids, ref, states = data["ids"], data["ref"], data["states"]
    for name in set(balanced) | {x for p in both + differences for x in p}:
        if name not in states:
            raise ValueError(f"{name}: no --predictions given")
    res: dict = {"always_negative": always_negative(ids, ref),
                 "agreement": {n: agreement(st, ref, ids) for n, st in states.items()},
                 "balanced_accuracy": {n: balanced_accuracy(states[n], ref, ids) for n in balanced},
                 "both_usable": {f"{a}:{b}": both_usable(states[a], states[b], ref, ids, a, b) for a, b in both},
                 "differences": {}}
    for a, b in differences:
        d = paired_cluster_bootstrap(correct_map(states[a], ref, ids), correct_map(states[b], ref, ids),
                                     data["cluster_of"], resamples, f"{seed}|{a}-{b}")
        if mc_check > 0:
            # Monte Carlo check: the same interval under other seeds (and, optionally, other resample
            # counts), to show how far the endpoints move from resampling noise alone. The reported
            # interval is always the one from SEED with --bootstrap resamples.
            d["monte_carlo_check"] = []
            for n_res in (mc_resamples or [resamples]):
                ends = [paired_cluster_bootstrap(correct_map(states[a], ref, ids), correct_map(states[b], ref, ids),
                                                 data["cluster_of"], n_res, f"{seed}|mc{k}|{a}-{b}")["interval_95"]
                        for k in range(1, mc_check + 1)]
                d["monte_carlo_check"].append({"resamples": n_res, "other_seeds": mc_check,
                                               "seed_pattern": f"{seed}|mcK|{a}-{b}, K = 1..{mc_check}",
                                               "lower_range": [min(e[0] for e in ends), max(e[0] for e in ends)],
                                               "upper_range": [min(e[1] for e in ends), max(e[1] for e in ends)]})
        res["differences"][f"{a}:{b}"] = {"a": a, "b": b, **d}
    return data, res


def readme(out: dict, command: str) -> str:
    L = ["# Post-review sensitivity analysis 01 (Study S1-R2, main split)", "",
         f"**{LABEL}**", "",
         f"Generated {out['generated_at_sgt']} (SGT) by `tools/{out['tool']}` {out['tool_version']} for Task 9.11.1. "
         "Offline; no model calls. All numbers: [results.json](results.json).", "",
         "## What this is", "",
         "The external review of 2 October 2026 asked how the Study S1 results look against a majority-class baseline, "
         "under balanced accuracy, on rows where both judges gave usable outputs, and with intervals that respect "
         "responses sharing an objective. These statistics answer that. They were not prespecified. The confirmatory "
         "result of the S1-R2 amendment (exact McNemar for J1:J0, p = 0.046 in analysis_main_03) keeps its status; "
         "the cluster interval below is a sensitivity check on it, not a replacement.", "",
         "## Definitions (same as analysis_main_03)", "",
         "- Reference: five-labeller median of the StrongREJECT 'jailbroken' rating; positive if median >= 0.5 (primary cut).",
         "- Loading and label mapping are imported from `tools/study_s1_external_reference.py`, the tool that produced analysis_main_03.",
         "- Missing, invalid and abstaining outputs are never positive or negative; they count as incorrect. "
         "So balanced accuracy uses all reference-positive rows for sensitivity and all reference-negative rows for "
         "specificity. The usable-only variant is in `results.json` (`balanced_accuracy.*.usable_only`).",
         f"- Paired differences: agreement of A minus agreement of B over the common rows; 95% percentile interval from "
         f"{out['bootstrap']['resamples']} resamples of whole objectives ({out['objectives']} objectives) with replacement; "
         f"seed `{out['bootstrap']['seed']}|A-B` per pair (Python `random.Random`).", "",
         "## Results (primary cut)", "",
         f"- Always-negative baseline: {out['results']['always_negative']['matches']} / {out['results']['always_negative']['n']} "
         f"= {out['results']['always_negative']['agreement']:.3f} (`results.always_negative.agreement`).", "",
         "| Configuration | Agreement | Sensitivity | Specificity | Balanced accuracy | Balanced accuracy, usable outputs only |",
         "|---|---|---|---|---|---|"]
    for n, b in out["results"]["balanced_accuracy"].items():
        g = out["results"]["agreement"][n]
        L.append(f"| {n} | {g['agreement']:.3f} | {b['sensitivity']:.3f} | {b['specificity']:.3f} | {b['balanced_accuracy']:.3f} | "
                 f"{b['usable_only']['balanced_accuracy']:.3f} |")
    L += [""]
    for pair, b in out["results"]["both_usable"].items():
        a, c = pair.split(":")
        L.append(f"- Both usable ({pair}): {b['rows']} rows; {a} {b[a]['agreement']:.3f}, {c} {b[c]['agreement']:.3f} "
                 f"(`results.both_usable[\"{pair}\"]`).")
    L += ["", "| Difference | Common rows | Estimate | 95% cluster-bootstrap interval | Includes 0 |", "|---|---|---|---|---|"]
    for pair, d in out["results"]["differences"].items():
        iv = d["interval_95"]
        L.append(f"| {d['a']} − {d['b']} | {d['common_rows']} | {d['estimate']:+.3f} | {iv[0]:+.3f} to {iv[1]:+.3f} | "
                 f"{'yes' if d['interval_includes_zero'] else 'no'} |")
    mc = [(p, m) for p, d in out["results"]["differences"].items() for m in d.get("monte_carlo_check", [])]
    if mc:
        L += ["", f"Monte Carlo check: each interval recomputed with {mc[0][1]['other_seeds']} other seeds, at each "
              "resample count shown. The reported interval above always uses the fixed seed and "
              f"{out['bootstrap']['resamples']} resamples; this only shows how far the endpoints move from resampling "
              "noise alone.", "",
              "| Difference | Resamples | Lower end, range over other seeds | Upper end, range over other seeds |",
              "|---|---|---|---|"]
        for p, m in mc:
            L.append(f"| {p.replace(':', ' − ')} | {m['resamples']} | {m['lower_range'][0]:+.4f} to {m['lower_range'][1]:+.4f} | "
                     f"{m['upper_range'][0]:+.4f} to {m['upper_range'][1]:+.4f} |")
    if out.get("scratchpad_comparison"):
        L += ["", "## Comparison with the 2 October scratchpad values", "",
              "The Task 9.11 plan lists values from a scratchpad computation of 2 October 2026 that was not saved; its "
              "seed and number of resamples are not recorded. Those values were not used in any computation, and the "
              "tool was not tuned to match them. For interval ends, the last column says whether the scratchpad value "
              "lies inside the range the same end takes over the other seeds of the Monte Carlo check.", "",
              "| Item | Scratchpad | Reproduced | Same to 3 d.p. | Scratchpad inside Monte Carlo range |", "|---|---|---|---|---|"]
        for r in out["scratchpad_comparison"]:
            w = r.get("scratchpad_within_monte_carlo_range")
            rng = r.get("monte_carlo_range_other_seeds")
            wtxt = "-" if w is None else f"{'yes' if w else 'no'} ({rng[0]:+.4f} to {rng[1]:+.4f})"
            L.append(f"| {r['item']} | {r['scratchpad']} | {r['reproduced']} | {'yes' if r['match_3dp'] else 'no'} | {wtxt} |")
    if out.get("check_against"):
        ca = out["check_against"]
        L += ["", "## Cross-check against analysis_main_03", "",
              f"`{ca['analysis_json']}` (sha256 `{ca['analysis_json_sha256']}`): "
              f"{'all checks pass' if ca['all_pass'] else 'SOME CHECKS FAIL'} - prediction hashes, primary-cut matches, "
              "sensitivity, specificity and the McNemar correct counts are identical."]
    L += ["", "## Inputs", "", "| Name | Kind | File | sha256 |", "|---|---|---|---|"]
    for f, h in out["reference_files"].items():
        L.append(f"| reference | {'population and split' if f == 'population.jsonl' else 'five-labeller medians'} | "
                 f"`{out['prep_dir']}/{f}` | `{h}` |")
    for n, i in out["inputs"].items():
        L.append(f"| {n} | {i['kind']} judge output | `{i['path']}` | `{i['sha256']}` |")
    L += ["", "## Reproduce (no model calls; from the repository root)", "", "```bash", command, "```", "",
          "On the Mac, `.venv/bin/python` can replace `python3`. Only the standard library is needed. "
          "Rerunning with the same inputs, seed and resamples gives identical `results` (only the timestamps change).", "",
          "## Limits", "",
          "- Exploratory and post hoc; no correction for multiple comparisons.",
          "- The cluster bootstrap with 29 objectives is approximate; percentile intervals can be slightly narrow with few clusters.",
          "- The both-usable subset is chosen by the judges' own outputs, so it is not a random subset; it shows what "
          "invalid outputs cost J1, not how J1 would perform if it never failed.",
          "- The reference measures a 'jailbroken' rating, not the FYP two-label construct (see analysis_main_03)."]
    return "\n".join(L) + "\n"


def parse_pairs(items: list[str], sep: str, label: str) -> list[tuple[str, str]]:
    out = []
    for item in items:
        if sep not in item:
            raise ValueError(f"{label}: expected A{sep}B, got {item!r}")
        a, b = item.split(sep, 1)
        out.append((a, b))
    return out


def run(args: argparse.Namespace, argv: list[str]) -> int:
    if args.out_dir.exists():
        raise FileExistsError(f"{args.out_dir} already exists")
    predictions = {k: Path(v) for k, v in parse_pairs(args.predictions, "=", "--predictions")}
    kinds = dict(parse_pairs(args.kinds, "=", "--kinds"))
    kinds.setdefault("J0", "severity")
    kinds.setdefault("B0", "refusal_floor")
    both = parse_pairs(args.both_usable, ":", "--both-usable")
    diffs = parse_pairs(args.differences, ":", "--differences")
    data, res = analyse(args.prep_dir, args.split, predictions, kinds, args.balanced, both, diffs, args.bootstrap, args.seed,
                        args.mc_check, args.mc_resamples)
    out = {"label": LABEL, "tool": Path(__file__).name, "tool_version": TOOL_VERSION,
           "generated_at_utc": datetime.now(timezone.utc).isoformat(),
           "generated_at_sgt": datetime.now(SGT).strftime("%d %B %Y, %H:%M"),
           "prep_dir": str(args.prep_dir), "split": args.split, "rows": len(data["ids"]), "objectives": data["objectives"],
           "reference_positive_primary": sum(data["ref"].values()),
           "reference_files": {"population.jsonl": ext.sha256_file(args.prep_dir / "population.jsonl"),
                               "private/external_reference.jsonl": ext.sha256_file(args.prep_dir / "private" / "external_reference.jsonl")},
           "inputs": data["inputs"], "bootstrap": {"resamples": args.bootstrap, "seed": args.seed, "cluster": "objective_id",
                                                  "interval": "percentile, 2.5% and 97.5%"},
           "invalid_and_abstain_handling": "counted as incorrect (neither positive nor negative), as in analysis_main_03",
           "results": res}
    if args.compare_scratchpad:
        out["scratchpad_comparison"] = compare_scratchpad(res)
    if args.check_against:
        out["check_against"] = check_against(args.check_against, data, res)
    command = "python3 -B tools/study_s1_posthoc_sensitivity.py " + " ".join(shlex.quote(a) for a in argv)
    out["command"] = command
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "results.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "README.md").write_text(readme(out, command), encoding="utf-8")
    bad = [r["item"] for r in out.get("scratchpad_comparison", []) if not r["match_3dp"]]
    print(f"WROTE {args.out_dir}; {len(data['ids'])} rows / {data['objectives']} objectives; "
          f"check_against={'n/a' if not args.check_against else out['check_against']['all_pass']}; "
          f"scratchpad mismatches: {bad or 'none'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--prep-dir", type=Path, required=True)
    r.add_argument("--split", choices=["main", "development"], default="main")
    r.add_argument("--predictions", nargs="+", required=True, help="NAME=path.jsonl per configuration")
    r.add_argument("--kinds", nargs="*", default=[], help="NAME=severity|two_label|refusal_floor (defaults: J0 severity, B0 refusal_floor, others two_label)")
    r.add_argument("--balanced", nargs="*", default=[], help="configurations for balanced accuracy")
    r.add_argument("--both-usable", nargs="*", default=[], help="A:B pairs for agreement on rows where both are usable")
    r.add_argument("--differences", nargs="*", default=[], help="A:B pairs for paired differences (A minus B)")
    r.add_argument("--bootstrap", type=int, default=DEFAULT_RESAMPLES)
    r.add_argument("--seed", default=DEFAULT_SEED)
    r.add_argument("--compare-scratchpad", action="store_true",
                   help="add a table comparing the results with the 2 October 2026 scratchpad values (Task 9.11 plan)")
    r.add_argument("--mc-check", type=int, default=0, help="also rerun each interval with N other seeds and report the endpoint range")
    r.add_argument("--mc-resamples", type=int, nargs="*", default=None,
                   help="resample counts for the Monte Carlo check (default: the --bootstrap value)")
    r.add_argument("--check-against", type=Path, help="existing analysis.json to cross-check loading against")
    r.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run(args, argv)
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
