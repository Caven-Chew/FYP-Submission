"""Turning a Benchmark's rows into the summary numbers the pages display.

Two rules govern everything in here, and both exist because breaking them
produced published numbers that were wrong:

1. **Only rows that reached the target count.** A converter that refused, a
   send that failed, an attack that never started — these are absences of
   data, not zeros. `Prompt.counts_toward_stats` is the single predicate; when
   an attack has none, its rate is None ("no data"), never 0%.

2. **Rates are per objective, not per row.** Best-of-N sends one seed up to N
   times and stops at the first success, so a broken seed leaves one row and an
   unbroken one leaves N. Dividing successes by rows measures neither thing:
   benchmark 57 displayed harmbench_persuasion at 11.1% when 3 of its 10 seeds
   actually fell (30%), and put that number on the same axis as a
   single-attempt attack. `Prompt.group_key` is the unit.

3. **Severities from different rubrics are not the same quantity.** Most attacks
   are scored on the 1-5 HARM rubric; system-prompt extraction is scored on the
   1-5 DISCLOSURE rubric, because nothing harmful is requested and a harm judge
   scores a complete leak a 1. The two share a range and a parser and mean
   entirely different things, so pooling them into one histogram or one mean
   produces a number with no referent. A rate (broken / measured) IS comparable
   across rubrics — "the attack achieved what it set out to" is well defined
   either way — so rates stay pooled and only the SEVERITY aggregates split.
   When a run mixes rubrics the pooled mean is None, per rule 1's habit of
   preferring "no number" to a wrong one, and `rubric_stats` carries one
   histogram and mean per rubric.
"""
from __future__ import annotations

import itertools
from ..attacks import by_slug
from ..models import Benchmark


def benchmark_totals(b) -> dict:
    """The headline numbers for one benchmark.

    Thin wrapper over Benchmark.compute_totals, which lives on the model
    because engine/ needs it too and engine must never import web/
    (settings/prompts <- engine/targets <- web). Kept as a function here so
    this package has one obvious import site for it.
    """
    return b.compute_totals()

#: severity_source values whose numbers were invented rather than measured, and
#: so must never enter a histogram or a mean. Nothing writes these any more (see
#: the Stage 2 fix in engine/dynamic_attack.py and engine/azure_attack.py); the
#: guard stays so a future path cannot quietly reintroduce one.
_FABRICATED_SEVERITY_SOURCES = {"synthetic"}


def rubric_for(attack_slug: str) -> str:
    """Which 1-5 scale an attack's rows are on: "harm" (default) or "leak".

    Read from the catalog rather than stored per row: the rubric is a property
    of the attack, and a row written before the leak rubric existed was still
    judged by whatever its attack declared at the time.
    """
    return (by_slug(attack_slug) or {}).get("scoring", "harm")


def _scored(prompts):
    """Rows carrying a severity we actually measured."""
    return [p for p in prompts
            if p.counts_toward_stats
            and p.severity is not None
            and 1 <= p.severity <= 5
            and p.severity_source not in _FABRICATED_SEVERITY_SOURCES]


def _group_stats(prompts) -> dict:
    """Collapse rows into per-objective counts.

    `seeds_attempted` counts every objective the attack tried, including ones
    where nothing usable came back — that difference is what tells "0 of 10
    seeds fell" apart from "10 seeds were never actually attacked".
    """
    attempted: set[str] = set()
    captured: set[str] = set()
    measured: set[str] = set()
    broken: set[str] = set()
    counted_rows = replied_rows = 0
    for p in prompts:
        key = p.group_key
        attempted.add(key)
        if p.has_reply:
            captured.add(key)
            replied_rows += 1
        if not p.counts_toward_stats:
            continue
        counted_rows += 1
        measured.add(key)
        if p.succeeded:
            broken.add(key)
    scored = _scored(prompts)
    return {
        "seeds_attempted": len(attempted),
        "seeds_captured": len(captured),
        "seeds_measured": len(measured),
        "seeds_broken": len(broken),
        "rows": len(prompts),
        "rows_counted": counted_rows,
        "rows_excluded": len(prompts) - counted_rows,
        "rows_unscored": max(0, replied_rows - counted_rows),
        "rows_no_reply": len(prompts) - replied_rows,
        # None, not 0.0 — an attack with nothing to measure has no rate, and
        # rendering it as 0% reads as "the target resisted everything".
        "rate": (100.0 * len(broken) / len(measured)) if measured else None,
        "avg_severity": (round(sum(p.severity for p in scored) / len(scored), 2)
                         if scored else None),
        # Legacy aliases. The detail page and its live-refresh JS compute
        # `succeeded / total` themselves, so pointing these at the per-objective
        # numbers fixes those displays without touching their arithmetic, and
        # `total == 0` is already their "—" case.
        "total": len(measured),
        "succeeded": len(broken),
    }


def taxonomy_rollup(b) -> dict:
    """Per-taxonomy {total, succ, rate} for one benchmark.

    Counts objectives that produced a reply — not rows, and not attempts that
    never reached the target.
    """
    buckets: dict[str, dict] = {}
    for p in b.prompts:
        if not p.counts_toward_stats:
            continue
        agg = buckets.setdefault(p.taxonomy_category or "(uncategorised)",
                                 {"measured": set(), "broken": set()})
        agg["measured"].add(p.group_key)
        if p.succeeded:
            agg["broken"].add(p.group_key)
    out: dict[str, dict] = {}
    for cat, agg in buckets.items():
        total, succ = len(agg["measured"]), len(agg["broken"])
        out[cat] = {
            "total": total, "succ": succ,
            "rate": round(100.0 * succ / total, 1) if total else 0.0,
        }
    return out


def benchmark_rollups(b: Benchmark) -> dict:
    """Compute the live-updatable rollups for a benchmark from its Prompt rows.

    Shared by the detail page (initial server render) and the status.json poll
    endpoint, so the numbers shown live during a run match the final render
    exactly. Returns grouped, rollup, tax_rollup, sev_hist, avg_severity and
    totals.
    """
    grouped: dict[str, list] = {}
    for p in sorted(b.prompts, key=lambda x: (x.attack_slug, x.sequence)):
        grouped.setdefault(p.attack_slug, []).append(p)

    # Thread multi-turn (dynamic-attack) prompts into per-objective conversation
    # blocks so a Crescendo / Red-Teaming run displays as an escalating exchange
    # instead of a flat list of rows indistinguishable from single-turn static
    # prompts. Rows within one slug are already ordered by `sequence`, and every
    # turn of one objective shares its `objective_id` and is written contiguously
    # (see run_dynamic_attack's `sequence: obj_idx * 100 + i`), so grouping
    # consecutive equal objective_ids (no sort needed) recovers each conversation
    # intact. Static/Azure attacks never set objective_id, so their rows all fall
    # into "singles" runs and render exactly as before — no behaviour change there.
    display_groups: dict[str, list[dict]] = {}
    for slug, prompts in grouped.items():
        blocks: list[dict] = []
        for objective_id, turns_iter in itertools.groupby(prompts, key=lambda p: p.objective_id):
            turns = list(turns_iter)
            if objective_id and (len(turns) > 1 or (slug == "custom_conversation"
                    and b.llm_config.get("SCRIPTED_SCORE_UNIT") == "turn")):
                blocks.append({"kind": "conversation", "objective_id": objective_id, "turns": turns})
            elif blocks and blocks[-1]["kind"] == "singles":
                blocks[-1]["prompts"].extend(turns)
            else:
                blocks.append({"kind": "singles", "prompts": turns})
        display_groups[slug] = blocks

    # Completed sections follow their saved completion order, oldest first.
    # Unfinished/legacy sections follow first capture order; turn order stays intact.
    completion = {slug: i for i, slug in enumerate(b.completed_attacks)}
    display_groups = dict(sorted(display_groups.items(), key=lambda item: (
        item[0] not in completion, completion.get(item[0], 0),
        min((p.id or 0 for p in grouped[item[0]]), default=0))))

    # Per-attack rollup: objectives broken / objectives measured.
    rollup = []
    for slug, prompts in grouped.items():
        atk = by_slug(slug) or {"name": slug, "category": "?"}
        entry = {"slug": slug, "name": atk["name"], "category": atk["category"],
                 "rubric": next((p.rubric for p in prompts if getattr(p, "rubric", None)), atk.get("scoring", "harm"))}
        entry.update(_group_stats(prompts))
        rollup.append(entry)

    # Severity histogram (counts of 1..5) + overall average, over measured
    # severities only.
    sev_hist = [0, 0, 0, 0, 0]
    scored = _scored(b.prompts)
    for p in scored:
        sev_hist[p.severity - 1] += 1

    # One histogram and mean per rubric. `sev_hist` above stays pooled because
    # it is a distribution the page draws as bars, and the per-rubric split
    # below is what the reader is pointed at; the pooled MEAN, which invites a
    # single "average severity 2.8" claim, is withheld when the run mixes
    # rubrics — averaging a harm score with a disclosure score yields a number
    # that answers no question.
    rubric_stats: dict[str, dict] = {}
    for p in scored:
        st = rubric_stats.setdefault(
            ("harm" if (getattr(b, "llm_config", None) or {}).get("JUDGE_METHOD") == "dual_v1_v2" else (getattr(p, "rubric", None) or rubric_for(p.attack_slug))), {"hist": [0, 0, 0, 0, 0], "n": 0, "sum": 0})
        st["hist"][p.severity - 1] += 1
        st["n"] += 1
        st["sum"] += p.severity
    for st in rubric_stats.values():
        st["avg"] = round(st["sum"] / st["n"], 2) if st["n"] else None
    mixed_rubrics = len(rubric_stats) > 1
    avg_severity = (
        None if mixed_rubrics
        else (round(sum(p.severity for p in scored) / len(scored), 2) if scored else None)
    )

    # Per-taxonomy rollup (prompt-injection categories).
    tax_buckets: dict[str, dict] = {}
    for p in b.prompts:
        if not p.has_reply:
            continue
        agg = tax_buckets.setdefault(
            p.taxonomy_category or "(uncategorized)",
            {"captured": set(), "measured": set(), "broken": set(), "sev_sum": 0, "sev_n": 0,
             "rubrics": set()})
        agg["captured"].add(p.group_key)
        if not p.counts_toward_stats:
            continue
        agg["measured"].add(p.group_key)
        if p.succeeded:
            agg["broken"].add(p.group_key)
    for p in scored:
        agg = tax_buckets.get(p.taxonomy_category or "(uncategorized)")
        if agg is not None:
            agg["sev_sum"] += p.severity
            agg["sev_n"] += 1
            agg["rubrics"].add(("harm" if (getattr(b, "llm_config", None) or {}).get("JUDGE_METHOD") == "dual_v1_v2" else (getattr(p, "rubric", None) or rubric_for(p.attack_slug))))
    tax_rollup = [
        {
            "category": k,
            "captured": len(v["captured"]),
            "total": len(v["measured"]), "succeeded": len(v["broken"]),
            "rate": (100.0 * len(v["broken"]) / len(v["measured"])) if v["measured"] else None,
            # None rather than a mean across two rubrics — same reason as above.
            "avg_severity": (round(v["sev_sum"] / v["sev_n"], 2)
                             if v["sev_n"] and len(v["rubrics"]) <= 1 else None),
            "rubrics": sorted(v["rubrics"]),
        }
        for k, v in sorted(tax_buckets.items(), key=lambda kv: -len(kv[1]["measured"]))
    ]

    return {
        "grouped": grouped, "display_groups": display_groups, "rollup": rollup,
        "tax_rollup": tax_rollup, "sev_hist": sev_hist, "avg_severity": avg_severity,
        "rubric_stats": rubric_stats, "mixed_rubrics": mixed_rubrics,
        "totals": benchmark_totals(b),
    }
