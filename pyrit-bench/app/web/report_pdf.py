"""The supervisor-facing PDF report."""
from __future__ import annotations

import io
from datetime import datetime, timezone
from ..attacks import by_slug
from ..models import Benchmark
from ..engine import SUCCESS_SEVERITY

from .charts import curve_drawing
from .rollups import benchmark_rollups, two_label_rollup
from .text import break_long_words, html_escape, shorten
from ..engine.two_label import run_method


PDF_OUTCOME_HEADERS = {"no": "No", "partial": "Partial", "complete": "Complete",
                       "abstained": "Judge abstained", "input_review": "Input needs review",
                       "no_requirements": "No req. list", "technical_failure": "Technical failure",
                       "pending": "Not yet assessed"}


def _pct(value):
    return "" if value is None else f" ({value:.1f}%)"


def _outcome_table(t, keys, cell):
    """One objective-label x fulfilment table from rollups._table."""
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Table, TableStyle
    rows = [[Paragraph("<b>Objective label</b>", cell), Paragraph("<b>Responses</b>", cell)]
            + [Paragraph(f"<b>{PDF_OUTCOME_HEADERS[k]}</b>", cell) for k in keys]]
    for group in [g for g in t["rows"] if g["n"]] + [{"label": "All", **t["total"]}]:
        rows.append([Paragraph(html_escape(group["label"]), cell), str(group["n"])]
                    + [f"{group['counts'][k]}{_pct(group['shares'][k]) if group['counts'][k] else ''}"
                       for k in keys])
    table = Table(rows, colWidths=[2.9 * cm, 1.4 * cm] + [1.6 * cm] * len(keys), repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def _completion_lines(t) -> list:
    lines = [f"Coverage: {t['total']['valid_decisions']}/{t['n']} responses with a valid No, Partial or "
             f"Complete decision{_pct(t['total']['coverage'])}."]
    for group in t["rows"]:
        if group["key"] in ("Harmful", "Benign") and group["n"]:
            c = group["completion"]
            rate = f"{c['yes']}/{c['yes'] + c['no']} = {c['rate']:.1f}%" if c["rate"] is not None else "N/A"
            lines.append(f"Complete fulfilment on {group['label']} objectives (supplementary, completeness gate): "
                         f"{rate}; completeness coverage {c['yes'] + c['no']}/{c['n']}{_pct(c['coverage'])}; "
                         f"descriptive bounds {c['bound_low']:.1f}-{c['bound_high']:.1f}% (not a confidence interval).")
    return lines


def _two_label_section(b, styles: dict) -> list:
    """'Two-label assessment' section from the shared rollup (Task 7)."""
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    s = two_label_rollup(b)
    rel = b.llm_config.get("TWO_LABEL") or {}
    h2, body, small = styles["h2"], styles["body"], styles["small"]
    cell = styles["cell"]
    keys = [o["key"] for o in s["outcomes"]]

    out = [Paragraph("Two-label assessment", h2), Paragraph(html_escape(
        f"Judge method: {s['method']}. Release {rel.get('release_id', 'not recorded')}, manifest SHA-256 "
        f"{rel.get('release_manifest_sha256', 'not recorded')}, policy {rel.get('policy_id', 'not recorded')}. "
        "Objective labels are model judgements, not human references. Within each unit, shares use all "
        "eligible captured responses as the denominator, including failed and pending assessments; "
        "technical failures and missing requirement lists are shown separately. Static responses and "
        "conversation turns are separate units and are not pooled. Harmful + Partial is partial task "
        "fulfilment, not proof of harmful assistance."), small), Spacer(1, 4)]
    for unit in [u for u in s["units"] if u["n"]]:
        sg = unit["seed_groups"]
        out += [Paragraph(html_escape(f"{unit['label']} ({unit['n']}): all attempts, conditional on the "
                                      "best-of-N stop rule"), body), Spacer(1, 2),
                _outcome_table(unit["all"], keys, cell), Spacer(1, 3)]
        out += [Paragraph(html_escape(line), small) for line in _completion_lines(unit["all"])]
        out += [Spacer(1, 4), Paragraph(html_escape(
            f"{unit['label']}: first attempt only ({unit['first_attempt']['n']}), not affected by the stop rule"),
            body), Spacer(1, 2), _outcome_table(unit["first_attempt"], keys, cell), Spacer(1, 3)]
        out += [Paragraph(html_escape(line), small) for line in _completion_lines(unit["first_attempt"])]
        out.append(Paragraph(html_escape(
            f"Seeds or objectives (best-of-N groups): {sg['groups']}; with a Harmful + Complete candidate "
            f"within the attempt budget {sg['with_candidate']}; verified by review {sg['verified']}."), small))
        out.append(Spacer(1, 6))
    cand, fence = s["candidates"], s["fence_stripped"]
    lines = [f"Harmful + Complete candidates: {cand['candidate']} (pending review {cand['pending_review']}, "
             f"verified {cand['verified']}, rejected {cand['rejected']}).",
             f"Accepted only after removing one code fence (recorded deviation): {fence['objective']} objective "
             f"and {fence['fulfilment']} fulfilment assessment(s)."]
    if s["disclosure_rows"]:
        lines.append(f"{s['disclosure_rows']} system-prompt extraction response(s) are scored on the "
                     "disclosure severity rubric and are not included above.")
    out += [Paragraph(html_escape(line), body) for line in lines]
    if s["per_attack"]:
        rows = [["Attack", "Unit", "Responses", "Coverage", "Harmful completion (known)", "Candidates (verified)"]]
        for a in s["per_attack"]:
            hc = a["harmful_completion"]
            name = a["name"] + (" (judged undecoded)" if a["judged_undecoded"] else "")
            rows.append([Paragraph(html_escape(name), cell),
                         "turns" if a["unit"] == "conversation_turn" else "static",
                         str(a["n"]), f"{a['valid_decisions']}/{a['n']}",
                         (f"{hc['yes']}/{hc['yes'] + hc['no']}" if hc["rate"] is not None else "N/A")
                         + f" ({hc['yes'] + hc['no']}/{hc['n']} known)",
                         f"{a['candidates']} ({a['verified']})"])
        t = Table(rows, colWidths=[5 * cm, 1.4 * cm, 1.8 * cm, 2 * cm, 4 * cm, 3 * cm], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        out += [Spacer(1, 6), t]
        if any(a["judged_undecoded"] for a in s["per_attack"]):
            out.append(Paragraph("Judged undecoded: a flat-encoding attack (ROT13 or Base64). The severity "
                                 "judge also sees a decoded view; the frozen two-label contract assesses the "
                                 "reply exactly as captured (a stated limit).", small))
    out.append(Spacer(1, 0.4 * cm))
    return out


# ---------- PDF report (reportlab) --------------------------------------

SEV_COLORS = {1: "#16a34a", 2: "#84cc16", 3: "#eab308", 4: "#f97316", 5: "#dc2626"}


SEV_LABELS = {1: "Safe", 2: "Mostly safe", 3: "Borderline", 4: "Unsafe", 5: "Severely unsafe"}


def render_benchmark_pdf(b: Benchmark) -> bytes:
    """Build a polished PDF report. Caller wraps response."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title=f"Benchmark report — {b.name}",
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=13)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11, textColor=colors.HexColor("#475569"))
    code = ParagraphStyle("code", parent=body, fontName="Courier", fontSize=8, leading=10,
                          backColor=colors.HexColor("#f8fafc"))

    story: list = []

    # ----- Header / cover -----
    story.append(Paragraph("AI Security Benchmark Report", h1))
    story.append(Paragraph(b.name, h2))
    story.append(Paragraph(
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')}Z &middot; "
        f"PyRIT-Bench (FYP prototype)", small
    ))
    story.append(Spacer(1, 0.4 * cm))

    # ----- Summary table -----
    totals = b.compute_totals()
    roll = benchmark_rollups(b)
    sev_values = [p.severity for p in b.prompts if p.has_valid_score]
    avg_sev = roll["avg_severity"]

    summary_rows = [
        ["Target", f"{b.target.name}  ({b.target.provider}/{b.target.model_name})"],
    ]
    if b.target.architecture:
        summary_rows.append(["Architecture", b.target.architecture])
    summary_rows += [
        ["Status", b.status],
        ["Started",  b.started_at.isoformat(timespec="seconds") if b.started_at else "—"],
        ["Finished", b.finished_at.isoformat(timespec="seconds") if b.finished_at else "—"],
        ["Attacks run", ", ".join(b.attack_slugs)],
        # A Paragraph, not a bare string: a table cell string never wraps, and
        # a summary naming a backup or a change between starts runs long.
        ["Models used", Paragraph(html_escape(b.llm_config_summary),
                                  ParagraphStyle("cell", parent=body, fontSize=9, leading=11))],
        ["Judge method", run_method(b)],
        ["Responses captured", str(totals["counted"])],
    ]
    if run_method(b) == "two_label":
        # No harm severity exists in a two-label run, so "Responses scored 0" and a
        # 0% risk would mislead: report two-label coverage per unit instead.
        tl = two_label_rollup(b)
        for unit in [u for u in tl["units"] if u["n"]]:
            total = unit["all"]["total"]
            summary_rows.append([f"Two-label coverage ({unit['label'].lower()})",
                                 f"{total['valid_decisions']}/{unit['n']} valid No/Partial/Complete decisions"
                                 f"{_pct(total['coverage'])}"])
        summary_rows.append(["Harmful + Complete candidates",
                             f"{tl['candidates']['candidate']} (verified {tl['candidates']['verified']})"])
        if tl["disclosure_rows"]:
            leak = roll["rubric_stats"].get("leak") or {"n": 0, "avg": None}
            summary_rows.append(["Disclosure rubric (system-prompt attacks)",
                                 f"{leak['n']} scored, average {leak['avg'] if leak['avg'] is not None else '—'}; "
                                 f"{totals['successes']} at or above {SUCCESS_SEVERITY}"])
    else:
        summary_rows += [
            ["Responses scored", str(totals["scored"])],
            ["Captured, unscored", str(totals["unscored"])],
            [f"Successful (severity ≥ {SUCCESS_SEVERITY})", str(totals["successes"])],
            ["Risk score (0–100)", f"{b.risk_score:.1f}" if totals["scored"] else "—"],
            ["Risk score 95% CI",
             (f"{b.risk_ci['low']:.1f}–{b.risk_ci['high']:.1f}  (Wilson, n={b.risk_ci['n']})"
              if totals["scored"] else "— (no scored responses)")],
            ["Objectives reached (best-of-N)",
             (f"{b.best_of_n['reached']}/{b.best_of_n['groups']} ({b.best_of_n['rate']:.1f}%)"
              f" over up to {b.best_of_n['max_attempts']} attempt(s)"
              if b.best_of_n["groups"] else "— (no scored objectives)")],
            ["Average severity (1–5)",
             f"{avg_sev}" if avg_sev is not None else ("— (mixed rubrics)" if roll["mixed_rubrics"] else "—")],
        ]
    summary_rows.append(["Average recorded duration", f"{b.avg_latency_ms:.0f} ms" if b.avg_latency_ms else "—"])
    t = Table(summary_rows, colWidths=[4.5 * cm, 12 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))
    if roll["mixed_rubrics"]:
        means = ", ".join(
            f"{name}: {stats['avg']:.2f} (n={stats['n']})"
            for name, stats in sorted(roll["rubric_stats"].items())
        )
        story.append(Paragraph(
            "Average severity is shown separately because the rubrics measure "
            f"different outcomes: {html_escape(means)}.", small))
        story.append(Spacer(1, 0.2 * cm))

    # ----- Failure detail (only if benchmark failed) -----
    if b.status in ("failed", "completed_with_target_errors") and b.status_detail:
        story.append(Paragraph("Failure detail", h2))
        # Truncate to keep the PDF compact; full trace lives in CSV / DB.
        detail = b.status_detail[:1500]
        if len(b.status_detail) > 1500:
            detail += "  … (truncated)"
        story.append(Paragraph(html_escape(detail).replace("\n", "<br/>"), code))
        story.append(Spacer(1, 0.4 * cm))

    method = run_method(b)
    if method != "severity":
        story += _two_label_section(b, {"h2": h2, "body": body, "small": small, "cell": ParagraphStyle(
            "tl_cell", parent=body, fontSize=7.5, leading=9)})

    # ----- Severity distribution -----
    sev_hist = [0, 0, 0, 0, 0]
    for s in sev_values:
        if 1 <= s <= 5:
            sev_hist[s - 1] += 1
    max_count = max(sev_hist) or 1

    story.append(Paragraph("Severity distribution" + (" (not used: a two-label run has no severity scores)"
                                                      if method == "two_label" else ""), h2))
    dist_rows = [["Sev", "Label", "Count", "Distribution"]]
    for sev in range(1, 6):
        count = sev_hist[sev - 1]
        bar_width = int(15 * count / max_count) if max_count else 0
        bar = "█" * bar_width if bar_width else ""
        dist_rows.append([str(sev), SEV_LABELS[sev], str(count), bar])
    t = Table(dist_rows, colWidths=[1 * cm, 4 * cm, 1.5 * cm, 10 * cm])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
    ]
    # Colour the severity number cell
    for i in range(1, 6):
        style.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(SEV_COLORS[i])))
        style.append(("TEXTCOLOR",  (0, i), (0, i), colors.white))
        style.append(("FONTNAME",   (0, i), (0, i), "Helvetica-Bold"))
        style.append(("TEXTCOLOR",  (3, i), (3, i), colors.HexColor(SEV_COLORS[i])))
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    # ----- Per-taxonomy rollup -----
    tax_rows = roll["tax_rollup"]
    curve = b.reach_curve
    if curve and len(curve) > 1:
        story.append(Paragraph("Success vs attempt budget", h2))
        story.append(Paragraph(
            "Share of objectives reached at least once given N attempts each. Derived "
            "from a single run: every attempt is recorded with its index, so the "
            "best-of-N result is known for every N up to the budget used. A curve that "
            "keeps climbing means the target imposes a cost on the attacker rather "
            "than blocking the objective outright; a flat curve near zero is the "
            "stronger safety result.", small))
        story.append(Spacer(1, 6))
        story.append(curve_drawing(curve))
        rows = [["Attempt budget (N)"] + [str(pt["n"]) for pt in curve],
                ["Objectives reached"] + [f"{pt['reached']}/{pt['groups']}" for pt in curve],
                ["Reach rate"] + [f"{pt['rate']:.1f}%" for pt in curve]]
        t = Table(rows, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 14))

    if tax_rows:
        story.append(Paragraph("By prompt-injection taxonomy", h2))
        rows = [["Taxonomy", "Captured", "Scored", "Succeeded", "Rate", "Avg severity"]]
        for v in tax_rows:
            rate = f"{v['rate']:.0f}%" if v["rate"] is not None else "—"
            avg_s = f"{v['avg_severity']:.2f}" if v["avg_severity"] is not None else "—"
            rows.append([v["category"], str(v["captured"]), str(v["total"]),
                         str(v["succeeded"]), rate, avg_s])
        t = Table(rows, colWidths=[5.5 * cm, 1.7 * cm, 1.5 * cm, 2 * cm, 1.5 * cm, 2.7 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4 * cm))

    # ----- Per-attack rollup -----
    grouped: dict[str, list] = {}
    for p in sorted(b.prompts, key=lambda x: (x.attack_slug, x.sequence)):
        grouped.setdefault(p.attack_slug, []).append(p)

    story.append(Paragraph("Per-attack rollup", h2))
    rollup_rows = [["Attack", "Category", "Captured", "Scored", "Broken", "Rate", "Avg severity"]]
    for item in roll["rollup"]:
        rollup_rows.append([
            item["name"], item["category"], str(item["seeds_captured"]),
            str(item["seeds_measured"]), str(item["seeds_broken"]),
            f"{item['rate']:.0f}%" if item["rate"] is not None else "—",
            f"{item['avg_severity']:.2f}" if item["avg_severity"] is not None else "—",
        ])
    t = Table(rollup_rows, colWidths=[4.7 * cm, 2.9 * cm, 1.6 * cm, 1.3 * cm,
                                     1.3 * cm, 1.4 * cm, 2.8 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ----- Per-attack detail with sample prompts -----
    story.append(Paragraph("Sample prompts per attack", h2))
    story.append(Paragraph(
        "Up to five worst-severity prompts per attack shown below. Full per-prompt "
        "data is available via CSV export.", small
    ))
    story.append(Spacer(1, 0.2 * cm))

    for slug, prompts in grouped.items():
        atk = by_slug(slug) or {"name": slug}
        # Sort by severity (highest first), take top 5
        samples = sorted(
            prompts,
            key=lambda p: (-(p.severity or 0) if p.has_valid_score else 0, p.sequence),
        )[:5]

        story.append(Paragraph(
            f"<b>{html_escape(atk['name'])}</b>  "
            f"<font color='#64748b'>({len(prompts)} prompts, "
            f"{sum(1 for p in prompts if p.has_valid_score and p.succeeded)} succeeded)</font>",
            body,
        ))

        cell_style = ParagraphStyle(
            "cell", parent=body, fontSize=8, leading=10,
        )
        hdr_style = ParagraphStyle(
            "cell_hdr", parent=cell_style, fontName="Helvetica-Bold",
        )

        def _para(text: str, style=cell_style) -> "Paragraph":
            # Break first (on raw text), then escape so ReportLab's XML parser
            # sees clean markup with injected spaces as word-break opportunities.
            safe = html_escape(break_long_words(text))
            return Paragraph(safe, style)

        rows = [[
            Paragraph("#", hdr_style),
            Paragraph("Sev", hdr_style),
            Paragraph("Prompt", hdr_style),
            Paragraph("Response (excerpt)", hdr_style),
        ]]
        for p in samples:
            sev_cell = str(p.severity) if p.has_valid_score else "—"
            rows.append([
                _para(str(p.sequence)),
                _para(sev_cell),
                _para(shorten(p.prompt_text, 140)),
                _para(shorten(p.response or "(empty)", 200)),
            ])
        if len(samples) == 0:
            rows.append([_para("—"), _para("—"), _para("no prompts"), _para("—")])
        t = Table(rows, colWidths=[0.8 * cm, 0.9 * cm, 6.5 * cm, 8.8 * cm])
        ts = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        for i, p in enumerate(samples, start=1):
            if p.has_valid_score:
                ts.append(("BACKGROUND", (1, i), (1, i), colors.HexColor(SEV_COLORS[p.severity])))
                ts.append(("TEXTCOLOR",  (1, i), (1, i), colors.white))
                ts.append(("FONTNAME",   (1, i), (1, i), "Helvetica-Bold"))
        t.setStyle(TableStyle(ts))
        story.append(KeepTogether([t, Spacer(1, 0.3 * cm)]))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
