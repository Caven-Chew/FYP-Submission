"""The supervisor-facing PDF report."""
from __future__ import annotations

import io
from datetime import datetime, timezone
from ..attacks import by_slug
from ..models import Benchmark
from ..engine import SUCCESS_SEVERITY

from .charts import curve_drawing
from .evidence import mask_keys
from .text import break_long_words, html_escape, shorten


# ---------- PDF report (reportlab) --------------------------------------

SEV_COLORS = {1: "#16a34a", 2: "#84cc16", 3: "#eab308", 4: "#f97316", 5: "#dc2626"}


# Deck wording (finding #65), so the PDF and the run page use one scale.
SEV_LABELS = {1: "No failure", 2: "Minor, not a failure", 3: "Partial failure", 4: "Clear failure",
              5: "Severe failure"}


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
        "SC3098 LLM Safety Benchmark", small
    ))
    story.append(Spacer(1, 0.4 * cm))
    if any(slug in {"custom_prompt_list", "custom_conversation"} for slug in b.attack_slugs):
        story.append(Paragraph(
            "This PDF is a summary: replies are shortened and multi-turn conversations show "
            "their combined text. For complete evidence use Download CSV.", body))
        story.append(Spacer(1, 0.3 * cm))

    if b.llm_config.get("JUDGE_METHOD") == "dual_v1_v2":
        story.append(Paragraph("Both judges are saved. Aggregate statistics below use v1 (original FYP harm severity). v2 assesses five criteria; its overall score is their maximum. The scores measure different things.", body))
        story.append(Spacer(1, 8))

    # ----- Summary table -----
    sev_values = [p.severity for p in b.prompts if p.severity is not None]
    avg_sev = round(sum(sev_values) / len(sev_values), 2) if sev_values else None

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
        ["Models used", b.llm_config_summary],
        ["Assessment", "Original harm severity (Version 1)" if b.llm_config.get("JUDGE_METHOD") == "j0_severity" else b.llm_config.get("CUSTOM_RUBRIC", "Dataset rubric")],
        ["Prompts sent", str(b.total_prompts or 0)],
        [f"Successful (severity ≥ {SUCCESS_SEVERITY})", str(b.total_successes or 0)],
        ["Risk score (0–100)", f"{b.risk_score:.1f}"],
        ["Risk score 95% CI",
         f"{b.risk_ci['low']:.1f}–{b.risk_ci['high']:.1f}  (Wilson, n={b.risk_ci['n']})"],
        ["Objectives reached (best-of-N)",
         f"{b.best_of_n['reached']}/{b.best_of_n['groups']} ({b.best_of_n['rate']:.1f}%)"
         f" over up to {b.best_of_n['max_attempts']} attempt(s)"],
        ["Average severity (1–5)", f"{avg_sev}" if avg_sev is not None else "—"],
        ["Average latency", f"{b.avg_latency_ms:.0f} ms" if b.avg_latency_ms else "—"],
    ]
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
    if b.llm_config.get("ASSESSMENT_RULES") and b.llm_config.get("JUDGE_METHOD") != "j0_severity":
        story.append(Paragraph("Saved target rules", h2))
        # Separate flowables allow long policies to continue across pages.
        for line in mask_keys(b.llm_config["ASSESSMENT_RULES"]).splitlines():
            story.append(Paragraph(html_escape(line or " "), body))
        story.append(Spacer(1, 0.3 * cm))

    # ----- Failure detail (only if benchmark failed) -----
    if b.status in ("failed", "completed_with_target_errors") and b.status_detail:
        story.append(Paragraph("Failure detail", h2))
        # Truncate to keep the PDF compact; full trace lives in CSV / DB.
        detail = b.status_detail[:1500]
        if len(b.status_detail) > 1500:
            detail += "  … (truncated)"
        story.append(Paragraph(html_escape(detail).replace("\n", "<br/>"), code))
        story.append(Spacer(1, 0.4 * cm))

    # ----- Severity distribution -----
    sev_hist = [0, 0, 0, 0, 0]
    for s in sev_values:
        if 1 <= s <= 5:
            sev_hist[s - 1] += 1
    max_count = max(sev_hist) or 1

    story.append(Paragraph("Severity distribution", h2))
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
    tax_buckets: dict[str, dict] = {}
    for p in b.prompts:
        key = p.taxonomy_category or "(uncategorized)"
        agg = tax_buckets.setdefault(key, {"total": 0, "succ": 0, "sev_sum": 0, "sev_n": 0})
        agg["total"] += 1
        if p.succeeded:
            agg["succ"] += 1
        if p.severity:
            agg["sev_sum"] += p.severity
            agg["sev_n"] += 1
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

    if tax_buckets:
        story.append(Paragraph("By prompt-injection taxonomy", h2))
        rows = [["Taxonomy", "Total", "Succeeded", "Rate", "Avg severity"]]
        for tax, v in sorted(tax_buckets.items(), key=lambda kv: -kv[1]["total"]):
            rate = f"{100 * v['succ'] / v['total']:.0f}%" if v["total"] else "—"
            avg_s = f"{v['sev_sum']/v['sev_n']:.2f}" if v["sev_n"] else "—"
            rows.append([tax, str(v["total"]), str(v["succ"]), rate, avg_s])
        t = Table(rows, colWidths=[6 * cm, 1.5 * cm, 2 * cm, 1.5 * cm, 2.5 * cm])
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
    rollup_rows = [["Attack", "Category", "Total", "Succeeded", "Rate", "Avg severity"]]
    for slug, prompts in grouped.items():
        atk = by_slug(slug) or {"name": slug, "category": "?"}
        sevs = [p.severity for p in prompts if p.severity is not None]
        succ = sum(1 for p in prompts if p.succeeded)
        rate = f"{100 * succ / len(prompts):.0f}%" if prompts else "—"
        rollup_rows.append([
            atk["name"], atk["category"], str(len(prompts)), str(succ), rate,
            f"{sum(sevs)/len(sevs):.2f}" if sevs else "—",
        ])
    t = Table(rollup_rows, colWidths=[5.5 * cm, 3.5 * cm, 1.5 * cm, 2 * cm, 1.5 * cm, 2.5 * cm])
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
            key=lambda p: (-(p.severity or 0), p.sequence),
        )[:5]

        story.append(Paragraph(
            f"<b>{html_escape(atk['name'])}</b>  "
            f"<font color='#64748b'>({len(prompts)} prompts, "
            f"{sum(1 for p in prompts if p.succeeded)} succeeded)</font>",
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
            sev_cell = str(p.severity) if p.severity else "—"
            rows.append([
                _para(str(p.sequence)),
                _para(sev_cell),
                _para(("Converter input — not sent to target: " if p.converter_not_sent else "") + shorten(p.prompt_text, 140)),
                _para("No target request or response. Converter: " + shorten(mask_keys(p.error or p.note or p.status), 150) if p.converter_not_sent else shorten(mask_keys(p.response, b.llm_config.get("CANARY")) or "(empty)", 200)),
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
            if p.severity:
                ts.append(("BACKGROUND", (1, i), (1, i), colors.HexColor(SEV_COLORS[p.severity])))
                ts.append(("TEXTCOLOR",  (1, i), (1, i), colors.white))
                ts.append(("FONTNAME",   (1, i), (1, i), "Helvetica-Bold"))
        t.setStyle(TableStyle(ts))
        story.append(KeepTogether([t, Spacer(1, 0.3 * cm)]))

    if b.llm_config.get("JUDGE_METHOD") == "dual_v1_v2":
        from ..engine.scoring import judge_version_results
        story.append(PageBreak())
        story.append(Paragraph("Both judge results", h2))
        for p in sorted(b.prompts, key=lambda row: (row.attack_slug, row.sequence)):
            story.append(Paragraph(html_escape(f"{p.attack_slug} — prompt {p.sequence + 1}"), h2))
            for version, result in judge_version_results(p.verdict).items():
                score = result['score'] if result['score'] is not None else 'unscored'
                story.append(Paragraph(html_escape(f"{version} score: {score}"), body))
                for line in mask_keys(result['verdict'], b.llm_config.get('CANARY')).splitlines():
                    story.append(Paragraph(html_escape(break_long_words(line or ' ')), small))
                story.append(Spacer(1, 8))
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
