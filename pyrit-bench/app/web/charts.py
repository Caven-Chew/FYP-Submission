"""Inline SVG charts (no JS, no chart library) for the detail and compare pages."""
from __future__ import annotations


# Greyscale line styles. Series are distinguished by dash pattern rather than
# colour so the chart survives mono printing (and photocopying into a report
# appendix) without any series becoming ambiguous.
CURVE_DASHES = ["", "7 3", "2 3", "9 3 2 3", "1 4", "5 2 1 2"]


def curve_drawing(curve, width: int = 440, height: int = 150):
    """Greyscale line chart of the reach curve for the PDF report.

    Drawn in black on grey gridlines rather than colour so the report is legible
    photocopied or printed mono.
    """
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing, Line, String, Circle, PolyLine

    pad_l, pad_b, pad_t, pad_r = 34, 24, 10, 8
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_b - pad_t
    max_n = curve[-1]["n"]
    d = Drawing(width, height)

    # reportlab's y axis points up, unlike SVG
    def x_of(n):
        return pad_l + (n - 1) / (max_n - 1) * plot_w

    def y_of(rate):
        return pad_b + (rate / 100.0) * plot_h

    grey = colors.HexColor("#94a3b8")
    light = colors.HexColor("#dbe2ea")
    for v in (0, 25, 50, 75, 100):
        y = y_of(v)
        d.add(Line(pad_l, y, pad_l + plot_w, y, strokeColor=light, strokeWidth=0.5))
        lbl = String(pad_l - 5, y - 3, f"{v}%", fontSize=6.5, fillColor=grey)
        lbl.textAnchor = "end"
        d.add(lbl)
    d.add(Line(pad_l, pad_b, pad_l, pad_b + plot_h, strokeColor=grey, strokeWidth=0.6))

    step = max(1, (max_n - 1) // 8)
    ticks = list(range(1, max_n + 1, step))
    if ticks[-1] != max_n:
        ticks.append(max_n)
    for n in ticks:
        lbl = String(x_of(n), pad_b - 10, str(n), fontSize=6.5, fillColor=grey)
        lbl.textAnchor = "middle"
        d.add(lbl)
    cap = String(pad_l + plot_w / 2, 2, "attempts per objective (N)",
                 fontSize=6.5, fillColor=grey)
    cap.textAnchor = "middle"
    d.add(cap)

    pts = []
    for pt in curve:
        x, y = x_of(pt["n"]), y_of(pt["rate"])
        pts.extend([x, y])
        d.add(Circle(x, y, 1.8, fillColor=colors.black, strokeColor=None))
    d.add(PolyLine(pts, strokeColor=colors.black, strokeWidth=1.2))
    return d


def reach_chart(series, width: int = 620, height: int = 220) -> dict | None:
    """Lay out a success-vs-attempt-budget chart as plain inline SVG geometry.

    `series` is [(label, curve), ...] where curve comes from models.reach_curve.
    Returns the pre-computed geometry the template needs, or None when there is
    nothing to plot (a single-attempt run has no curve, just a point).

    Deliberately hand-rolled rather than pulling in a charting library: the app
    ships as a self-contained offline .exe, so an inline SVG with no runtime
    dependency is the only thing guaranteed to render everywhere.
    """
    series = [(label, c) for label, c in series if c]
    max_n = max((c[-1]["n"] for _, c in series), default=1)
    if not series or max_n < 2:
        return None

    pad_l, pad_r, pad_t, pad_b = 44, 10, 12, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def x_of(n: int) -> float:
        return round(pad_l + (n - 1) / (max_n - 1) * plot_w, 2)

    def y_of(rate: float) -> float:
        return round(pad_t + (100.0 - rate) / 100.0 * plot_h, 2)

    lines = []
    for i, (label, curve) in enumerate(series):
        pts = [(x_of(pt["n"]), y_of(pt["rate"])) for pt in curve]
        lines.append({
            "label": label,
            "dash": CURVE_DASHES[i % len(CURVE_DASHES)],
            "points": " ".join(f"{x},{y}" for x, y in pts),
            "markers": [{"x": x, "y": y} for x, y in pts],
            "final": curve[-1]["rate"],
        })

    # Keep the x axis readable when the budget is large.
    step = max(1, (max_n - 1) // 8)
    x_ticks = [{"n": n, "x": x_of(n)} for n in range(1, max_n + 1, step)]
    if x_ticks[-1]["n"] != max_n:
        x_ticks.append({"n": max_n, "x": x_of(max_n)})

    return {
        "width": width, "height": height,
        "pad_l": pad_l, "pad_t": pad_t,
        "plot_w": plot_w, "plot_h": plot_h,
        "x_ticks": x_ticks,
        "y_ticks": [{"v": v, "y": y_of(v)} for v in (0, 25, 50, 75, 100)],
        "lines": lines,
        "max_n": max_n,
    }
