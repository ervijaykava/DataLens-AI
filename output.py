"""
output.py
----------
Everything that turns computed numbers into a visual or downloadable
output:

    line_chart / forecast_chart / bar_chart / donut_chart / scatter_chart
        hand-built SVG strings (no matplotlib, no frontend JS chart
        library). Templates drop the returned string directly into the
        page with Jinja2's `| safe` filter. Every function returns an
        <svg>...</svg> string, or an empty string if there isn't enough
        data to draw anything - the template then shows an "empty state"
        message instead.

    build_excel_report / build_pdf_report
        the downloadable business report. Excel export uses pandas +
        openpyxl. PDF export uses a minimal reportlab layout.
"""

import io

import pandas as pd

PRIMARY = "#4f46e5"     # indigo
SECONDARY = "#0ea5e9"   # sky blue
POSITIVE = "#16a34a"    # green
NEGATIVE = "#dc2626"    # red
GRID = "#e5e7eb"
TEXT = "#6b7280"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def line_chart(points: list, width: int = 640, height: int = 260, value_key: str = "value", label_key: str = "label") -> str:
    """A simple line + area chart for trends, e.g. monthly revenue."""
    if not points or len(points) < 2:
        return ""

    pad_left, pad_right, pad_top, pad_bottom = 50, 20, 20, 30
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    values = [p[value_key] for p in points]
    min_v, max_v = min(values), max(values)
    if min_v == max_v:
        min_v -= 1
        max_v += 1
    span = max_v - min_v

    def x_at(i):
        return pad_left + (i / (len(points) - 1)) * plot_w

    def y_at(v):
        return pad_top + plot_h - ((v - min_v) / span) * plot_h

    coords = [(x_at(i), y_at(p[value_key])) for i, p in enumerate(points)]
    line_path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coords))
    area_path = line_path + f" L{coords[-1][0]:.1f},{pad_top + plot_h:.1f} L{coords[0][0]:.1f},{pad_top + plot_h:.1f} Z"

    grid_lines = ""
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = pad_top + plot_h - frac * plot_h
        grid_lines += f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1" />'

    label_step = max(1, len(points) // 6)
    labels = ""
    for i, p in enumerate(points):
        if i % label_step == 0 or i == len(points) - 1:
            labels += f'<text x="{x_at(i):.1f}" y="{height - 8}" font-size="10" fill="{TEXT}" text-anchor="middle">{p[label_key]}</text>'

    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{PRIMARY}" />' for x, y in coords)

    return f'''<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" class="chart-svg">
        {grid_lines}
        <path d="{area_path}" fill="{PRIMARY}" fill-opacity="0.08" stroke="none" />
        <path d="{line_path}" fill="none" stroke="{PRIMARY}" stroke-width="2.5" />
        {dots}
        {labels}
    </svg>'''


def forecast_chart(history: list, forecast: list, width: int = 640, height: int = 280) -> str:
    """History (solid line) followed by forecast (dashed line + shaded band)."""
    if not history or not forecast:
        return ""

    pad_left, pad_right, pad_top, pad_bottom = 55, 20, 20, 30
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    all_values = [p["value"] for p in history] + [p["value"] for p in forecast] + \
                 [p.get("lower", p["value"]) for p in forecast] + [p.get("upper", p["value"]) for p in forecast]
    min_v, max_v = min(all_values), max(all_values)
    if min_v == max_v:
        min_v -= 1
        max_v += 1
    span = max_v - min_v
    total_points = len(history) + len(forecast)

    def x_at(i):
        return pad_left + (i / (total_points - 1)) * plot_w

    def y_at(v):
        return pad_top + plot_h - ((v - min_v) / span) * plot_h

    hist_coords = [(x_at(i), y_at(p["value"])) for i, p in enumerate(history)]
    fc_start = len(history) - 1
    fc_coords = [hist_coords[-1]] + [(x_at(fc_start + i + 1), y_at(p["value"])) for i, p in enumerate(forecast)]
    upper_coords = [hist_coords[-1]] + [(x_at(fc_start + i + 1), y_at(p.get("upper", p["value"]))) for i, p in enumerate(forecast)]
    lower_coords = [hist_coords[-1]] + [(x_at(fc_start + i + 1), y_at(p.get("lower", p["value"]))) for i, p in enumerate(forecast)]

    hist_path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(hist_coords))
    fc_path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(fc_coords))

    band_path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(upper_coords))
    band_path += " " + " ".join(f"L{x:.1f},{y:.1f}" for x, y in reversed(lower_coords)) + " Z"

    grid_lines = ""
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = pad_top + plot_h - frac * plot_h
        grid_lines += f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1" />'

    divider_x = x_at(fc_start)

    return f'''<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" class="chart-svg">
        {grid_lines}
        <line x1="{divider_x:.1f}" y1="{pad_top}" x2="{divider_x:.1f}" y2="{pad_top + plot_h}" stroke="{GRID}" stroke-width="1" stroke-dasharray="4 4" />
        <path d="{band_path}" fill="{SECONDARY}" fill-opacity="0.12" stroke="none" />
        <path d="{hist_path}" fill="none" stroke="{PRIMARY}" stroke-width="2.5" />
        <path d="{fc_path}" fill="none" stroke="{SECONDARY}" stroke-width="2.5" stroke-dasharray="6 4" />
    </svg>'''


def _truncate(text: str, max_chars: int) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "\u2026"


def bar_chart(items: list, width: int = 560, height: int = 260, horizontal: bool = True) -> str:
    """A simple horizontal bar chart, e.g. top products / top regions."""
    if not items:
        return ""
    items = items[:8]
    max_v = max(i["value"] for i in items) or 1

    pad_left, pad_right, pad_top = 168, 66, 10
    row_h = 32
    plot_w = width - pad_left - pad_right
    height = pad_top * 2 + row_h * len(items)

    bars = ""
    for i, item in enumerate(items):
        y = pad_top + i * row_h
        bar_w = (item["value"] / max_v) * plot_w
        label = _truncate(item["name"], 20)
        bars += f'<text x="{pad_left - 12}" y="{y + row_h * 0.6:.1f}" font-size="12" fill="#111827" text-anchor="end">{label}</text>'
        bars += f'<rect x="{pad_left}" y="{y + 6}" width="{bar_w:.1f}" height="{row_h - 14}" rx="4" fill="{PRIMARY}" />'
        bars += f'<text x="{pad_left + bar_w + 8:.1f}" y="{y + row_h * 0.6:.1f}" font-size="12" fill="{TEXT}">{item["value"]:,.0f}</text>'

    return f'''<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" class="chart-svg">
        {bars}
    </svg>'''


def donut_chart(items: list, width: int = 260, height: int = 260) -> str:
    """A simple donut chart, e.g. customer segment distribution."""
    if not items:
        return ""
    items = items[:6]
    total = sum(i["value"] for i in items) or 1
    colors = ["#4f46e5", "#0ea5e9", "#16a34a", "#f59e0b", "#dc2626", "#8b5cf6"]

    cx, cy, r, r_inner = width / 2, height / 2, 100, 60
    start_angle = -90
    segments = ""
    legend = ""
    import math

    for i, item in enumerate(items):
        fraction = item["value"] / total
        angle = fraction * 360
        end_angle = start_angle + angle

        def point(angle_deg, radius):
            rad = math.radians(angle_deg)
            return cx + radius * math.cos(rad), cy + radius * math.sin(rad)

        x1, y1 = point(start_angle, r)
        x2, y2 = point(end_angle, r)
        x3, y3 = point(end_angle, r_inner)
        x4, y4 = point(start_angle, r_inner)
        large_arc = 1 if angle > 180 else 0
        color = colors[i % len(colors)]

        segments += (
            f'<path d="M{x1:.1f},{y1:.1f} A{r},{r} 0 {large_arc} 1 {x2:.1f},{y2:.1f} '
            f'L{x3:.1f},{y3:.1f} A{r_inner},{r_inner} 0 {large_arc} 0 {x4:.1f},{y4:.1f} Z" '
            f'fill="{color}" />'
        )
        legend += (
            f'<circle cx="16" cy="{20 + i * 20}" r="5" fill="{color}" />'
            f'<text x="28" y="{24 + i * 20}" font-size="11" fill="#374151">{_truncate(item["name"], 18)} ({item["value"]})</text>'
        )
        start_angle = end_angle

    return f'''<svg viewBox="0 0 {width + 170} {max(height, len(items) * 20 + 20)}" xmlns="http://www.w3.org/2000/svg" class="chart-svg">
        {segments}
        <g transform="translate({width - 10}, 0)">{legend}</g>
    </svg>'''


def scatter_chart(points: list, width: int = 560, height: int = 300) -> str:
    """Scatter plot used for anomaly visualization: normal vs flagged points."""
    if not points:
        return ""
    pad = 40
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    min_x, max_x = min(xs), max(xs) or 1
    min_y, max_y = min(ys), max(ys) or 1
    if min_x == max_x:
        max_x += 1
    if min_y == max_y:
        max_y += 1

    def x_at(v):
        return pad + (v - min_x) / (max_x - min_x) * (width - 2 * pad)

    def y_at(v):
        return height - pad - (v - min_y) / (max_y - min_y) * (height - 2 * pad)

    dots = ""
    for p in points:
        color = NEGATIVE if p.get("is_anomaly") else "#93c5fd"
        r = 4 if p.get("is_anomaly") else 3
        dots += f'<circle cx="{x_at(p["x"]):.1f}" cy="{y_at(p["y"]):.1f}" r="{r}" fill="{color}" fill-opacity="0.8" />'

    return f'''<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" class="chart-svg">
        <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="{GRID}" />
        <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="{GRID}" />
        {dots}
    </svg>'''


# ---------------------------------------------------------------------------
# Downloadable reports
# ---------------------------------------------------------------------------
def build_excel_report(kpis: dict, insights: list, top_products: list, top_regions: list, recommendations: list) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([kpis]).to_excel(writer, sheet_name="KPIs", index=False)
        pd.DataFrame({"Insight": insights}).to_excel(writer, sheet_name="Insights", index=False)
        if top_products:
            pd.DataFrame(top_products).to_excel(writer, sheet_name="Top Products", index=False)
        if top_regions:
            pd.DataFrame(top_regions).to_excel(writer, sheet_name="Top Regions", index=False)
        if recommendations:
            pd.DataFrame(recommendations).to_excel(writer, sheet_name="Recommendations", index=False)
    output.seek(0)
    return output.read()


def build_pdf_report(kpis: dict, insights: list, top_products: list, top_regions: list, recommendations: list) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()

    story = [
        Paragraph("Business Intelligence Copilot - Executive Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph("Key Performance Indicators", styles["Heading2"]),
        Table(
            [["Metric", "Value"]] + [[k.replace("_", " ").title(), str(v)] for k, v in kpis.items()],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]),
        ),
        Spacer(1, 16),
        Paragraph("Key Insights", styles["Heading2"]),
    ]
    for i in insights:
        story.append(Paragraph(f"- {i}", styles["Normal"]))

    if top_products:
        story.append(Spacer(1, 16))
        story.append(Paragraph("Top Products", styles["Heading2"]))
        for p in top_products[:5]:
            story.append(Paragraph(f"- {p['name']}: ${p['value']:,.2f}", styles["Normal"]))

    if top_regions:
        story.append(Spacer(1, 16))
        story.append(Paragraph("Top Regions", styles["Heading2"]))
        for r in top_regions[:5]:
            story.append(Paragraph(f"- {r['name']}: ${r['value']:,.2f}", styles["Normal"]))

    if recommendations:
        story.append(Spacer(1, 16))
        story.append(Paragraph("Recommendations", styles["Heading2"]))
        for r in recommendations:
            story.append(Paragraph(f"- {r['title']} ({r['reason']})", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
