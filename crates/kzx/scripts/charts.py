#!/usr/bin/env python3
"""Render the README charts from benchmark CSVs written by `examples/bench`.

    python3 scripts/charts.py assets/bench/private.csv assets/private \
        --title "Metrale GB10 kernel set" --note "GB10 Grace CPU (Cortex-X925), single thread, ..."

Writes `<out>-ratio.svg` and `<out>-speed.svg`. Pure standard library; every number comes from
the CSV (no hand-entered values). The SVGs use a transparent background and mid-tone text so
they read on GitHub's light and dark themes; kzx is the one accent colour, baselines are muted.
"""
import argparse
import csv
import math
from xml.sax.saxutils import escape

ACCENT = "#e4572e"
MUTED = ["#8da0b3", "#a3b1c2", "#b9c3cf", "#7f93a6", "#97a8b9", "#aebbc8", "#6f8599"]
TEXT = "#8b949e"
GRID = "#8b949e"
FONT = "font-family='ui-sans-serif,system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"
MODE_PREFERENCE = ["solid", "sections", "per-file"]
CLASS_ORDER = ["ptx", "cubin", "body", "kbp"]
CLASS_LABEL = {"ptx": "PTX", "cubin": "cubin (SASS ELF)", "body": "postcard body", "kbp": "full .kbp"}


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("files", "raw_bytes", "packed_bytes"):
            r[k] = int(r[k])
        for k in ("ratio", "c_secs", "d_secs", "c_mbps", "d_mbps"):
            r[k] = float(r[k])
    return rows


def select(rows):
    """One mode per class (the preference order), rows grouped by class then tool."""
    out = {}
    for cls in CLASS_ORDER:
        for mode in MODE_PREFERENCE:
            sub = [r for r in rows if r["class"] == cls and r["mode"] == mode]
            if sub:
                out[cls] = (mode, sub)
                break
    return out


def tools_in(selected):
    seen = ["kzx"]
    for _, (_, sub) in selected.items():
        for r in sub:
            if r["tool"] not in seen:
                seen.append(r["tool"])
    return seen


def colour(tool, tools):
    if tool == "kzx":
        return ACCENT
    return MUTED[(tools.index(tool) - 1) % len(MUTED)]


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def svg_header(w, h):
    return [f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' {FONT}>"]


def text(x, y, s, size=12, anchor="start", weight="normal", fill=TEXT, rotate=None):
    t = f" transform='rotate({rotate} {x} {y})'" if rotate else ""
    return f"<text x='{x}' y='{y}' font-size='{size}' text-anchor='{anchor}' font-weight='{weight}' fill='{fill}'{t}>{escape(s)}</text>"


def legend(parts, tools, x, y):
    cx = x
    for t in tools:
        parts.append(f"<rect x='{cx}' y='{y - 9}' width='12' height='12' rx='2' fill='{colour(t, tools)}'/>")
        parts.append(text(cx + 16, y + 1, t, 11))
        cx += 16 + 7 * len(t) + 22


def footnote(parts, note, x, y, width):
    words = note.split()
    line, lines = "", []
    for w in words:
        if len(line) + len(w) + 1 > width // 6:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    for i, l in enumerate(lines):
        parts.append(text(x, y + i * 14, l, 10))
    return len(lines)


def ratio_chart(selected, title, note, out):
    tools = tools_in(selected)
    classes = list(selected)
    w, left, right, top = 900, 70, 20, 60
    group_w = (w - left - right) / max(1, len(classes))
    bar_w = min(26, (group_w - 24) / max(1, len(tools)))
    plot_h = 300
    ymax = max(r["ratio"] for _, (_, sub) in selected.items() for r in sub) * 1.15
    h = top + plot_h + 160
    parts = svg_header(w, h)
    parts.append(text(left, 24, title, 16, weight="bold"))
    parts.append(text(left, 42, "Compression ratio (raw bytes / compressed bytes), higher is better", 12))
    step = 10 ** math.floor(math.log10(ymax / 5)) if ymax > 0 else 1
    step = step * (5 if ymax / step > 25 else 2 if ymax / step > 12 else 1)
    y = 0
    while y <= ymax:
        py = top + plot_h - y / ymax * plot_h
        parts.append(f"<line x1='{left}' y1='{py:.1f}' x2='{w - right}' y2='{py:.1f}' stroke='{GRID}' stroke-opacity='0.25'/>")
        parts.append(text(left - 8, py + 4, f"{y:g}x", 11, anchor="end"))
        y += step
    for gi, cls in enumerate(classes):
        mode, sub = selected[cls]
        gx = left + gi * group_w + 12
        by_tool = {r["tool"]: r for r in sub}
        for ti, t in enumerate(tools):
            r = by_tool.get(t)
            if not r:
                continue
            bh = r["ratio"] / ymax * plot_h
            x = gx + ti * bar_w
            parts.append(f"<rect x='{x:.1f}' y='{top + plot_h - bh:.1f}' width='{bar_w - 3:.1f}' height='{bh:.1f}' rx='2' fill='{colour(t, tools)}'/>")
            parts.append(text(x + (bar_w - 3) / 2, top + plot_h - bh - 4, f"{r['ratio']:.1f}", 9, anchor="middle", weight="bold" if t == "kzx" else "normal", fill=ACCENT if t == "kzx" else TEXT))
        raw = sub[0]["raw_bytes"]
        parts.append(text(gx + len(tools) * bar_w / 2, top + plot_h + 18, CLASS_LABEL.get(cls, cls), 12, anchor="middle", weight="bold"))
        parts.append(text(gx + len(tools) * bar_w / 2, top + plot_h + 33, f"{mode}, {sub[0]['files']} file(s), {human(raw)}", 10, anchor="middle"))
    parts.append(f"<line x1='{left}' y1='{top + plot_h}' x2='{w - right}' y2='{top + plot_h}' stroke='{GRID}' stroke-opacity='0.6'/>")
    legend(parts, tools, left, top + plot_h + 58)
    footnote(parts, note, left, top + plot_h + 84, w - left - right)
    parts.append("</svg>")
    with open(out, "w") as f:
        f.write("\n".join(parts))


def speed_panel(parts, selected, tools, key, label, x0, y0, pw, ph):
    classes = list(selected)
    vals = [r[key] for _, (_, sub) in selected.items() for r in sub if r[key] > 0]
    lo = 10 ** math.floor(math.log10(min(vals))) if vals else 0.1
    hi = 10 ** math.ceil(math.log10(max(vals) * 1.3)) if vals else 1000
    span = math.log10(hi) - math.log10(lo)

    def py(v):
        return y0 + ph - (math.log10(max(v, lo)) - math.log10(lo)) / span * ph

    parts.append(text(x0, y0 - 10, label, 12, weight="bold"))
    d = lo
    while d <= hi:
        parts.append(f"<line x1='{x0}' y1='{py(d):.1f}' x2='{x0 + pw}' y2='{py(d):.1f}' stroke='{GRID}' stroke-opacity='0.25'/>")
        parts.append(text(x0 - 6, py(d) + 4, f"{d:g}", 10, anchor="end"))
        d *= 10
    group_w = pw / max(1, len(classes))
    bar_w = min(22, (group_w - 20) / max(1, len(tools)))
    for gi, cls in enumerate(classes):
        _, sub = selected[cls]
        by_tool = {r["tool"]: r for r in sub}
        gx = x0 + gi * group_w + 10
        for ti, t in enumerate(tools):
            r = by_tool.get(t)
            if not r or r[key] <= 0:
                continue
            x = gx + ti * bar_w
            top_y = py(r[key])
            parts.append(f"<rect x='{x:.1f}' y='{top_y:.1f}' width='{bar_w - 3:.1f}' height='{y0 + ph - top_y:.1f}' rx='2' fill='{colour(t, tools)}'/>")
            if t == "kzx":
                parts.append(text(x + (bar_w - 3) / 2, top_y - 3, f"{r[key]:.1f}", 9, anchor="middle", weight="bold", fill=ACCENT))
        parts.append(text(gx + len(tools) * bar_w / 2, y0 + ph + 16, CLASS_LABEL.get(cls, cls), 11, anchor="middle", weight="bold"))
    parts.append(f"<line x1='{x0}' y1='{y0 + ph}' x2='{x0 + pw}' y2='{y0 + ph}' stroke='{GRID}' stroke-opacity='0.6'/>")


def speed_chart(selected, title, note, out):
    tools = tools_in(selected)
    w, left, top = 900, 60, 60
    ph = 220
    h = top + ph + 150
    parts = svg_header(w, h)
    parts.append(text(left, 24, title, 16, weight="bold"))
    parts.append(text(left, 42, "Throughput in MB/s (raw bytes per second, log scale), single thread, higher is better", 12))
    pw = (w - left - 30 - 50) / 2
    speed_panel(parts, selected, tools, "c_mbps", "compress", left, top + 20, pw, ph)
    speed_panel(parts, selected, tools, "d_mbps", "decompress", left + pw + 50, top + 20, pw, ph)
    legend(parts, tools, left, top + ph + 56)
    footnote(parts, note, left, top + ph + 82, w - left - 30)
    parts.append("</svg>")
    with open(out, "w") as f:
        f.write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("out_prefix")
    ap.add_argument("--title", default="kzx benchmark")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    rows = load(a.csv)
    selected = select(rows)
    if not selected:
        raise SystemExit("no rows for the known classes")
    ratio_chart(selected, a.title, a.note, f"{a.out_prefix}-ratio.svg")
    speed_chart(selected, a.title, a.note, f"{a.out_prefix}-speed.svg")
    print(f"wrote {a.out_prefix}-ratio.svg and {a.out_prefix}-speed.svg")


if __name__ == "__main__":
    main()
