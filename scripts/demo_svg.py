"""Renders a real runmedic report into a terminal-style SVG for the README.

python scripts/demo_svg.py examples/logs/nan assets/demo.svg
"""

import html
import re
import subprocess
import sys

BG, FG, DIM = "#0d1117", "#c9d1d9", "#8b949e"
COLORS = {"CRITICAL": "#ff7b72", "WARNING": "#d29922", "INFO": "#58a6ff"}
LINE_H, CHAR_W, PAD = 20, 8.4, 20


def color_for(line: str) -> tuple[str, bool]:
    m = re.search(r"\[(CRITICAL|WARNING|INFO)\]", line)
    if m:
        return COLORS[m.group(1)], True
    if line.startswith(("Found:", "$ ")):
        return FG, True
    if line.strip().startswith("·"):
        return FG, False
    if line.strip().startswith(("Likely causes", "What to try")):
        return "#7ee787", True
    return DIM, False


def main(path: str, out: str, max_lines: int = 22) -> None:
    report = subprocess.run(["runmedic", "diagnose", path], capture_output=True, text=True, encoding="utf-8").stdout
    lines = report.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Found:"))
    lines = [f"$ runmedic diagnose {path.replace(chr(92), '/')}", ""] + lines[start : start + max_lines]
    while lines and not lines[-1].strip():
        lines.pop()

    width = int(max(len(ln) for ln in lines) * CHAR_W + 2 * PAD)
    height = len(lines) * LINE_H + 2 * PAD + 28
    rows = []
    for i, ln in enumerate(lines):
        color, bold = color_for(ln)
        y = PAD + 28 + (i + 1) * LINE_H - 6
        weight = ' font-weight="600"' if bold else ""
        rows.append(f'<text x="{PAD}" y="{y}" fill="{color}"{weight} xml:space="preserve">{html.escape(ln)}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" rx="10" fill="{BG}"/>
<circle cx="22" cy="20" r="6" fill="#ff5f57"/><circle cx="42" cy="20" r="6" fill="#febc2e"/><circle cx="62" cy="20" r="6" fill="#28c840"/>
<g font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="14">
{chr(10).join(rows)}
</g>
</svg>
"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(svg)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
