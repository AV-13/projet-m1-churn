"""Génère un PDF propre du rapport à partir du Markdown.

Chaîne : Markdown -> HTML stylé (images embarquées en base64) -> PDF (Chromium).
Aucune dépendance LaTeX. Usage : python scripts/build_report_pdf.py
"""
from __future__ import annotations

import base64
import mimetypes
import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "03-rapport.md"
HTML = ROOT / "docs" / "03-rapport.html"
PDF = ROOT / "docs" / "03-rapport.pdf"

CSS = """
@page { size: A4; margin: 17mm 18mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
       color: #1A2233; font-size: 10.5pt; line-height: 1.5; margin: 0; }
h1 { font-size: 21pt; letter-spacing: -.01em; margin: 0 0 .15em; }
h2 { font-size: 13pt; margin: 1.3em 0 .5em; padding-bottom: .22em;
     border-bottom: 2px solid #4F46E5; page-break-after: avoid; }
h3 { font-size: 11pt; color: #4F46E5; margin: 1em 0 .25em; page-break-after: avoid; }
p { margin: .5em 0; }
strong { color: #1A2233; }
.meta { color: #6B7385; font-size: 9.5pt; margin: .4em 0 0; line-height: 1.7; }
hr { border: none; border-top: 1px solid #E6E9EF; margin: 1.1em 0; }
table { border-collapse: collapse; width: 100%; margin: .8em 0; font-size: 9.5pt;
        page-break-inside: avoid; }
th { background: #4F46E5; color: #fff; text-align: left; padding: 6px 10px; font-weight: 600; }
td { border-bottom: 1px solid #E6E9EF; padding: 6px 10px; }
tr:nth-child(even) td { background: #F7F8FA; }
code { background: #F0F2F5; padding: 1px 5px; border-radius: 4px; font-size: 9pt; }
figure { margin: .6em 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; border: 1px solid #E6E9EF; border-radius: 8px; }
figcaption { font-size: 8.3pt; color: #6B7385; margin-top: .35em; font-style: italic; }
.figrow { display: flex; gap: 14px; align-items: flex-start; }
.figrow figure { flex: 1; min-width: 0; }
ul, ol { margin: .4em 0; padding-left: 1.3em; }
li { margin: .15em 0; }
"""


def embed_images(html: str, base_dir: Path) -> str:
    """Remplace les src d'images par des data URIs base64 (PDF autonome)."""
    def repl(m: re.Match) -> str:
        src = m.group(1)
        path = (base_dir / src).resolve()
        if not path.exists():
            return m.group(0)
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode()
        return f'src="data:{mime};base64,{data}"'

    return re.sub(r'src="([^"]+)"', repl, html)


def main() -> int:
    body = markdown.markdown(
        MD.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "attr_list", "md_in_html"],
    )
    body = embed_images(body, MD.parent)
    page = (f"<!doctype html><html lang='fr'><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    HTML.write_text(page, encoding="utf-8")

    chromium = next((c for c in ("/usr/bin/chromium-browser", "/snap/bin/chromium",
                                 "/usr/bin/chromium") if Path(c).exists()), None)
    if not chromium:
        print("Chromium introuvable — HTML généré :", HTML)
        return 1
    subprocess.run(
        [chromium, "--headless=new", "--no-sandbox", "--disable-gpu",
         "--no-pdf-header-footer", f"--print-to-pdf={PDF}", HTML.as_uri()],
        check=True, capture_output=True,
    )
    print("PDF généré :", PDF)
    return 0


if __name__ == "__main__":
    sys.exit(main())
