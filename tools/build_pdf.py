"""Render the technical report and the AI usage statement to PDF.

Word COM hangs on this machine and no LibreOffice/pandoc is installed, so the PDF is
produced from the SAME structured content that build_report.py and build_ai_statement.py
use for the DOCX, laid out as styled HTML and printed by Microsoft Edge (Chromium) through
Playwright's msedge channel. No browser is downloaded; the Edge already on the machine is
used. Page numbers are drawn in a clean footer template, so the two formats match and
neither carries a temp-file path.
"""
from __future__ import annotations

import base64
import html as _html
import json
from pathlib import Path

from build_report import (AI_USE_HEADING, AI_USE_TEXT, ART, EVID, FIG,
                          REFERENCES, REPO, ROOT, SECTIONS, _read, count_words)
from build_ai_statement import (DECIDED, HEADER, RESPONSIBLE, SUMMARY,
                                 TABLE_HEADERS, TABLE_ROWS, UNDERSTAND,
                                 evidence_items)

REP_PDF = ROOT / "reports" / "S00213739_AINT602_A2_Technical_Report.pdf"
STM_PDF = ROOT / "reports" / "S00213739_AINT602_A2_AI_Usage_Statement.pdf"

CSS = """
* { box-sizing: border-box; }
body { font-family: "Calibri","Segoe UI",Arial,sans-serif; font-size: 11pt; line-height: 1.4;
       color: #222; margin: 0; }
p { margin: 0 0 8px 0; text-align: justify; }
h1 { color: #1B3A6B; font-size: 16pt; font-weight: 700; margin: 18px 0 8px;
     padding-bottom: 4px; border-bottom: 2px solid #C9A227; break-after: avoid; }
h2 { color: #1B3A6B; font-size: 12.5pt; font-weight: 700; margin: 14px 0 4px; break-after: avoid; }
ul { margin: 0 0 8px 0; padding-left: 20px; }
li { margin: 0 0 4px 0; text-align: justify; }
table { width: 100%; border-collapse: collapse; margin: 6px 0; }
th { background: #1B3A6B; color: #fff; font-size: 9.5pt; font-weight: 700; padding: 5px 6px;
     text-align: left; border: 1px solid #1B3A6B; }
td { font-size: 9.5pt; padding: 4px 6px; border: 1px solid #bbb; vertical-align: top; }
.caption { font-style: italic; font-size: 9pt; color: #555; text-align: center; margin: 4px 0 10px; }
figure { margin: 10px 0; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; height: auto; }
pre { font-family: "Consolas","Courier New",monospace; font-size: 8.3pt; line-height: 1.25;
      background: #F5F5F5; border: 1px solid #ddd; border-radius: 3px; padding: 8px 10px;
      white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere; margin: 4px 0 10px; }
.title { text-align: center; color: #1B3A6B; font-size: 20pt; font-weight: 700; margin: 40px 0 6px; }
.subtitle { text-align: center; color: #555; font-size: 12pt; margin: 0 0 14px; }
.meta { text-align: center; font-size: 10.5pt; margin: 2px 0; }
.hdr { text-align: left; margin: 2px 0; }
.refs .ref { padding-left: 0.5in; text-indent: -0.5in; font-size: 10.5pt; text-align: left; margin: 0 0 6px; }
.page-break { break-before: page; }
"""


def rich(text: str) -> str:
    """Escape for HTML and honour a simple **bold** marker, mirroring docbuild._set_runs."""
    parts = str(text).split("**")
    out = []
    for i, part in enumerate(parts):
        esc = _html.escape(part)
        out.append(f"<strong>{esc}</strong>" if i % 2 == 1 else esc)
    return "".join(out)


def data_uri(path: Path) -> str:
    raw = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def figure_html(path: Path, caption: str, width_in: float) -> str:
    w_cm = width_in * 2.54
    return (f'<figure><img src="{data_uri(path)}" style="width:{w_cm:.1f}cm">'
            f'<figcaption class="caption">{rich(caption)}</figcaption></figure>')


def table_html(headers, rows, caption) -> str:
    th = "".join(f"<th>{rich(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{rich(c)}</td>" for c in row) + "</tr>" for row in rows)
    out = f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"
    if caption:
        out += f'<div class="caption" style="text-align:left">{rich(caption)}</div>'
    return out


def code_html(text: str) -> str:
    return f"<pre>{_html.escape(text)}</pre>"


def ul(items) -> str:
    return "<ul>" + "".join(f"<li>{rich(x)}</li>" for x in items) + "</ul>"


def page(title: str, body: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{_html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def report_html() -> str:
    words = count_words()
    b = []
    b.append('<div class="title">An MLOps Pipeline for 30-Day Hospital Readmission Prediction</div>')
    b.append('<div class="subtitle">AINT602 Assessment 2: Applied MLOps Pipeline Project</div>')
    for line in ["Sriramkumar Suri THIRUKKOLLURI (S00213739)",
                 f"Repository: {REPO}", f"Main body word count: {words}"]:
        b.append(f'<div class="meta">{rich(line)}</div>')
    b.append('<div class="page-break"></div>')

    for i, (title, blocks) in enumerate(SECTIONS, start=1):
        b.append(f"<h1>{i}. {rich(title)}</h1>")
        for kind, payload in blocks:
            if kind == "p":
                b.append(f"<p>{rich(payload)}</p>")
            elif kind == "b":
                b.append(ul(payload))
            elif kind == "fig":
                path, caption, width = payload
                b.append(figure_html(path, caption, width))
            elif kind == "table":
                headers, rows, caption = payload
                b.append(table_html(headers, rows, caption))

    b.append(f"<h1>{rich(AI_USE_HEADING)}</h1>")
    b.append(f"<p>{rich(AI_USE_TEXT)}</p>")
    b.append("<h1>References</h1>")
    b.append('<div class="refs">' + "".join(f'<p class="ref">{rich(r)}</p>' for r in REFERENCES) + "</div>")

    b.append('<div class="page-break"></div>')
    b.append("<h1>Evidence appendix</h1>")
    b.append(f"<p>{rich('Each item below is output produced by running the pipeline in this repository. The technical report refers to these items by number.')}</p>")

    def appx(num, title, note=None):
        s = f"<h2>Appendix {num}. {rich(title)}</h2>"
        if note:
            s += f"<p>{rich(note)}</p>"
        return s

    b.append(appx("A1", "Repository and commit history",
                  "The repository is public at the link on the title page. The commit history below reflects how the pipeline was built."))
    b.append(code_html(_read(EVID / "git_log.txt")))
    b.append(appx("A2", "Data validation output (strict mode, full dataset)"))
    b.append(code_html(_read(EVID / "validation.txt")))
    b.append(appx("A3", "Automated tests and lint"))
    b.append(code_html(_read(EVID / "ruff.txt") + "\n\n" + _read(EVID / "pytest.txt")))
    b.append(appx("A4", "GitHub Actions continuous integration run"))
    b.append(code_html(_read(EVID / "ci_run.txt", max_lines=10)))
    b.append(appx("A5", "Run metadata and provenance (artifacts/run_metadata.json)"))
    b.append(code_html(_read(ART / "run_metadata.json")))
    b.append(appx("A6", "MLflow experiment-tracking store (mlruns/)",
                  "MLflow records each run on disk under mlruns/. The structure below lists one experiment with its runs, parameters, metrics and logged artefacts."))
    b.append(code_html(_read(EVID / "mlruns_tree.txt", max_lines=26)))
    b.append(appx("A7", "Sample prediction on new data"))
    b.append(code_html(_read(EVID / "predict.txt")))
    b.append(appx("A8", "Benchmark metrics (artifacts/metrics.json, selected excerpt)"))
    metrics = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
    excerpt = {
        "selected_model": metrics["selected_model"],
        "threshold": round(metrics["threshold"], 4),
        "validation": metrics["validation"],
        "test_logistic_regression": {k: metrics["test"]["logistic_regression"][k]
                                     for k in ("roc_auc", "pr_auc", "recall", "precision", "f1")},
        "test_baseline": {k: metrics["test"]["baseline"][k] for k in ("accuracy", "pr_auc", "recall")},
    }
    b.append(code_html(json.dumps(excerpt, indent=2)))
    b.append(appx("A9", "Evaluation figures (test set)"))
    for fn, cap, w in [
        ("roc_curve.png", "Figure 2. ROC curve.", 3.4),
        ("pr_curve.png", "Figure 3. Precision-recall curve against the base rate.", 3.4),
        ("confusion_matrix.png", "Figure 4. Confusion matrix at the operating threshold.", 3.2),
        ("calibration.png", "Figure 5. Calibration of predicted probabilities.", 3.4),
        ("fairness_by_sex.png", "Figure 6. Recall and flagged rate by sex, showing the small Other group is not served.", 3.8),
    ]:
        b.append(figure_html(FIG / fn, cap, w))

    return page("Technical Report", "\n".join(b))


def statement_html() -> str:
    wc = count_words()
    b = []
    b.append('<div class="title" style="margin-top:0">AI Usage Statement</div>')
    for label, value in HEADER:
        b.append(f'<p class="hdr"><strong>{rich(label)}:</strong> {rich(value)}</p>')
    b.append("<h2>1. Summary</h2>")
    b.append(f"<p>{rich(SUMMARY)}</p>")
    b.append("<h2>2. What I decided and did myself</h2>")
    b.append(ul(DECIDED))
    b.append("<h2>3. What the AI assisted with</h2>")
    b.append('<div class="caption" style="text-align:left">Table 1. Tasks for which AI assistance was used.</div>')
    b.append(table_html(TABLE_HEADERS, TABLE_ROWS, None))
    b.append("<h2>4. Supporting evidence</h2>")
    b.append(ul(evidence_items(wc)))
    b.append("<h2>5. How I made sure I understand the work</h2>")
    b.append(ul(UNDERSTAND))
    b.append("<h2>6. Responsible and ethical use</h2>")
    b.append(ul(RESPONSIBLE))
    return page("AI Usage Statement", "\n".join(b))


def render_pdf(html_text: str, out_pdf: Path, footer_label: str) -> None:
    from playwright.sync_api import sync_playwright
    footer = (f'<div style="width:100%;font-size:8px;color:#555;padding:0 1.2cm;text-align:center;">'
              f'{_html.escape(footer_label)} &nbsp;&nbsp; Page '
              f'<span class="pageNumber"></span> of <span class="totalPages"></span></div>')
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        pg = browser.new_page()
        pg.set_content(html_text, wait_until="load")
        pg.pdf(
            path=str(out_pdf),
            format="A4",
            print_background=True,
            margin={"top": "2cm", "bottom": "1.8cm", "left": "2cm", "right": "2cm"},
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=footer,
        )
        browser.close()


if __name__ == "__main__":
    render_pdf(report_html(), REP_PDF, "AINT602 Assessment 2 | S00213739")
    render_pdf(statement_html(), STM_PDF, "AI Usage Statement | S00213739")
    print(f"report pdf    -> {REP_PDF}  ({REP_PDF.stat().st_size} bytes)")
    print(f"statement pdf -> {STM_PDF}  ({STM_PDF.stat().st_size} bytes)")
