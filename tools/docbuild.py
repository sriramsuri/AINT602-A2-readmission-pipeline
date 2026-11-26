"""Shared helpers for building the assessment documents with python-docx.

This is document-generation tooling, separate from the ML pipeline in src/. It styles
a Word document (Calibri body, navy headings, shaded tables, captioned figures and
footer page numbers) so the report and the AI statement share one consistent look.
"""
from __future__ import annotations

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

NAVY = RGBColor(0x1B, 0x3A, 0x6B)
GREY = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def new_document():
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    pf = normal.paragraph_format
    pf.line_spacing = 1.15
    pf.space_after = Pt(6)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
    return doc


def _set_runs(paragraph, text):
    """Add text to a paragraph, honouring a simple **bold** marker."""
    for i, part in enumerate(text.split("**")):
        if not part:
            continue
        run = paragraph.add_run(part)
        if i % 2 == 1:
            run.bold = True


def body(doc, text, justify=True):
    p = doc.add_paragraph()
    _set_runs(p, text)
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def heading1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(15)
    run.font.color.rgb = NAVY
    return p


def heading2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(12.5)
    run.font.color.rgb = NAVY
    return p


def bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        _set_runs(p, item)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def _shade(cell, hexfill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexfill)
    tc_pr.append(shd)


def table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    header_cells = t.rows[0].cells
    for i, text in enumerate(headers):
        _shade(header_cells[i], "1B3A6B")
        run = header_cells[i].paragraphs[0].add_run(text)
        run.bold = True
        run.font.color.rgb = WHITE
        run.font.size = Pt(10)
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            run = cells[i].paragraphs[0].add_run(str(value))
            run.font.size = Pt(10)
    return t


def figure(doc, image_path, caption, width_inches=6.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(image_path), width=Inches(width_inches))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = GREY


def code_block(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text.rstrip())
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    return p


def page_break(doc):
    doc.add_page_break()


def _add_page_number(paragraph):
    run = paragraph.add_run()
    run.font.size = Pt(8)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(end)


def footer_pagenum(doc, left_text=""):
    footer = doc.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if left_text:
        run = p.add_run(left_text + "     ")
        run.font.size = Pt(8)
        run.font.color.rgb = GREY
    run = p.add_run("Page ")
    run.font.size = Pt(8)
    run.font.color.rgb = GREY
    _add_page_number(p)


def title_block(doc, title, subtitle, meta_lines):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(title)
    run.bold = True
    run.font.size = Pt(20)
    run.font.color.rgb = NAVY
    if subtitle:
        s = doc.add_paragraph()
        s.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = s.add_run(subtitle)
        run.font.size = Pt(12)
        run.font.color.rgb = GREY
    for line in meta_lines:
        m = doc.add_paragraph()
        m.alignment = WD_ALIGN_PARAGRAPH.CENTER
        m.paragraph_format.space_after = Pt(2)
        run = m.add_run(line)
        run.font.size = Pt(10.5)
    doc.add_paragraph()
