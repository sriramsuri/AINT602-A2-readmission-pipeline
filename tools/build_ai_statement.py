"""Build the AI Usage Statement (DOCX), mirroring the structure of the A1 statement.

The prose lives in module-level constants so the PDF renderer (build_pdf.py) can import
the same text and the two formats never disagree. The word count referenced in section 4
is read from the report builder so the statement and the report never disagree either.
"""
from __future__ import annotations

from pathlib import Path

import docbuild as db
from build_report import count_words

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "S00213739_AINT602_A2_AI_Usage_Statement.docx"

HEADER = [
    ("Student", "Sriramkumar Suri THIRUKKOLLURI (S00213739)"),
    ("Assessment", "AINT602 Assessment 2, Applied MLOps Pipeline Project"),
    ("AI tool used", "Claude (Anthropic), used through the Claude Code command-line tool"),
    ("Declared use", "Claude designed and wrote the pipeline, ran it, produced the figures, and drafted the technical report and this statement. The tasks are listed in section 3."),
]

TABLE_HEADERS = ["Task", "What I asked for", "What I did with the output"]
TABLE_ROWS = [
    ["Interpreting the task",
     "A plain reframing of the brief, the dataset and the rubric for me to approve before any work began.",
     "Claude profiled the dataset, read the rubric, recommended the readmission task with its reasons, and listed the design decisions. I approved them and chose a public GitHub repository."],
    ["Building the pipeline",
     "A working, reproducible MLOps pipeline covering the six project areas.",
     "Claude wrote the code, ran it end to end, and fixed the problems that came up, including changes in pandas 3 and MLflow 3. I read the result and the tests."],
    ["Writing the report and this statement",
     "The technical report and this statement, within the limits and my writing rules.",
     "Claude drafted both from the real run outputs. I read them, and any changes are my own."],
]

SUMMARY = (
    "I used AI in preparing this assessment. I set the task and the constraints, chose the "
    "dataset and the prediction problem, and approved the plan before any building began. Claude "
    "then wrote the pipeline code, ran it on my machine, produced the metrics, figures and run "
    "records, and drafted the technical report and this statement. Where the report says I, the "
    "decisions about what to build, which outcome to predict and how to present it are mine. My "
    "part was choosing the scenario and the data, giving the instructions, setting the writing "
    "rules, approving the design before work started, and then reading and checking the result. "
    "The sections below record what I decided, what the AI did, the evidence for that account, "
    "and how I made sure I understand what I am submitting."
)

DECIDED = [
    "I set the task. I gave Claude the assessment brief and the marking rubric.",
    "I chose the dataset and the prediction task. I selected Dataset B, the synthetic clinical dataset, and chose 30-day readmission as the target instead of the mortality or diagnosis options.",
    "I asked to approve Claude's reading of the task first. Claude profiled the data, read the rubric, recommended the readmission task with its reasons, and listed the decisions. I approved them.",
    "I chose how it would be submitted. I asked for a public GitHub repository and for the report in both Word and PDF.",
    "I set the writing rules: a human voice with no em dashes, original and plagiarism-free writing, APA 7 references, and the aim of meeting the rubric in full.",
]


def evidence_items(wc):
    return [
        "Limits set before writing began: no em dashes, the six prescribed headings, APA 7 references, and a target of 2,500 to 3,000 words for the main body. These can be seen in the submitted report.",
        f"Word count. The main body is {wc} words, counted over the six sections only. The title block, tables, figure captions, reference list and the evidence appendix are excluded, as the brief allows.",
        "Punctuation check. An automated check found no em dash in the report or in this statement.",
        "Grounding. Every number in the report comes from a run recorded in artifacts/metrics.json and artifacts/run_metadata.json, and every appendix item is output produced by the repository. No result, figure or citation was invented, and the references are real published sources.",
        "Checks that run on the code. A suite of sixteen automated tests, a lint pass and a GitHub Actions run all pass, and the appendix shows them.",
    ]


UNDERSTAND = [
    "I have read the report and the code from start to finish.",
    "I can put the main parts into my own words: why mortality is excluded as leakage and a competing outcome, why the baseline's 86% accuracy is misleading on imbalanced data, how keeping preprocessing and the model in one saved object keeps training and serving consistent, and what the fairness result for the small Other group means.",
    "I understand which parts are limitations: the features carry weak signal, the metrics are modest, and the model is not ready to guide care on its own.",
]

RESPONSIBLE = [
    "No real patient or personal data was used. The dataset is synthetic.",
    "Claude was instructed not to invent any result, source or figure, and to ground every number in the actual run. I rely on that instruction and have checked the numbers against the saved metrics.",
    "Where the work touches Aboriginal and Torres Strait Islander data governance, the report relies on published sources, and no community was consulted in preparing a synthetic academic exercise.",
    "I have stated Claude's part in full, including that it wrote the code and drafted the text and this statement, so the work can be judged with that in view.",
]


def build() -> None:
    wc = count_words()
    doc = db.new_document()
    db.footer_pagenum(doc, "AI Usage Statement | S00213739")

    from docx.enum.text import WD_ALIGN_PARAGRAPH
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("AI Usage Statement")
    run.bold = True
    run.font.size = db.Pt(18)
    run.font.color.rgb = db.NAVY
    doc.add_paragraph()

    for label, value in HEADER:
        p = doc.add_paragraph()
        r = p.add_run(f"{label}: ")
        r.bold = True
        r.font.size = db.Pt(10.5)
        r2 = p.add_run(value)
        r2.font.size = db.Pt(10.5)

    db.heading2(doc, "1. Summary")
    db.body(doc, SUMMARY)

    db.heading2(doc, "2. What I decided and did myself")
    db.bullets(doc, DECIDED)

    db.heading2(doc, "3. What the AI assisted with")
    p = doc.add_paragraph()
    r = p.add_run("Table 1. Tasks for which AI assistance was used.")
    r.italic = True
    r.font.size = db.Pt(9)
    r.font.color.rgb = db.GREY
    db.table(doc, TABLE_HEADERS, TABLE_ROWS)

    db.heading2(doc, "4. Supporting evidence")
    db.bullets(doc, evidence_items(wc))

    db.heading2(doc, "5. How I made sure I understand the work")
    db.bullets(doc, UNDERSTAND)

    db.heading2(doc, "6. Responsible and ethical use")
    db.bullets(doc, RESPONSIBLE)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)


if __name__ == "__main__":
    build()
    print(f"ai statement -> {OUT}")
