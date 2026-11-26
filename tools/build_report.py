"""Build the technical report (DOCX) from structured content and the real run evidence.

The prose is grounded in the artefacts produced by the pipeline (artifacts/*.json,
reports/figures/*, reports/evidence/*). Running this script regenerates the report, so
the document itself is reproducible. The main-body word count is computed and printed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import docbuild as db

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
EVID = ROOT / "reports" / "evidence"
ART = ROOT / "artifacts"
OUT = ROOT / "reports" / "S00213739_AINT602_A2_Technical_Report.docx"

REPO = "https://github.com/sriramsuri/AINT602-A2-readmission-pipeline"

# --- Main body content. Each block is (kind, payload). "p" and "b" blocks are counted
# --- toward the word limit; headings, tables, figures and captions are not. -----------

SECTIONS = [
    ("System purpose, prediction context and pipeline architecture", [
        ("p", "I built a system that estimates the risk that a patient will return to hospital within 30 days of being discharged. The target is readmission_30d, a yes or no outcome, so the task is binary classification. Each prediction is made for one discharge, which means one patient episode rather than one person across their whole history."),
        ("p", "The prediction is meant to support a care team, not to replace it. In a real service it would run at the point of discharge and give the discharge planner a short list of higher risk patients to follow up, for example with a phone call or an earlier review. The people affected are wider than the users. Patients carry the benefit of better follow up and the cost of being contacted when they did not need it, and a patient who is missed carries the most serious cost of all."),
        ("p", "Prediction time decides which columns are honest features. At discharge I know the patient's demographics, the vital signs and laboratory results recorded during the stay, the recorded comorbidities, and the discharge diagnosis. I do not yet know whether the patient later dies, and a patient who dies cannot be readmitted, so mortality is a competing outcome that only becomes known after the prediction would be made. I excluded it for that reason. I also excluded patient_id, which is only a label. Both exclusions are written into config.yaml with their reasons, so the decision is visible rather than assumed."),
        ("fig", (FIG / "architecture.png", "Figure 1. Pipeline architecture, generated from the implemented stage names in src/make_diagram.py so the diagram stays aligned with the code.", 5.4)),
        ("p", "Figure 1 shows the architecture. Raw data is ingested, validated against a schema, then passed through feature engineering and preprocessing before it is used to train and benchmark models. The evaluation step produces the metrics and figures, MLflow and the structured JSON files record each run, the selected pipeline is saved as a single artefact, and a separate prediction workflow reuses that artefact on new data. Monitoring sits at the end as a consideration beyond the project."),
        ("p", "The shape of the pipeline follows the task rather than a template. Because the data is clinical and imbalanced, validation and leakage control come early and are given real weight, the feature logic and preprocessing are kept inside one saved object so the same steps run at training and at prediction, and the evaluation is built around metrics that survive class imbalance rather than accuracy. Each of these choices is made for this problem, and the sections below explain and justify them in turn."),
    ]),
    ("Data ingestion, validation and feature engineering", [
        ("p", "Ingestion is a single function in src/ingest.py that reads the dataset from the path named in config.yaml and returns a data frame. Keeping the source in configuration means a file, a URL or a database query could be swapped in later without changing any downstream stage."),
        ("p", "Validation is the part I gave the most weight, because the sample data is unusually clean, with 10,000 rows, no missing values and no duplicates, and real data will not be. src/validate.py checks each batch against a contract in data/schema.json: the required columns are present, the types parse, values fall inside clinically plausible ranges, categories belong to an allowed set, and there are no duplicate rows or identifiers. The same module runs in two modes. In strict mode, used for training, a quality problem is an error and stops the run, so a corrupt batch cannot quietly train a model. In lenient mode, used when serving, the same problem becomes a warning and the pipeline's imputer and encoder absorb it, so one odd value does not crash a live prediction. Appendix A2 shows the validation output, and the tests in tests/test_validate.py confirm that corrupted inputs, such as an age of 500 or a sex outside the allowed set, are caught."),
        ("p", "I added four engineered features in src/features.py. Mean arterial pressure and pulse pressure are derived from the two blood pressure readings and carry circulatory information that neither reading gives on its own. Comorbidity count combines diabetes and hypertension into a single burden score. BMI category places the numeric BMI into clinical bands, which lets a linear model use it without assuming a straight line. Each one is computed row by row from values available at discharge, so none of them reaches into the future."),
        ("p", "Two decisions protect against leakage and against training and serving drifting apart, which is a well known source of failure in machine learning systems (Sculley et al., 2015). First, every transformation that learns from the data, namely median imputation, scaling and one-hot encoding, sits inside a scikit-learn pipeline and is fitted on the training split only (Pedregosa et al., 2011), so nothing from the validation or test data leaks into the fitted numbers. Second, the feature engineering is a transformer inside that same pipeline rather than a separate script, so the prediction path in src/predict.py applies the identical steps used in training. A test in tests/test_pipeline.py passes raw rows with no engineered columns to the fitted pipeline and confirms it still predicts, which shows the engineering happens inside the object and cannot be skipped at serving time. The column selector also drops every column it is not told to use, so the excluded mortality and identifier columns cannot slip into the model even when they are present in the input, and tests/test_leakage.py fails if anyone adds them back."),
    ]),
    ("Model development, benchmarking and evaluation", [
        ("p", "I split the data into training, validation and test sets of 6,400, 1,600 and 2,000 rows, stratified so the 14% readmission rate is held steady in each. The split has a clear division of labour: models are fitted on training, compared on validation, and the chosen model is reported once on test. The operating threshold is also chosen on validation, never on test, so the test numbers are not tuned on the data they report."),
        ("p", "The baseline always predicts the majority class, which is not readmitted. It reaches 86% accuracy on the test set and yet catches none of the 278 readmissions. That is the reason for including it. On imbalanced data a high accuracy can describe a model that is useless for the decision that matters, which is why I report precision, recall and the area under the precision-recall curve rather than leaning on accuracy (Saito & Rehmsmeier, 2015)."),
        ("table", (
            ["Model", "Val PR-AUC", "Test PR-AUC", "Test ROC-AUC", "Recall", "Precision"],
            [
                ["Baseline (most frequent)", "0.139", "0.139", "0.500", "0.00", "0.00"],
                ["Logistic regression (selected)", "0.219", "0.203", "0.602", "0.54", "0.19"],
                ["Histogram gradient boosting", "0.177", "0.171", "0.555", "0.01", "0.17"],
            ],
            "Table 1. Benchmark on the held-out test set. Recall and precision are reported at each model's operating threshold (0.50 for the baseline, 0.48 for the candidates).",
        )),
        ("p", "I trained two candidates, a logistic regression with balanced class weights and a histogram gradient boosting model. Logistic regression had the stronger validation PR-AUC, 0.219 against 0.177, and was selected. On the test set it reached a PR-AUC of 0.203 against the baseline's 0.139 and an ROC-AUC of 0.602. At the chosen threshold of 0.48 it found 54% of the real readmissions at a precision of 0.19, which means about one in five flagged patients was in fact readmitted. The ROC, precision-recall, confusion and calibration figures are in Appendix A9."),
        ("p", "The gradient boosting model is worth a sentence because it did worse here, not better. Without class weighting, and judged at the threshold tuned for logistic regression, it flagged almost nobody and its recall fell to near zero. That is a useful result rather than an awkward one, because it shows that the more complex model is not automatically the better one, especially on an imbalanced target and without its own tuning."),
        ("p", "I read these numbers as a modest but real gain. The selected model ranks patients better than chance and clearly better than the baseline, and at an operating point that favours recall it surfaces just over half of the patients who go on to return. It is not accurate enough to act on by itself. The signal in these features is weak, which is not surprising, because the strongest predictors of readmission in practice, such as previous admissions, length of stay, discharge destination and medications, are not in this dataset (Kansagara et al., 2011). The evaluation shows that the pipeline works and that the model beats a fair baseline. It does not show that the model is ready for care decisions, and the distance between those two statements is the honest conclusion."),
    ]),
    ("Versioning, tracking and reproducibility", [
        ("p", "The project is a git repository whose history follows how it was built, from the first commit that adds the configuration and data to the later commits for validation, the model pipeline, training, evaluation, the tests and the run evidence (Appendix A1). The public repository is linked on the title page."),
        ("p", "Three things are tracked so that a run can be reproduced. The code is versioned by git. The environment is pinned in requirements.txt, with the full transitive freeze in requirements.lock.txt, so the same library versions install again. This mattered in practice, because the project runs on pandas 3.1 and MLflow 3.18, and both changed behaviour from earlier versions during the build. The settings that define a run, the seed, the split sizes, the feature lists and the excluded columns, live in config.yaml instead of being scattered through the code."),
        ("p", "Each training run is recorded twice. MLflow logs the parameters, metrics and saved model under a named experiment (Zaharia et al., 2018), shown in Appendix A6. In parallel, artifacts/metrics.json holds the validation and test metrics for every model, and artifacts/run_metadata.json records the seed, the exact library versions, the row count, the split sizes, the selected model, the threshold and a SHA-256 hash of the data file (Appendix A5). That hash is what lets another person confirm they are training on the same data I used."),
        ("p", "Together these answer the question the brief asks, which is how a team member could tell which data, code, configuration and evidence produced the selected model. The data is identified by its hash, the code by its commit, the configuration by config.yaml, and the metrics and model by the files in artifacts/. The README gives the setup and run instructions, and a second person can reproduce the selected model by installing the locked environment and running the training and evaluation commands."),
    ]),
    ("Automation, quality checks and reusable prediction workflow", [
        ("p", "Every stage can be run with a single command, either through the module interface, for example python -m src.train, or through the Makefile and the PowerShell script run.ps1, which chain the stages together. This keeps the workflow repeatable and removes any need to remember an order or a set of manual steps."),
        ("p", "Quality is checked automatically in three ways. A ruff lint pass checks the code, a pytest suite of sixteen tests checks behaviour, and the data validation runs as its own step. The tests are written around the claims that matter rather than around easy lines. They confirm that bad data is caught, that the engineered features compute the right values, that the excluded columns never reach the model, and that a row passed to the fitted pipeline is transformed the same way at serving as during training. A GitHub Actions workflow in .github/workflows/ci.yml runs the lint, the validation and the tests on every push, on Python 3.12, which also shows that the pinned stack reproduces on a different Python version from the one I built on. Appendix A3 and A4 show a passing test run and a passing CI run."),
        ("p", "The trained pipeline is saved as one object in artifacts/model.joblib together with its operating threshold. src/predict.py loads that object and applies it to a CSV of new records, writing a probability and a label for each. Because the saved object is the whole pipeline, with feature engineering, preprocessing and model inside it, the prediction path cannot apply different logic from training. Appendix A7 shows the sample input and the predictions it produced."),
        ("p", "To move this from a project to a production workflow, a few things would be added rather than changed. The batch script would be wrapped in a scheduled job or a small service, the validation step would raise alerts rather than only write a report, the model and its threshold would be versioned in a registry with an approval step before release, and the monitoring described below would be connected. None of this changes the pipeline's structure, which is why the structure was the first thing I built."),
    ]),
    ("Responsible AI, operational reliability and cultural safety", [
        ("p", "The users of this system are the discharge and care teams who would act on its output. The people affected are the patients, both those helped by earlier follow up and those contacted without need, and, less visibly, any group the model serves less well."),
        ("p", "The clearest limitation is in the data. The features are a thin slice of what drives readmission, so the model's ceiling is low and its output should be read as a rough sort rather than a judgement. The evaluation makes the error trade-off concrete. At the chosen threshold the model misses about 46% of the patients who are readmitted, the false negatives, and wrongly flags many who are not, the false positives. In this setting a false negative is the more serious error, because it means a patient who needed follow up did not get it, while a false positive mostly costs staff time, so an operating point that favours recall is defensible. That still has to be a deliberate decision made with clinicians, not a default."),
        ("p", "The errors are not shared evenly. When I broke the test results down by sex, recall was similar for female and male patients, 0.52 and 0.57, but for the small group of 25 patients recorded as Other the model caught none of the readmissions at all. That is a real result from my own run, not a hypothetical, and it is the kind of harm an aggregate score hides (Barocas et al., 2019). It happens because the group is too small for the model to learn from, and it is a reason not to deploy without addressing it, for example by collecting more data, reporting performance for each group as a standing check, and keeping a person in the loop for every flagged and unflagged case."),
        ("p", "Operationally, the parts most likely to fail are the data feeds, not the model. A changed column name, a new category, a delayed laboratory result or a value outside the expected range would each degrade predictions quietly. The validation step is the first defence against these, and in production it should raise alerts rather than only write a report. Over time the data will also drift away from what the model was trained on, so the inputs, the prediction distribution and the per-group performance should all be monitored, and the model revalidated and retrained on a schedule. The saved model is loaded with joblib, so the artefact itself would need to be stored securely and integrity checked, because loading a tampered file can run arbitrary code. Before any real use the system would also need governance, meaning a named owner, an approval step, and a clear statement that it advises and does not decide."),
        ("p", "On cultural safety, this dataset is synthetic and holds no real or Indigenous data, so no specific community is represented. That absence is not a clearance. If a system like this were built on real Australian hospital data, Aboriginal and Torres Strait Islander people's information would carry obligations of data sovereignty and governance (Walter & Suina, 2019). The right course would be to consult the relevant communities, follow the Indigenous data governance principles set out by bodies such as the Maiam nayri Wingara Indigenous Data Sovereignty Collective (2018), and check that the model does not widen existing gaps in care before it is used."),
    ]),
]

REFERENCES = [
    "Anthropic. (2026). Claude [Large language model]. https://claude.ai",
    "Barocas, S., Hardt, M., & Narayanan, A. (2019). Fairness and machine learning: Limitations and opportunities. fairmlbook.org.",
    "Kansagara, D., Englander, H., Salanitro, A., Kagen, D., Theobald, C., Freeman, M., & Kripalani, S. (2011). Risk prediction models for hospital readmission: A systematic review. JAMA, 306(15), 1688-1698.",
    "Maiam nayri Wingara Indigenous Data Sovereignty Collective. (2018). Indigenous data sovereignty communique. Canberra, Australia.",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3), e0118432.",
    "Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). Hidden technical debt in machine learning systems. In Advances in Neural Information Processing Systems (Vol. 28, pp. 2503-2511).",
    "Walter, M., & Suina, M. (2019). Indigenous data, indigenous methodologies and indigenous data sovereignty. International Journal of Social Research Methodology, 22(3), 233-243.",
    "Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., Murching, S., Nykodym, T., Ogilvie, P., Parkhe, M., Xie, F., & Zumar, C. (2018). Accelerating the machine learning lifecycle with MLflow. IEEE Data Engineering Bulletin, 41(4), 39-45.",
]

AI_USE_HEADING = "Use of artificial intelligence"
AI_USE_TEXT = (
    "Claude (Anthropic, 2026) drafted this report and assisted me in preparing the project. A "
    "separate AI usage statement, submitted with this assessment, records what I directed and "
    "decided, what Claude contributed, the evidence for that account, and how I checked that I "
    "understand the analysis and judgements it contains."
)


def _read(path: Path, max_lines: int | None = None) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").strip("\n")
    if max_lines:
        lines = text.splitlines()
        if len(lines) > max_lines:
            text = "\n".join(lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"])
    return text


def count_words() -> int:
    total = 0
    for _title, blocks in SECTIONS:
        for kind, payload in blocks:
            if kind == "p":
                total += len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'%.-]*", payload.replace("**", "")))
            elif kind == "b":
                for item in payload:
                    total += len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'%.-]*", item.replace("**", "")))
    return total


def build() -> int:
    words = count_words()
    doc = db.new_document()
    db.footer_pagenum(doc, "AINT602 Assessment 2 | S00213739")

    db.title_block(
        doc,
        "An MLOps Pipeline for 30-Day Hospital Readmission Prediction",
        "AINT602 Assessment 2: Applied MLOps Pipeline Project",
        [
            "Sriramkumar Suri THIRUKKOLLURI (S00213739)",
            f"Repository: {REPO}",
            f"Main body word count: {words}",
        ],
    )
    db.page_break(doc)

    for i, (title, blocks) in enumerate(SECTIONS, start=1):
        db.heading1(doc, f"{i}. {title}")
        for kind, payload in blocks:
            if kind == "p":
                db.body(doc, payload)
            elif kind == "b":
                db.bullets(doc, payload)
            elif kind == "fig":
                path, caption, width = payload
                db.figure(doc, path, caption, width_inches=width)
            elif kind == "table":
                headers, rows, caption = payload
                db.table(doc, headers, rows)
                cap = doc.add_paragraph()
                run = cap.add_run(caption)
                run.italic = True
                run.font.size = db.Pt(9)
                run.font.color.rgb = db.GREY

    db.heading1(doc, AI_USE_HEADING)
    db.body(doc, AI_USE_TEXT)

    # References
    db.heading1(doc, "References")
    for ref in REFERENCES:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = db.Inches(0.5)
        p.paragraph_format.first_line_indent = db.Inches(-0.5)
        p.add_run(ref).font.size = db.Pt(10.5)

    # Appendix
    db.page_break(doc)
    db.heading1(doc, "Evidence appendix")
    db.body(doc, "Each item below is output produced by running the pipeline in this repository. The technical report refers to these items by number.", justify=False)

    def appx(num, title, note=None):
        db.heading2(doc, f"Appendix {num}. {title}")
        if note:
            db.body(doc, note)

    appx("A1", "Repository and commit history",
         "The repository is public at the link on the title page. The commit history below reflects how the pipeline was built.")
    db.code_block(doc, _read(EVID / "git_log.txt"))

    appx("A2", "Data validation output (strict mode, full dataset)")
    db.code_block(doc, _read(EVID / "validation.txt"))

    appx("A3", "Automated tests and lint")
    db.code_block(doc, _read(EVID / "ruff.txt") + "\n\n" + _read(EVID / "pytest.txt"))

    appx("A4", "GitHub Actions continuous integration run")
    db.code_block(doc, _read(EVID / "ci_run.txt", max_lines=10))

    appx("A5", "Run metadata and provenance (artifacts/run_metadata.json)")
    db.code_block(doc, _read(ART / "run_metadata.json"))

    appx("A6", "MLflow experiment-tracking store (mlruns/)",
         "MLflow records each run on disk under mlruns/. The structure below lists one experiment with its runs, parameters, metrics and logged artefacts.")
    db.code_block(doc, _read(EVID / "mlruns_tree.txt", max_lines=26))

    appx("A7", "Sample prediction on new data")
    db.code_block(doc, _read(EVID / "predict.txt"))

    appx("A8", "Benchmark metrics (artifacts/metrics.json, selected excerpt)")
    metrics = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
    excerpt = {
        "selected_model": metrics["selected_model"],
        "threshold": round(metrics["threshold"], 4),
        "validation": metrics["validation"],
        "test_logistic_regression": {k: metrics["test"]["logistic_regression"][k]
                                     for k in ("roc_auc", "pr_auc", "recall", "precision", "f1")},
        "test_baseline": {k: metrics["test"]["baseline"][k] for k in ("accuracy", "pr_auc", "recall")},
    }
    db.code_block(doc, json.dumps(excerpt, indent=2))

    appx("A9", "Evaluation figures (test set)")
    db.figure(doc, FIG / "roc_curve.png", "Figure 2. ROC curve.", 3.4)
    db.figure(doc, FIG / "pr_curve.png", "Figure 3. Precision-recall curve against the base rate.", 3.4)
    db.figure(doc, FIG / "confusion_matrix.png", "Figure 4. Confusion matrix at the operating threshold.", 3.2)
    db.figure(doc, FIG / "calibration.png", "Figure 5. Calibration of predicted probabilities.", 3.4)
    db.figure(doc, FIG / "fairness_by_sex.png", "Figure 6. Recall and flagged rate by sex, showing the small Other group is not served.", 3.8)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return words


if __name__ == "__main__":
    n = build()
    print(f"main-body word count: {n}")
    print(f"report -> {OUT}")
