"""Guard: excluded columns never reach the model, and the target is never a feature."""
from src.config import load_config
from src.ingest import load_raw
from src.pipeline import build_pipeline


def _active_model_inputs(pipe):
    """Column names fed to the active transformers (everything except remainder='drop')."""
    ct = pipe.named_steps["preprocess"]
    used = []
    for name, _trans, cols in ct.transformers_:
        if name == "remainder":
            continue
        if isinstance(cols, (list, tuple)):
            used.extend(cols)
    return used


def test_excluded_columns_not_in_model_inputs():
    cfg = load_config()
    df = load_raw(cfg).head(300)
    target = cfg["task"]["target"]
    X = df.drop(columns=[target])
    y = df[target]

    pipe = build_pipeline(cfg, "logistic_regression")
    pipe.fit(X, y)
    used = _active_model_inputs(pipe)

    assert "mortality" not in used, "mortality must be excluded (leakage / competing risk)"
    assert cfg["task"]["id_column"] not in used, "patient_id must be excluded (identifier)"
    assert target not in used, "the target must never be a feature"
