# One-command pipeline stages for Windows PowerShell.
# Usage:  ./run.ps1 all   (or: validate | train | evaluate | diagram | predict | test | lint)
param([string]$task = "all")
$ErrorActionPreference = "Stop"

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

function Validate { & $py -m src.validate }
function Train    { & $py -m src.train }
function Evaluate { & $py -m src.evaluate }
function Diagram  { & $py -m src.make_diagram }
function Sample   { & $py -m src.make_sample }
function Predict  { Sample; & $py -m src.predict --input data/sample_new_patients.csv --output artifacts/predictions.csv }
function Test     { & $py -m pytest -q }
function Lint     { & $py -m ruff check src tests }

switch ($task) {
    "validate" { Validate }
    "train"    { Train }
    "evaluate" { Evaluate }
    "diagram"  { Diagram }
    "sample"   { Sample }
    "predict"  { Predict }
    "test"     { Test }
    "lint"     { Lint }
    "all" {
        Validate; Train; Evaluate; Diagram; Predict; Test
        Write-Host "Pipeline complete."
    }
    default { Write-Host "Unknown task: $task" }
}
