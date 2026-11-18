$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pathsnapPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pathsnapPython)) {
    throw 'Create .venv and install requirements.txt first.'
}
$env:EVAL_DATA_DIR = 'data/evaluator_test'
Write-Host 'Open http://127.0.0.1:8000/ui/ in your browser. Press Ctrl+C to stop.'
& $pathsnapPython -m uvicorn api.main:app --host 127.0.0.1 --port 8000
