# Round 2 of text-to-JSON: train on the varied data, merge, load into Ollama, measure with the schema.
# Run from anywhere: powershell -File thai-doc-ocr\run_json_v2.ps1
# No $ErrorActionPreference = "Stop": PowerShell 5.1 turns any stderr line (a Python warning) into an error.
# Each step's exit code is checked instead.
$env:PYTHONIOENCODING = "utf-8"
$root = Split-Path $PSScriptRoot -Parent
$factory = Join-Path $root "LlamaFactory"
$here = $PSScriptRoot

Write-Output "== STAGE unload ollama"
ollama ps | Select-Object -Skip 1 | ForEach-Object { $name = ($_ -split "\s+")[0]; if ($name) { ollama stop $name } }

Write-Output "== STAGE train"
Set-Location $factory
& .\.venv\Scripts\python.exe ..\thai-doc-ocr\train.py ..\thai-doc-ocr\configs\qwen3_1.7b_json_train_v2.yaml 2>&1 |
    Out-File -Encoding utf8 ..\thai-doc-ocr\results\train_json_v2_log.txt
if ($LASTEXITCODE -ne 0) { Write-Output "== FAILED train (see results\train_json_v2_log.txt)"; exit 1 }

Write-Output "== STAGE export"
& .\.venv\Scripts\llamafactory-cli.exe export ..\thai-doc-ocr\configs\export_json_v2.yaml 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Output "== FAILED export"; exit 1 }

Write-Output "== STAGE ollama"
Set-Location $here
& ..\LlamaFactory\.venv\Scripts\python.exe to_ollama.py ..\LlamaFactory\models\qwen3-1.7b-thaidoc-json-v2 qwen3-1.7b-json-v2-q8 --text 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Output "== FAILED ollama"; exit 1 }

foreach ($run in @(@('test', 'typhoon'), @('test_long', 'typhoon'), @('test', 'truth'))) {
    Write-Output "== STAGE evaluate $($run[0]) $($run[1])"
    & ..\.venv\Scripts\python.exe evaluate_json.py qwen3-1.7b-json-v2-q8 --split $run[0] --source $run[1] 2>$null |
        Select-String "recall="
}
Write-Output "== DONE"
