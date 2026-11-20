# Run ThaiOCRBench (300 samples) on 3 Ollama models in rounds of 100,
# so every model always has the same prefix of the seed-42 shuffle if we have to stop early.
# Resumable: re-running skips finished items.
param([int[]]$Rounds = @(100, 200, 300))
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = '1'
$py = Join-Path $root '.venv\Scripts\python.exe'
$models = @('typhoon', 'qwen3vl-base-q8', 'qwen3vl-thaidoc-q8')
New-Item -ItemType Directory -Force ollama_pred\logs, ollama_res | Out-Null

foreach ($n in $Rounds) {
    foreach ($m in $models) {
        $log = "ollama_pred\logs\$m.log"
        "=== round max_samples=$n  $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
        cmd /c "`"$py`" ollama_eval\run_ollama.py --model $m --max_samples $n --output_path ollama_pred\${m}_300.json >> $log 2>&1"
        ollama stop $m 2>$null   # free VRAM before the next model (does not delete anything)
        cmd /c "`"$py`" eval_scripts\eval.py --input_path ollama_pred\${m}_300.json --output_path ollama_res\${m}_300.json > ollama_res\${m}_300.eval.txt 2>&1"
    }
    "round $n finished $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 ollama_pred\logs\rounds.log
}
"ALL DONE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 ollama_pred\logs\rounds.log
