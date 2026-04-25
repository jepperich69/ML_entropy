param(
    [switch]$Quick
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\Users\rich\AppData\Local\miniconda3\python.exe"
$RunAnalysis = Join-Path $ScriptDir "run_probability_analysis.py"
$MakeFigures = Join-Path $ScriptDir "make_probability_figures.py"
$ResultsDir = Join-Path $ScriptDir "results"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python not found: $Python"
}

if ($Quick) {
    $Prefix = Join-Path $ResultsDir "probability_analysis_card3_quick"
    $Burn = 2000
    $Samples = 1000
    $Thin = 5
} else {
    $Prefix = Join-Path $ResultsDir "probability_analysis_card3"
    $Burn = 5000
    $Samples = 2000
    $Thin = 10
}

Write-Host "Running card-3 probability analysis"
Write-Host "Output prefix: $Prefix"
Write-Host "Burn=$Burn Samples=$Samples Thin=$Thin"

& $Python $RunAnalysis `
    --datasets compas monks1 tictactoe adult `
    --max-cardinality 3 `
    --burn $Burn `
    --samples $Samples `
    --thin $Thin `
    --out-prefix $Prefix

$FigurePrefix = Split-Path -Leaf $Prefix

Write-Host "Generating figures for $FigurePrefix"
& $Python $MakeFigures --prefix $FigurePrefix

Write-Host "Done."
