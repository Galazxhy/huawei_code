param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$scripts = @(
    "Q4_1\Q_4_1_1_rebuild_C8.py",
    "Q4_1\Q_4_1_2_match_C1_C4.py",
    "Q4_1\Q_4_1_3_c6_bridge_plan.py",
    "Q4_2\Q_4_2_1_sample_and_frontier.py",
    "Q4_2\Q_4_2_2_compute_frontier.py",
    "Q4_2\Q_4_2_3_bridge_and_mechanism.py",
    "Q4_2\Q_4_2_4_forecast.py",
    "Q4_2\Q_4_2_5_validation.py",
    "Q4_2\Q_4_2_7_verify_outputs.py"
)

foreach ($relativePath in $scripts) {
    $scriptPath = Join-Path $root $relativePath
    $workingDirectory = Split-Path -Parent $scriptPath
    Write-Host "Running $relativePath"
    Push-Location $workingDirectory
    try {
        & $Python $scriptPath
        if ($LASTEXITCODE -ne 0) {
            throw "$relativePath failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}
