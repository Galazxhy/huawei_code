param(
    [string]$Python = "python"
)

# Runs the q2 stages in dependency order. Every script is launched from its own
# directory, so a script's working directory never changes what it reads.
$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

# matplotlib needs a writable cache directory; the per-user default is not
# always writable, which only shows up as a warning until a plot is saved.
$mplCache = Join-Path $root ".." ".mplcfg"
New-Item -ItemType Directory -Force -Path $mplCache | Out-Null
$env:MPLCONFIGDIR = (Resolve-Path $mplCache).Path

# Every script's full output is kept under <repo>/log/<timestamp>Q2/.
$logRoot = Join-Path $root ("..\log\" + (Get-Date -Format "yyyyMMdd-HHmmss") + "Q2")
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
Write-Host "Logs: $logRoot"

$stages = @(
    @{
        Name = "Q2.1 classic scaling law"
        Scripts = @(
            "Q_2_1/Q_2_1_1_classic_law.py",
            "Q_2_1/Q_2_1_2_classic_law_early_stopping.py",
            "Q_2_1/Q_2_1_3_classic_law_analysis.py",
            "Q_2_1/Q_2_1_4_classic_law_figures_zh.py"
        )
    },
    @{
        Name = "Q2.2 generalized scaling law with quality and mixture"
        Scripts = @(
            "Q_2_2/Q_2_2_1_generalized_law.py"
        )
    },
    @{
        # Q_2_3_1 writes the handoff payload Q_2_2_2 draws (mixture_response.json),
        # so the generalized-law figures run after it, not with Q2.2.
        Name = "Q2.3 handoff to problem 3"
        Scripts = @(
            "Q_2_3/Q_2_3_1_handoff_build.py",
            "Q_2_2/Q_2_2_2_generalized_law_figures.py",
            "Q_2_3/Q_2_3_2_handoff_paper_form.py",
            "Q_2_3/Q_2_3_3_handoff_identification.py"
        )
    },
    @{
        Name = "Q2.4 marginal utility and substitution"
        Scripts = @(
            "Q_2_4/Q_2_4_1_elasticity_analysis.py",
            "Q_2_4/Q_2_4_2_elasticity_figures.py",
            "Q_2_4/Q_2_4_3_factor_elasticity_analysis.py",
            "Q_2_4/Q_2_4_4_factor_elasticity_figures.py",
            "Q_2_4/Q_2_4_5_quality_substitution_analysis.py",
            "Q_2_4/Q_2_4_6_quality_substitution_figures.py",
            "Q_2_4/Q_2_4_7_joint_substitution_analysis.py",
            "Q_2_4/Q_2_4_8_joint_substitution_figures.py"
        )
    }
)

foreach ($stage in $stages) {
    Write-Host "== $($stage.Name) =="
    foreach ($relativePath in $stage.Scripts) {
        $scriptPath = Join-Path $root $relativePath
        $workingDirectory = Split-Path -Parent $scriptPath
        $logPath = Join-Path $logRoot ((Split-Path -Leaf $relativePath) -replace '\.py$', '.log')
        Write-Host "Running $relativePath"
        Write-Host "  log -> $logPath"
        Push-Location $workingDirectory
        try {
            # Tee so progress stays visible while the full output is kept in log/.
            & $Python $scriptPath 2>&1 | Tee-Object -FilePath $logPath
            if ($LASTEXITCODE -ne 0) {
                throw "$relativePath failed with exit code $LASTEXITCODE"
            }
        }
        finally {
            Pop-Location
        }
    }
}
