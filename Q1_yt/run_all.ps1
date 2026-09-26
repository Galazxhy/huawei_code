param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$stages = @(
    @{
        Name = "Q1.1 quality representation"
        Directory = "Q1_1"
        Scripts = @(
            "Q_1_1_1_audit_A1.py",
            "Q_1_1_2_scalarize_A1.py",
            "Q_1_1_3_freeze_scale.py",
            "Q_1_1_3_audit_standardized_A1.py",
            "Q_1_1_4_structure.py",
            "Q_1_1_5_score_A1.py",
            "Q_1_1_6_validate_extensions.py",
            "Q_1_1_7_blind_text_review.py",
            "Q_1_1_7b_ai_review_complete.py",
            "Q_1_1_8_baselines.py",
            "Q_1_1_9_ai_text_audit.py",
            "Q_1_1_10_verify_outputs.py",
            "Q_1_1_11_sensitivity_A1.py",
            "Q_1_1_12_direction_evidence.py"
        )
    },
    @{
        Name = "Q1.2 conflict resolution"
        Directory = "Q1_2"
        Scripts = @(
            "Q_1_2_1_prepare_signals.py",
            "Q_1_2_2_relation_graph.py",
            "Q_1_2_3_crossfit_residuals.py",
            "Q_1_2_4_conflict_scores.py",
            "Q_1_2_5_cause_analysis.py",
            "Q_1_2_6_reliability_and_Qstar.py",
            "Q_1_2_7_baselines.py",
            "Q_1_2_8_sensitivity.py",
            "Q_1_2_9_text_cases.py",
            "Q_1_2_11_tail_evidence.py",
            "Q_1_2_10_verify_outputs.py"
        )
    },
    @{
        Name = "Q1.3 mixture modeling"
        Directory = "Q1_3"
        Scripts = @(
            "Q_1_3_1_load_and_standardize.py",
            "Q_1_3_2_scheffe_features.py",
            "Q_1_3_3_fit_scheffe_models.py",
            "Q_1_3_4_quality_prior.py",
            "Q_1_3_5_substitution_complement.py",
            "Q_1_3_6_validation.py",
            "Q_1_3_7_bootstrap_uncertainty.py",
            "Q_1_3_10_evidence_checks.py",
            "Q_1_3_11_final_checks.py",
            "Q_1_3_13_experiment_manifest.py",
            "Q_1_3_12_handoff.py",
            "Q_1_3_9_verify_outputs.py"
        )
    }
)

foreach ($stage in $stages) {
    $stageDirectory = Join-Path $root $stage.Directory
    New-Item -ItemType Directory -Force -Path (Join-Path $stageDirectory "results") | Out-Null
    Write-Host "== $($stage.Name) =="
    foreach ($scriptName in $stage.Scripts) {
        $scriptPath = Join-Path $stageDirectory $scriptName
        Write-Host "Running $scriptName"
        & $Python $scriptPath
        if ($LASTEXITCODE -ne 0) {
            throw "$scriptName failed with exit code $LASTEXITCODE"
        }
    }
}
