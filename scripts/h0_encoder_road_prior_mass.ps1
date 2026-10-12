param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('train', 'sweep', 'test', 'topology')]
    [string]$Mode,
    [ValidateSet('full', 'off')]
    [string]$Variant = 'off',
    [string]$RunDir = '',
    [int]$Seed = 1234,
    [string]$DataRoot = 'D:\Code\Massachusetts\tiff',
    [string]$Pretrain = 'D:\Code\Swin-Unet-main\pretrained_ckpt\swinv2_tiny_patch4_window8_256.pth',
    [string]$Python = 'D:\torch-cu128\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$env:OPENCV_LOG_LEVEL = 'ERROR'
$Repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $Repo
$Cfg = Join-Path $Repo 'configs\swin_tiny_patch4_window7_224_lite.yaml'
$Evaluation = Join-Path $Repo 'evaluation\road_comparison'
$Results = Join-Path $Repo 'model_out'

function Require-Success([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit code $LASTEXITCODE)" }
}

if ($Mode -eq 'train') {
    if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) { throw "Missing data: $DataRoot" }
    if (-not (Test-Path -LiteralPath $Pretrain -PathType Leaf)) { throw "Missing pretrain: $Pretrain" }
    & $Python .\scripts\check_mass_layout.py --root_path $DataRoot --source_size 1500
    Require-Success 'Massachusetts layout check'

    if (-not $RunDir) {
        $RunName = 'h0_mass1500_roadprior_{0}_fp32_noema80_s{1}_{2}' -f $Variant, $Seed, (Get-Date -Format 'yyyyMMdd_HHmmss')
        $RunDir = Join-Path $Results $RunName
    } else {
        $RunDir = [System.IO.Path]::GetFullPath($RunDir)
        $Results = Split-Path -Path $RunDir -Parent
        $RunName = Split-Path -Path $RunDir -Leaf
    }
    if (Test-Path -LiteralPath $RunDir) { throw "Run directory exists: $RunDir" }

    $PriorArg = if ($Variant -eq 'off') { '--no-enable_encoder_road_prior' } else { '--enable_encoder_road_prior' }
    $TrainArgs = @(
        '.\train_image.py',
        '--root_path', $DataRoot, '--output_dir', $Results, '--run_name', $RunName,
        '--cfg', $Cfg, '--pretrain_ckpt', $Pretrain,
        '--pretrained_lr', '5e-5', '--pretrained_min_lr', '5e-6',
        '--new_lr', '2e-4', '--new_min_lr', '1e-5', '--warmup_epochs', '10',
        '--amp_dtype', 'none', '--max_epochs', '80', '--val_interval', '5',
        '--batch_size', '2', '--accumulation_steps', '2', '--num_workers', '4',
        '--img_size', '512', '--source_patch_size', '1500', '--overlap_stride', '256',
        '--random_crop_train', '--random_crops_per_image', '1',
        '--no-use_ema', '--ema_decay', '0.999', '--seed', [string]$Seed, '--threshold', '0.2',
        '--structure_profile', 'stage23_boundary_0626', '--stage_skeleton_mode', 'direct',
        '--enable_highres_structure_stream', '--enable_h3_surface_fusion',
        '--remove_stage2_pre_topology_source',
        '--highres_structure_fuse_stages', 'stage23',
        '--highres_structure_fusion_mode', 'stage23',
        '--enable_global_topology', '--global_topology_max_nodes', '32',
        '--global_topology_heads', '4', '--global_topology_alpha_max', '0.05',
        '--stage2_skeleton_weight', '0.008', '--stage3_skeleton_weight', '0.012',
        '--stage2_skeleton_gradient_ratio', '0.5',
        '--stage3_skeleton_gradient_ratio', '0.5',
        '--stage3_gate_topology_gradient_ratio', '0.0',
        '--skeleton_pos_weight', '2.0', '--highres_structure_skeleton_weight', '0.0',
        '--stage_direction_factor', '0.2', '--stage_connectivity_factor', '3.0',
        '--road_attention_weight', '0.0', '--masked_connectivity_center_experiment',
        '--connectivity_pos_weight', '2.0', '--connectivity_focal_gamma', '1.5',
        '--edge_contrastive_margin', '0.1',
        '--directional_pos_weight_cardinal', '1.0',
        '--directional_pos_weight_diagonal', '2.5', '--surface_focal_gamma', '1.0',
        $PriorArg
    )
    Write-Host "RUN_DIR=$RunDir"
    & $Python @TrainArgs
    Require-Success 'H0 Massachusetts training'
    return
}

if (-not $RunDir) { throw '-RunDir is required for sweep, test and topology' }
$RunDir = [System.IO.Path]::GetFullPath($RunDir)
$Best = Join-Path $RunDir 'best.pth'
if (-not (Test-Path -LiteralPath $Best -PathType Leaf)) { throw "Missing checkpoint: $Best" }
$LatestAuditFile = Join-Path $RunDir 'latest_road_prior_audit.txt'

if ($Mode -eq 'sweep') {
    $Audit = Join-Path $RunDir ('road_prior_audit_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
    if (Test-Path -LiteralPath $Audit) { throw "Audit directory exists: $Audit" }
    New-Item -ItemType Directory -Path $Audit | Out-Null
    $PredictionRoot = Join-Path $Audit 'predictions'
    $Manifest = Join-Path $Audit 'comparison.json'
    $Thresholds = @('0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60')

    & $Python (Join-Path $Evaluation 'build_h0_manifest.py') `
        --checkpoint $Best --data_root $DataRoot `
        --predictions_root $PredictionRoot --output $Manifest --check_only
    Require-Success 'Checkpoint protocol check'
    $InferenceArgs = @(
        '.\test_image.py', '--root_path', $DataRoot, '--model_path', $Best,
        '--cfg', $Cfg, '--split', 'val', '--img_size', '512',
        '--source_patch_size', '1500', '--overlap_infer',
        '--output_dir', (Join-Path $PredictionRoot 'val'), '--export_thresholds'
    ) + $Thresholds
    & $Python @InferenceArgs
    Require-Success 'Validation sliding inference'
    & $Python (Join-Path $Evaluation 'build_h0_manifest.py') `
        --checkpoint $Best --data_root $DataRoot `
        --predictions_root $PredictionRoot --output $Manifest
    Require-Success 'Validation manifest'
    & $Python (Join-Path $Evaluation 'compare_predictions.py') `
        --manifest $Manifest --split val --select_threshold_only `
        --output_dir (Join-Path $Audit 'val') --progress_every 10
    Require-Success 'Validation threshold sweep'
    Set-Content -LiteralPath $LatestAuditFile -Value $Audit -Encoding UTF8
    Write-Host "SWEEP_CSV=$(Join-Path $Audit 'val\val_threshold_scores.csv')"
    Write-Host "SELECTION=$(Join-Path $Audit 'val\val_selection.json')"
    return
}

if (-not (Test-Path -LiteralPath $LatestAuditFile -PathType Leaf)) { throw 'Run sweep first' }
$Audit = (Get-Content -LiteralPath $LatestAuditFile -Raw).Trim()
$PredictionRoot = Join-Path $Audit 'predictions'
$Manifest = Join-Path $Audit 'comparison.json'
$Selection = Join-Path $Audit 'val\val_selection.json'
if (-not (Test-Path -LiteralPath $Selection -PathType Leaf)) { throw "Missing selection: $Selection" }

if ($Mode -eq 'test') {
    $Selected = Get-Content -LiteralPath $Selection -Raw | ConvertFrom-Json
    $ModelEntries = @($Selected.models.PSObject.Properties)
    if ($ModelEntries.Count -ne 1) { throw 'Expected exactly one model in validation selection' }
    $Threshold = [double]$ModelEntries[0].Value.threshold
    $ThresholdText = $Threshold.ToString('0.00', [System.Globalization.CultureInfo]::InvariantCulture)
    & $Python .\test_image.py `
        --root_path $DataRoot --model_path $Best --cfg $Cfg `
        --split test --img_size 512 --source_patch_size 1500 --overlap_infer `
        --output_dir (Join-Path $PredictionRoot 'test') `
        --threshold $ThresholdText --export_thresholds $ThresholdText
    Require-Success 'Test sliding inference'
    & $Python (Join-Path $Evaluation 'build_h0_manifest.py') `
        --checkpoint $Best --data_root $DataRoot `
        --predictions_root $PredictionRoot --output $Manifest --selection $Selection
    Require-Success 'Test manifest'
}

& $Python (Join-Path $Evaluation 'compare_predictions.py') `
    --manifest $Manifest --split test --selection $Selection `
    --output_dir (Join-Path $Audit 'test') --progress_every 10
Require-Success 'Test topology metrics'
Write-Host "TEST_SUMMARY=$(Join-Path $Audit 'test\test_comparison.csv')"
Write-Host "TEST_PER_IMAGE_DIR=$(Join-Path $Audit 'test')"
