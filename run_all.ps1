# Quick start script for Essential Pipeline
# Run all 3 analysis types on an image
# Usage: .\run_all.ps1 test_images/full_body.jpg

param(
    [Parameter(Mandatory=$true)]
    [string]$ImagePath
)

if (-not (Test-Path $ImagePath)) {
    Write-Host "Error: Image not found: $ImagePath" -ForegroundColor Red
    exit 1
}

Write-Host "Processing: $ImagePath" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] Extracting body measurements..." -ForegroundColor Yellow
python smpl_process_image_lite.py "$ImagePath" --device cpu
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "[2/3] Generating DensePose segmentation..." -ForegroundColor Yellow
python generate_densepose_iuv.py "$ImagePath"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "[3/3] Generating 3D mesh..." -ForegroundColor Yellow
python generate_3d_mesh.py "$ImagePath" --device cpu
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "✓ All analysis complete! Check output/ folder for results." -ForegroundColor Green
