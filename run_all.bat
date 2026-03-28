@echo off
REM Quick start script for Essential Pipeline
REM Run all 3 analysis types on an image

if "%1"=="" (
    echo Usage: run_all.bat ^<image_path^>
    echo Example: run_all.bat test_images/full_body.jpg
    exit /b 1
)

set IMAGE=%1
echo Processing: %IMAGE%
echo.

echo [1/3] Extracting body measurements...
python smpl_process_image_lite.py "%IMAGE%" --device cpu
if errorlevel 1 goto error

echo.
echo [2/3] Generating DensePose segmentation...
python generate_densepose_iuv.py "%IMAGE%"
if errorlevel 1 goto error

echo.
echo [3/3] Generating 3D mesh...
python generate_3d_mesh.py "%IMAGE%" --device cpu
if errorlevel 1 goto error

echo.
echo ✓ All analysis complete! Check output/ folder for results.
exit /b 0

:error
echo ✗ Error processing image
exit /b 1
