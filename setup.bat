@echo off
REM Solar Detection Project Setup Script
REM Automatically creates virtual environment and installs dependencies

echo ========================================
echo Solar Detection Project Setup
echo ========================================
echo.

REM Check if virtual environment exists
if exist "solar\Scripts\activate.bat" (
    echo [INFO] Virtual environment 'solar' already exists
    goto :install_deps
)

echo [STEP 1] Creating virtual environment 'solar'...
python -m venv solar
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    echo Please ensure Python 3.8+ is installed and in PATH
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment created
echo.

:install_deps
echo [STEP 2] Activating virtual environment...
call solar\Scripts\activate.bat

echo.
echo [STEP 3] Detecting GPU availability...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo [INFO] PyTorch not installed yet, will install based on requirements
    set GPU_DETECTED=0
) else (
    for /f "tokens=*" %%i in ('python -c "import torch; print(torch.cuda.is_available())"') do set GPU_CHECK=%%i
    if "%GPU_CHECK%"=="True" (
        set GPU_DETECTED=1
        echo [INFO] GPU detected - will use CUDA requirements
    ) else (
        set GPU_DETECTED=0
        echo [INFO] No GPU detected - will use CPU requirements
    )
)

echo.
echo [STEP 4] Installing dependencies...
if %GPU_DETECTED%==1 (
    if exist "requirements_cuda.txt" (
        echo [INFO] Installing CUDA requirements from requirements_cuda.txt
        pip install --upgrade pip
        pip install -r requirements_cuda.txt
        if exist "requirements.txt" (
            pip install -r requirements.txt
        )
    ) else (
        echo [WARNING] requirements_cuda.txt not found, using requirements.txt
        pip install --upgrade pip
        pip install -r requirements.txt
    )
) else (
    if exist "requirements_cpu.txt" (
        echo [INFO] Installing CPU requirements from requirements_cpu.txt
        pip install --upgrade pip
        pip install -r requirements_cpu.txt
    ) else (
        echo [INFO] Installing requirements from requirements.txt
        pip install --upgrade pip
        pip install -r requirements.txt
    )
)

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Virtual environment 'solar' is ready
echo To activate: solar\Scripts\activate.bat
echo.
echo Quick Start Commands:
echo   - Test inference:  python Segmentation\MaskRCNN_Solar\inference_finetuned.py --mode test
echo   - Full inference:  python Segmentation\MaskRCNN_Solar\inference_finetuned.py --mode full
echo   - Train model:     python Segmentation\MaskRCNN_Solar\finetune_solar_detector.py
echo.
pause
