@echo off
setlocal
cd /d "%~dp0"

echo === VoiceDenoiser Setup ===

if not exist venv (
    echo [1/4] Creating venv...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create venv. Is Python installed?
        pause & exit /b 1
    )
) else (
    echo [1/4] venv already exists. Skipping.
)

where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo [2/4] NVIDIA GPU detected. Installing CUDA build of PyTorch...
    venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo [2/4] No NVIDIA GPU detected. Installing CPU build of PyTorch...
    venv\Scripts\python.exe -m pip install torch torchaudio
)
if errorlevel 1 (
    echo PyTorch install failed.
    pause & exit /b 1
)

echo [3/4] Installing dependencies...
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency install failed.
    pause & exit /b 1
)

echo [4/4] Downloading Resemble Enhance model (approx. 700MB)...
venv\Scripts\python.exe -m resemble_enhance.enhancer.download
if errorlevel 1 (
    echo Model download failed. Check your network and run setup.bat again.
    pause & exit /b 1
)

echo.
echo Setup complete. Run run.bat to launch.
pause
