@echo off
setlocal enabledelayedexpansion

echo === PaperNarrator Setup ===
echo.

REM Check if we're in the project directory
if not exist "pyproject.toml" (
    echo Error: Please run this script from the PaperNarrator project root directory
    exit /b 1
)

REM Step 1: Install Python dependencies
echo [Step 1/3] Installing Python dependencies...
echo.
where uv >nul 2>&1
if %ERRORLEVEL% equ 0 (
    REM Create venv with Python 3.11 if available (better package compatibility)
    where python3.11 >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        uv venv --python 3.11 .venv 2>nul || uv venv .venv
    ) else (
        uv venv .venv
    )
    uv pip install -e . --compile-bytecode
    echo Python dependencies installed successfully.
    echo.
    echo [Step 1b/3] Installing CUDA-enabled PyTorch (cu128)...
    uv pip install --python .venv torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    echo CUDA PyTorch installed.
) else (
    echo Warning: 'uv' not found. Installing with pip...
    pip install -e .
    echo.
    echo Installing CUDA-enabled PyTorch...
    pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
)

echo.
echo.

REM Step 2: Setup VibeVoice
echo [Step 2/3] Setting up VibeVoice TTS model...
echo.
if exist "setup_vibevoice.bat" (
    call setup_vibevoice.bat
    if %ERRORLEVEL% neq 0 (
        echo Warning: VibeVoice setup failed. Continuing...
    )
) else (
    echo Warning: setup_vibevoice.bat not found. Skipping VibeVoice setup.
)

echo.
echo.

REM Step 3: Setup ffmpeg
echo [Step 3/3] Setting up ffmpeg...
echo.
if exist "download_ffmpeg.bat" (
    call download_ffmpeg.bat
    if %ERRORLEVEL% neq 0 (
        echo Warning: ffmpeg setup failed. Continuing...
    )
) else (
    echo Warning: download_ffmpeg.bat not found. Skipping ffmpeg setup.
)

echo.
echo === Setup Complete ===
echo.
echo Available commands:
echo   uv run python app.py              REM Start Gradio server
echo   uv run pytest                     REM Run tests
echo.
echo Note: If using MP3 output, add to your PATH:
echo   %CD%\ffmpeg\bin
echo.
pause
