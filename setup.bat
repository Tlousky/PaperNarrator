@echo off
setlocal enabledelayedexpansion

echo === PaperNarrator Setup ===
echo.

REM Check if we're in the project directory
if not exist "pyproject.toml" (
    echo Error: Please run this script from the PaperNarrator project root directory
    exit /b 1
)

REM Step 1: Setup VibeVoice
echo [Step 1/2] Setting up VibeVoice TTS model...
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

REM Step 2: Setup ffmpeg
echo [Step 2/2] Setting up ffmpeg...
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
