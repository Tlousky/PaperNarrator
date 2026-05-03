@echo off
setlocal enabledelayedexpansion

echo === VibeVoice-1.5B Setup Script (Windows) ===
echo.

set "MODEL_NAME=microsoft/VibeVoice-1.5B"
set "VOICE_NAMES=Carter Davis Emma"
set "PYTHON_VENV=.venv"

echo Checking Python environment...
if not exist "%PYTHON_VENV%\Scripts\activate.bat" (
    echo Python virtual environment '%PYTHON_VENV%' not found. Creating one now...
    python -m venv %PYTHON_VENV%
)

call %PYTHON_VENV%\Scripts\activate.bat

echo Using Python: %PYTHON%
python --version

echo.
echo Installing dependencies...
pip install --upgrade pip
pip install huggingface_hub soundfile pydub transformers torch requests

echo.
echo Checking for ffmpeg...
where ffmpeg >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ffmpeg not found. Downloading portable version...
    powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip' -OutFile 'ffmpeg.zip'}"
    powershell -Command "& {Expand-Archive -Path 'ffmpeg.zip' -DestinationPath 'ffmpeg' -Force}"
    set "PATH=%CD%\ffmpeg\bin;%PATH%"
    echo Added ffmpeg to PATH
)

echo.
python download_vibevoice.py

if %ERRORLEVEL% neq 0 (
    echo Failed to download model or voices.
    exit /b 1
)

echo.
echo Creating test text file...
mkdir test_data 2>nul

(
echo Hello, this is a test of the VibeVoice text-to-speech system.
echo This sample text is short enough to generate quickly but long enough to demonstrate the quality.
echo Thank you for listening.
) > test_data\sample.txt

echo Created test_data\sample.txt

echo.
echo === Setup Complete ===
echo Model location: .\models\%MODEL_NAME%
echo Voice samples: .\models\%MODEL_NAME%\voices\
echo Test file: test_data\sample.txt
echo.
echo Available voices: %VOICE_NAMES%
echo.
echo To test:
echo   python -c "from tts.vibevoice import VibeVoiceTTS; tts = VibeVoiceTTS(model_name='./models/microsoft/VibeVoice-1.5B', speaker_name='Carter'); tts.generate_audio('Hello world', 'test.wav')"

echo.
echo Setup successful!
pause
