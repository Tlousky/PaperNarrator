@echo off
setlocal enabledelayedexpansion

echo === VibeVoice-1.5B Setup Script (Windows) ===
echo.

set "MODEL_NAME=microsoft/VibeVoice-1.5B"
set "VOICE_NAMES=Carter Wayne Avery Carter"
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
pip install huggingface_hub soundfile pydub transformers torch

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
echo Downloading VibeVoice-1.5B model (%MODEL_NAME%)...
echo This may take 5-10 minutes depending on your connection.

python - << 'PYTHON_SCRIPT'
import os
from huggingface_hub import snapshot_download

model_name = "microsoft/VibeVoice-1.5B"
local_dir = f"./models/{model_name}"

print(f"Downloading {model_name}...")
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    resume_download=True,
    local_dir_use_symlinks=False
)
print(f"Model downloaded to {local_dir}")
PYTHON_SCRIPT

if %ERRORLEVEL% neq 0 (
    echo Failed to download model.
    exit /b 1
)

echo.
echo Downloading voice samples...

python - << 'PYTHON_SCRIPT'
import os
from huggingface_hub import hf_hub_download

model_name = "microsoft/VibeVoice-1.5B"
voices = ["Carter", "Wayne", "Avery", "Carter"]
voices_dir = f"./models/{model_name}/voices"

os.makedirs(voices_dir, exist_ok=True)

for voice in voices:
    print(f"Downloading {voice}...")
    hf_hub_download(
        repo_id=model_name,
        filename=f"voices/{voice}.pt",
        local_dir=f"./models/{model_name}",
        local_dir_use_symlinks=False
    )

print(f"Voice samples downloaded to {voices_dir}")
PYTHON_SCRIPT

if %ERRORLEVEL% neq 0 (
    echo Failed to download voice samples.
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
