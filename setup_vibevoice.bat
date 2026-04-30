@echo off
setlocal enabledelayedexpansion

echo === VibeVoice-Realtime-0.5B Setup Script (Windows) ===
echo.

set "MODEL_NAME=microsoft/VibeVoice-Realtime-0.5B"
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
echo Downloading VibeVoice-Realtime-0.5B model (%MODEL_NAME%)...
echo This may take 2-5 minutes (~2GB).

python - << 'PYTHON_SCRIPT'
import os
from huggingface_hub import snapshot_download

model_name = "microsoft/VibeVoice-Realtime-0.5B"
local_dir = f"./models/{model_name}"

print(f"Downloading {model_name}...")
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print(f"Model downloaded to {local_dir}")
PYTHON_SCRIPT

if %ERRORLEVEL% neq 0 (
    echo Failed to download model.
    exit /b 1
)

echo.
echo Downloading voice samples from GitHub...

python - << 'PYTHON_SCRIPT'
import os
import requests

model_name = "microsoft/VibeVoice-Realtime-0.5B"
voices_dir = f"./models/{model_name}/voices"
os.makedirs(voices_dir, exist_ok=True)

voices = [
    ("Carter", "en-Carter_man.pt"),
    ("Davis", "en-Davis_man.pt"),
    ("Emma", "en-Emma_woman.pt"),
]

for name, filename in voices:
    url = f"https://raw.githubusercontent.com/microsoft/VibeVoice/main/demo/voices/streaming_model/{filename}"
    dest_path = f"{voices_dir}/{name}.pt"
    print(f"  Downloading {name}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

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
echo   python -c "from tts.vibevoice import VibeVoiceTTS; tts = VibeVoiceTTS(model_name='./models/microsoft/VibeVoice-Realtime-0.5B', speaker_name='Carter'); tts.generate_audio('Hello world', 'test.wav')"

echo.
echo Setup successful!
pause
