@echo off
setlocal enabledelayedexpansion

echo === FFmpeg Setup Script (Windows) ===

REM Detect architecture
set "ARCH=x86_64"
where python >nul 2>&1
if %ERRORLEVEL%==0 (
    python -c "import platform; print(platform.architecture()[0])" 2>nul | find "64" >nul
    if %ERRORLEVEL% neq 0 set "ARCH=i686"
)

echo Detected Architecture: %ARCH%

REM Set filename for Windows 64-bit
set "FILENAME=ffmpeg-master-latest-win64-gpl.zip"
set "URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/%FILENAME%"

REM Create ffmpeg directory
if not exist .ffmpeg mkdir .ffmpeg

echo Downloading ffmpeg...
REM Download using PowerShell
powershell -Command "& {Invoke-WebRequest -Uri '%URL%' -OutFile '%FILENAME%'}"

if not exist "%FILENAME%" (
    echo Failed to download ffmpeg
    exit /b 1
)

echo Extracting...
REM Extract using PowerShell
powershell -Command "& {Expand-Archive -Path '%FILENAME%' -DestinationPath '.ffmpeg' -Force}"

REM Clean up
if exist "%FILENAME%" del "%FILENAME%"

REM Verify
if exist .ffmpeg\bin\ffmpeg.exe (
    echo FFmpeg installed successfully:
    .\ffmpeg\bin\ffmpeg.exe -version | findstr /C:"ffmpeg version"
    echo.
    echo FFmpeg binary location: .\ffmpeg\bin\ffmpeg.exe
    echo Add this to your PATH: %CD%\ffmpeg\bin
) else (
    echo Error: Could not install ffmpeg
    exit /b 1
)

echo FFmpeg setup complete!
