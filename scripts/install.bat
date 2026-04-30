@echo off
REM PaperNarrator installer for Windows

python scripts\installer.py

if %ERRORLEVEL% neq 0 (
    echo Installation failed!
    pause
)
