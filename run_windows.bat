@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0bootstrap_windows.ps1" (
    echo [ERROR] Missing bootstrap_windows.ps1. Copy the complete startup patch.
    pause
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap_windows.ps1" -Mode run
set "runResult=%errorlevel%"
if not "%runResult%"=="0" pause
exit /b %runResult%
