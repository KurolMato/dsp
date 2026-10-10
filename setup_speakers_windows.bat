@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap_windows.ps1" -Mode speakers
set "setupResult=%errorlevel%"
if not "%setupResult%"=="0" (
    echo [ERROR] Setup failed. Install local 64-bit Python 3.12 and retry.
)
pause
exit /b %setupResult%
