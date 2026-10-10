@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap_windows.ps1" -Mode music
set "setupResult=%errorlevel%"
if not "%setupResult%"=="0" (
    echo [ERROR] Setup failed. See the error above, check Python 3.12 and network access, then retry.
)
pause
exit /b %setupResult%
