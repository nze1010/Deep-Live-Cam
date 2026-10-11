@echo off
setlocal
cd /d "%~dp0"

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting Administrator privileges...
    powershell -NoProfile -Command "Start-Process cmd.exe -ArgumentList '/c `\"%~dpnx0`\"' -Verb RunAs"
    exit /b
)

regsvr32.exe /u "%~dp0UnityCaptureFilter64.dll"
regsvr32.exe /u "%~dp0UnityCaptureFilter32.dll"
echo Virtual Camera uninstalled.
pause
