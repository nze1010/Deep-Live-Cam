@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo   Installing Deep Live Cam Virtual Camera Driver
echo ========================================================
echo.

echo Registering 64-bit Virtual Camera...
regsvr32.exe "%~dp0UnityCaptureFilter64.dll"

echo Registering 32-bit Virtual Camera...
regsvr32.exe "%~dp0UnityCaptureFilter32.dll"

echo.
echo ========================================================
echo   Virtual Camera is now registered!
echo   You will now see "Unity Video Capture" in WhatsApp / Zoom.
echo ========================================================
echo.
pause
