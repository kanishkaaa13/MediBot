@echo off
REM run_medibot.bat  –  Regenerate world, validate, launch Webots
REM Run this from any directory; it always resolves relative to its own folder.

cd /d "%~dp0"

echo [1/3] Removing regression flag...
if exist "controllers\medibot_controller\regression.flag" (
    del "controllers\medibot_controller\regression.flag"
    echo       regression.flag removed.
) else (
    echo       regression.flag not present - OK.
)

echo.
echo [2/3] Regenerating world (generate_world.py + validate_world.py)...
py generate_world.py
if errorlevel 1 (
    echo.
    echo FAILED: world generation or validation failed. Fix the errors above.
    pause
    exit /b 1
)

echo.
echo [3/3] Launching Webots...
set WBT_PATH=%~dp0worlds\medibot.wbt
"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" "%WBT_PATH%"
