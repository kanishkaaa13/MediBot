@echo off
REM MediBot - one-click setup (Windows). Run from the medibot folder.
python -m pip install -r requirements.txt
if errorlevel 1 goto fail
python generate_world.py
if errorlevel 1 goto fail
python tests\test_core.py
if errorlevel 1 goto fail
echo.
echo ===== Setup OK. Now open worlds\medibot.wbt in Webots and press Run =====
pause
exit /b 0
:fail
echo.
echo Something failed - copy the messages above and send them to me.
pause
exit /b 1
