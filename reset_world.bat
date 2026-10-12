@echo off
REM Regenerate the code-generated Webots world and assets; never save from Webots.
cd /d "%~dp0"
py generate_world.py
if errorlevel 1 exit /b 1
echo Generated world restored. Open worlds\medibot.wbt in Webots.
exit /b 0
