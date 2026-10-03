@echo off
rem Start the local voice server and open its page.
rem Close this window to stop the server and release VRAM.
rem
rem   start.cmd                 loads the model, then opens http://127.0.0.1:8190/
rem   start.cmd --no-browser    server only, for scripts and agents
setlocal
cd /d "%~dp0"

set "BROWSER=1"
if /i "%~1"=="--no-browser" set "BROWSER="

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo.
  echo No install found. Run install.ps1 first.
  echo.
  pause
  exit /b 1
)

rem Wait for the server to answer before opening the browser: the model takes
rem a little while to load.
if defined BROWSER start "" /min cmd /c "for /l %%i in (1,1,150) do (curl.exe -s -o nul --max-time 2 http://127.0.0.1:8190/api/health && (start "" http://127.0.0.1:8190/ & exit /b) || timeout /t 2 >nul)"

rem The server binds to 127.0.0.1 only (hard-coded in server\app.py).
"%~dp0.venv\Scripts\python.exe" "%~dp0server\app.py" --port 8190

endlocal
