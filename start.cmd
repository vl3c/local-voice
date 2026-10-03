@echo off
rem Start the local voice server.
rem Close the server's window to stop it and release the GPU.
rem
rem   start.cmd                 loads the model, then opens http://127.0.0.1:8190/
rem   start.cmd --window        server only, in its own window (what agents use): returns at
rem                             once, and closing that window stops it
rem   start.cmd --window --safe the same in eager mode: no CUDA-graph warmup, ~6x slower,
rem                             for when the fast-path warmup fails
rem   start.cmd --no-browser    server only, in this console (for scripts that manage it)
setlocal
cd /d "%~dp0"

rem Windows' own timeout. Started from Git Bash, plain "timeout" can resolve to the GNU
rem coreutils one, which doesn't understand /t.
set "WAIT=%SystemRoot%\System32\timeout.exe"

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo.
  echo No install found. Run install.ps1 first.
  echo.
  pause
  exit /b 1
)

set "EXTRA="
if /i "%~2"=="--safe" set "EXTRA=--safe"

rem --window: relaunch this script in a new, visible console and return immediately, so
rem the server is always on screen and closing its window shuts it down (Windows ends
rem every process attached to a console when the console closes).
if /i "%~1"=="--window" (
  start "local-voice server - close this window to stop it" cmd /c ""%~f0" --in-window %EXTRA%"
  exit /b 0
)

set "BROWSER=1"
if /i "%~1"=="--no-browser" set "BROWSER="
if /i "%~1"=="--in-window" set "BROWSER="
if /i "%~1"=="--in-window" (
  title local-voice server - close this window to stop it
  echo.
  echo   local-voice server on http://127.0.0.1:8190/
  echo   Close this window to stop it and free the GPU.
  echo.
)

rem Wait for the server to answer before opening the browser: the model takes
rem a little while to load.
if defined BROWSER start "" /min cmd /c "for /l %%i in (1,1,150) do (curl.exe -s -o nul --max-time 2 http://127.0.0.1:8190/api/health && (start "" http://127.0.0.1:8190/ & exit /b) || "%WAIT%" /t 2 >nul)"

rem The server binds to 127.0.0.1 only (hard-coded in server\app.py).
"%~dp0.venv\Scripts\python.exe" "%~dp0server\app.py" --port 8190 %EXTRA%

if /i "%~1"=="--in-window" (
  echo.
  echo   The server has stopped ^(exit code %ERRORLEVEL%^). This window closes in 60 seconds;
  echo   press a key to close it now.
  "%WAIT%" /t 60 >nul
)

endlocal
