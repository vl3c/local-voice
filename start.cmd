@echo off
rem Start the local voice server in its own window.
rem Close that window to stop the server and release the GPU.
rem
rem   start.cmd                 the server opens in its own window and this returns at once;
rem                             the page opens in your browser once it's ready
rem   start.cmd --no-browser    the same without opening the page (what agents use)
rem   start.cmd --safe          eager mode: no CUDA-graph warmup, ~6x slower, for when the
rem                             fast-path warmup fails (combines with --no-browser)
rem   start.cmd --here          run in this console instead, for scripts that manage the
rem                             process themselves
setlocal
cd /d "%~dp0"
rem Captured before the option loop: it shifts the arguments (shift /1 leaves %0 alone,
rem but keeping these in variables makes that impossible to get wrong later).
set "SELF=%~f0"
set "HERE_DIR=%~dp0"

rem Windows' own timeout. Started from Git Bash, plain "timeout" can resolve to the GNU
rem coreutils one, which doesn't understand /t.
set "WAIT=%SystemRoot%\System32\timeout.exe"

if not exist "%HERE_DIR%.venv\Scripts\python.exe" (
  echo.
  echo No install found. Run install.ps1 first.
  echo.
  pause
  exit /b 1
)

set "BROWSER=1"
set "HERE="
set "INWIN="
set "EXTRA="
:args
if "%~1"=="" goto argsdone
if /i "%~1"=="--no-browser" (set "BROWSER=" & shift /1 & goto args)
if /i "%~1"=="--safe"       (set "EXTRA=--safe" & shift /1 & goto args)
if /i "%~1"=="--here"       (set "HERE=1" & shift /1 & goto args)
if /i "%~1"=="--in-window"  (set "INWIN=1" & shift /1 & goto args)
echo Unknown option "%~1". Use:  start.cmd [--no-browser] [--safe] [--here]
exit /b 2
:argsdone

rem By default, relaunch in a new, visible console and return immediately, so the server is
rem always on screen and closing its window shuts it down (Windows ends every process
rem attached to a console when the console closes).
if defined HERE goto run
if defined INWIN goto run
set "PASS=%EXTRA%"
if not defined BROWSER set "PASS=--no-browser %EXTRA%"
start "local-voice server - close this window to stop it" cmd /c ""%SELF%" --in-window %PASS%"
exit /b 0

:run
if defined INWIN (
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
"%HERE_DIR%.venv\Scripts\python.exe" "%HERE_DIR%server\app.py" --port 8190 %EXTRA%

if defined INWIN (
  echo.
  echo   The server has stopped ^(exit code %ERRORLEVEL%^). This window closes in 60 seconds;
  echo   press a key to close it now.
  "%WAIT%" /t 60 >nul
)

endlocal
