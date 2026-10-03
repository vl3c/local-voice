<#
.SYNOPSIS
    Set up a fully local voice-design TTS (Breeze TTS 2) with a sound-design
    effects chain, on this machine's NVIDIA GPU.

.DESCRIPTION
    Clones the Breeze TTS 2 inference code at a pinned commit into engine\,
    builds an isolated venv in .venv\, installs a CUDA build of PyTorch and the
    dependencies, and fetches the weights with checksum verification into
    models\. Nothing here uploads anything.

    Re-running is safe: every step is skipped if it is already done.

    Works on Windows PowerShell 5.1 and PowerShell 7+.

.PARAMETER PythonVersion
    Python version to build the venv from, passed to the 'py' launcher.

.PARAMETER ReuseModels
    The models\ folder of another local-voice install. Its weights are hard-linked
    (or copied, across drives) instead of downloaded, then checksum-verified.

.PARAMETER CudaIndex
    PyTorch wheel index. cu128 is the oldest that supports RTX 50-series
    (Blackwell, sm_120) cards.

.EXAMPLE
    .\install.ps1

.EXAMPLE
    .\install.ps1 -SkipModels

.EXAMPLE
    .\install.ps1 -ReuseModels D:\old\local-voice\models
#>
[CmdletBinding()]
param(
    [string] $PythonVersion = '3.13',
    [string] $CudaIndex     = 'cu128',
    [switch] $SkipModels,
    [string] $ReuseModels   = '',
    [switch] $NoPipCache
)

$ErrorActionPreference = 'Stop'

$RepoRoot = $PSScriptRoot
if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $RepoRoot) { $RepoRoot = (Get-Location).Path }

# The Breeze TTS 2 inference code revision this repo was built and tested against.
$ENGINE_REPO   = 'https://github.com/breezeblue-ai/breeze-tts.git'
$ENGINE_COMMIT = '58ec70c'
$TORCH_VERSION = '2.9.1'   # what engine\breeze-tts\requirements.txt pins
# The Romanian engine, OmniVoice, at the commit this repo was tested with.
$OMNI_PKG      = 'git+https://github.com/k2-fsa/OmniVoice.git@08be0b4ccbac3e13e374e86fbfead4b4cac343e2'

function Step ($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok   ($m) { Write-Host "    $m"   -ForegroundColor Green }
function Note ($m) { Write-Host "    $m"   -ForegroundColor DarkGray }
function Fail ($m) { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }

# Windows PowerShell 5.1 turns any stderr output from a native command into a
# terminating error while $ErrorActionPreference is 'Stop'; probes go through here.
function Probe {
    param([Parameter(Mandatory)][string] $Exe, [string[]] $Arguments = @())
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = (& $Exe @Arguments 2>&1 | Out-String)
        [pscustomobject]@{ Code = $LASTEXITCODE; Out = $out.Trim() }
    } catch {
        [pscustomobject]@{ Code = -1; Out = $_.Exception.Message }
    } finally {
        $ErrorActionPreference = $prev
    }
}

$engine = Join-Path $RepoRoot 'engine\breeze-tts'
$venv   = Join-Path $RepoRoot '.venv'
$venvPy = Join-Path $venv 'Scripts\python.exe'
$roVenv = Join-Path $RepoRoot '.venv-ro'
$roPy   = Join-Path $roVenv 'Scripts\python.exe'

# --- prerequisites --------------------------------------------------------
Step 'Checking prerequisites'
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail 'git is not on PATH. Install Git for Windows: https://git-scm.com/download/win'
}
Ok "git $((git --version) -replace 'git version ')"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Fail "The 'py' launcher is not on PATH. Install Python from python.org with the launcher option ticked."
}
$probe = Probe 'py' @("-$PythonVersion", '-c', 'import sys; sys.stdout.write(sys.executable)')
if ($probe.Code -ne 0) {
    Fail "Python $PythonVersion not found. Installed:`n$((Probe 'py' @('--list')).Out)`n       Pass -PythonVersion with one of those (3.10 or newer)."
}
Ok "Python $PythonVersion at $($probe.Out)"

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Ok "GPU: $((Probe 'nvidia-smi' @('--query-gpu=name,memory.total', '--format=csv,noheader')).Out)"
} else {
    Note 'nvidia-smi not found. Without an NVIDIA GPU this falls back to CPU and is very slow.'
}

$drive = Get-PSDrive -Name ($RepoRoot.Substring(0, 1)) -ErrorAction SilentlyContinue
if ($drive) {
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    Ok "$freeGB GB free"
    if ($freeGB -lt 25) { Note 'Under 25 GB free. Weights are ~12.6 GB, the two venvs ~10 GB, plus pip cache.' }
}

# --- engine ---------------------------------------------------------------
Step "Breeze TTS 2 inference code at $engine"
if (Test-Path (Join-Path $engine '.git')) {
    # An interrupted run can leave the clone on the wrong commit: check, and fix it.
    $head = (Probe 'git' @('-C', $engine, 'rev-parse', 'HEAD')).Out
    $want = (Probe 'git' @('-C', $engine, 'rev-parse', "$ENGINE_COMMIT^{commit}")).Out
    if ($head -and $head -eq $want) {
        Ok "Already cloned at $ENGINE_COMMIT"
    } else {
        Note "Clone is at $head, not $ENGINE_COMMIT; checking out the pinned commit"
        git -C $engine fetch --quiet origin
        git -C $engine checkout --quiet $ENGINE_COMMIT
        if ($LASTEXITCODE -ne 0) { Fail "could not check out $ENGINE_COMMIT in $engine (local changes? move it aside)" }
    }
} else {
    if (Test-Path $engine) { Fail "$engine exists but is not a git checkout. Move it aside." }
    New-Item -ItemType Directory -Path (Split-Path $engine -Parent) -Force | Out-Null
    git clone --quiet $ENGINE_REPO $engine
    if ($LASTEXITCODE -ne 0) { Fail 'git clone failed' }
    git -C $engine checkout --quiet $ENGINE_COMMIT
    if ($LASTEXITCODE -ne 0) { Fail "could not check out $ENGINE_COMMIT" }
    Ok 'Cloned'
}
Note "HEAD: $(git -C $engine log -1 --format='%h %ci %s')"

# --- virtualenv -----------------------------------------------------------
Step 'Python environment'
if (Test-Path $venvPy) {
    Ok 'venv already exists'
} else {
    & py "-$PythonVersion" -m venv $venv
    if (-not (Test-Path $venvPy)) { Fail 'venv creation failed' }
    Ok 'venv created'
}

$pipFlags = @('--disable-pip-version-check')
if ($NoPipCache) { $pipFlags += '--no-cache-dir' }

$torchProbe = Probe $venvPy @('-c', 'import torch;print(torch.__version__)')
if ($torchProbe.Code -eq 0) {
    Ok "torch $($torchProbe.Out) already installed"
} else {
    Note "Installing torch $TORCH_VERSION from the $CudaIndex index (~3 GB download)..."
    & $venvPy -m pip install --quiet @pipFlags --upgrade pip
    & $venvPy -m pip install @pipFlags --index-url "https://download.pytorch.org/whl/$CudaIndex" "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION"
    if ($LASTEXITCODE -ne 0) { Fail "torch install failed. Try -CudaIndex cu130." }
    Ok 'torch installed'
}

# Last line only: the very first CUDA initialisation can print a warning ahead of the answer.
# -W ignore: on a fresh venv NumPy isn't installed yet, and torch warns about it on stderr.
$cuda = Probe $venvPy @('-W', 'ignore', '-c', 'import torch;print(torch.cuda.is_available())')
if ((($cuda.Out -split "`r?`n") | Select-Object -Last 1).Trim() -ne 'True') {
    Note $cuda.Out
    Note 'torch cannot see CUDA. Delete .venv and re-run with a different -CudaIndex.'
} else {
    $gpu = ((Probe $venvPy @('-W', 'ignore', '-c', 'import torch;print(torch.cuda.get_device_name(0))')).Out -split "`r?`n") | Select-Object -Last 1
    Ok "CUDA available, $gpu"
}

Step 'Dependencies'
# The engine's list minus torch (already installed from the CUDA index, and a
# plain PyPI resolve would swap it for a CPU build) and its dev tools.
$engineReq = Get-Content (Join-Path $engine 'requirements.txt') |
    Where-Object { $_ -and $_ -notmatch '^(torch|torchaudio|pytest|ruff)\b' }
$filtered = Join-Path $env:TEMP 'local-voice-engine-req.txt'
Set-Content -Path $filtered -Value $engineReq -Encoding ascii
& $venvPy -m pip install --quiet @pipFlags -r $filtered -r (Join-Path $RepoRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { Fail 'dependency install failed' }
$torchAfter = (Probe $venvPy @('-c', 'import torch;print(torch.__version__)')).Out
if ($torchAfter -notmatch '\+cu') { Fail "a dependency replaced the CUDA torch with $torchAfter" }
Ok "installed (torch $torchAfter)"

# Triton lets torch.compile build the depth decoder's fast path. Without it the
# server still works, just with --safe (about 6x slower).
$t = Probe $venvPy (@('-m', 'pip', 'install', '--quiet') + $pipFlags + @('triton-windows<3.6'))
if ($t.Code -eq 0) { Ok 'triton-windows installed' }
else { Note 'triton-windows unavailable for this Python; start the server with --safe' }

# --- Romanian engine ------------------------------------------------------
# OmniVoice needs transformers 5.x; Breeze pins 4.57. So it gets its own venv,
# and the server runs it as a child process (server\omni_worker.py).
Step 'Romanian engine (OmniVoice) environment'
if (-not (Test-Path $roPy)) {
    & py "-$PythonVersion" -m venv $roVenv
    if (-not (Test-Path $roPy)) { Fail '.venv-ro creation failed' }
}
if ((Probe $roPy @('-c', 'import torch')).Code -ne 0) {
    & $roPy -m pip install --quiet @pipFlags --upgrade pip
    & $roPy -m pip install @pipFlags --index-url "https://download.pytorch.org/whl/$CudaIndex" "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION"
    if ($LASTEXITCODE -ne 0) { Fail 'torch install into .venv-ro failed' }
}
# --no-deps: OmniVoice's metadata pins an older torch that would replace the CUDA build.
& $roPy -m pip install --quiet @pipFlags --no-deps $OMNI_PKG
if ($LASTEXITCODE -ne 0) { Fail 'OmniVoice install failed' }
# --no-warn-conflicts: OmniVoice lists gradio, tensorboardX and webdataset, which inference
# doesn't need; pip would otherwise print a scary but harmless 'ERROR' about them.
& $roPy -m pip install --quiet --no-warn-conflicts @pipFlags -r (Join-Path $RepoRoot 'requirements-ro.txt')
if ($LASTEXITCODE -ne 0) { Fail 'Romanian engine dependency install failed' }
$omniProbe = Probe $roPy @('-W', 'ignore', '-c', 'import torch, omnivoice; print(torch.__version__)')
$omniLast = (($omniProbe.Out -split "`r?`n") | Select-Object -Last 1).Trim()
if ($omniProbe.Code -ne 0 -or $omniLast -notmatch '\+cu') {
    Note $omniProbe.Out
    Fail 'OmniVoice does not import with a CUDA torch in .venv-ro'
}
Ok "OmniVoice ready (torch $omniLast)"

# --- weights --------------------------------------------------------------
if ($SkipModels) {
    Step 'Skipping model download (-SkipModels)'
    Note "Fetch them later with:  & `"$venvPy`" scripts\download_models.py"
} else {
    Step 'Model weights (~12.6 GB, checksum-verified)'
    Note 'Resumable - if this is interrupted, just re-run install.ps1.'
    $dlArgs = @((Join-Path $RepoRoot 'scripts\download_models.py'))
    if ($ReuseModels) {
        if (-not (Test-Path $ReuseModels)) { Fail "-ReuseModels: $ReuseModels does not exist" }
        $dlArgs += @('--reuse', (Resolve-Path $ReuseModels).Path)
        Note "Reusing weights from $ReuseModels where they match"
    }
    & $venvPy @dlArgs
    if ($LASTEXITCODE -ne 0) { Fail 'model download or verification failed' }
}

Step 'Done'
Write-Host @"

  Start it with:      .\start.cmd
  Then open:          http://127.0.0.1:8190/

  Breeze TTS 2 weights and everything you generate with them are for
  research and non-commercial use only. See LICENSE-MODELS.md.

"@ -ForegroundColor Green
