param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

# Keep uv, its cache, and its managed Python on the project drive.
$env:UV_INSTALL_DIR = Join-Path $ProjectRoot "tools"
$env:UV_CACHE_DIR = Join-Path $ProjectRoot "uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot "python"
$env:Path = "$($env:UV_INSTALL_DIR);$($env:Path)"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$($env:UV_INSTALL_DIR);$($env:Path)"
}

uv python install 3.12 --no-registry --no-bin
if (-not (Test-Path ".venv-edof\Scripts\python.exe")) {
    uv venv .venv-edof --python 3.12
}

$Python = Join-Path $ProjectRoot ".venv-edof\Scripts\python.exe"
uv pip install --python $Python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $Python -r requirements-edof.txt

& $Python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
& $Python -c "import importlib.metadata; print('deeplens-core', importlib.metadata.version('deeplens-core'))"
& $Python -m pytest -q tests/test_edof_reproduction_poly1d.py tests/test_edof_reproduction_nafnet.py tests/test_edof_reproduction_pipeline.py

$DatasetRoot = Join-Path $ProjectRoot "datasets\DIV2K_train_HR"
if (-not (Test-Path $DatasetRoot)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "datasets") | Out-Null
    $Archive = Join-Path $ProjectRoot "datasets\DIV2K_train_HR.zip"
    if (-not (Test-Path $Archive)) {
        curl.exe -L --retry 5 --output $Archive "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    }
    Expand-Archive -Path $Archive -DestinationPath (Join-Path $ProjectRoot "datasets") -Force
}

Write-Output "WINDOWS_EDOF_READY"
