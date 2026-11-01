$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.venv-amd\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    py -3.14 -m venv (Join-Path $taskRoot '.venv-amd')
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.14 doit être installé pour cet environnement local.' }
}
& $taskPython -m pip install --index-url 'https://stable.repo.amd.com/rocm/whl-next/' 'torch[device-gfx1200]==2.14.0+rocm10.1.0'
if ($LASTEXITCODE -ne 0) { throw 'Installation PyTorch AMD interrompue.' }
& $taskPython -m pip install -e '.[train,vis]' ipykernel 'numpy==2.4.3' 'pillow==12.1.1' 'opencv-python-headless==4.11.0.86'
if ($LASTEXITCODE -ne 0) { throw 'Installation des dépendances interrompue.' }
& $taskPython -m ipykernel install --user --name ascii-flux-amd --display-name 'ASCII Flux - Radeon GPU'
& $taskPython scripts/check_gpu.py
if ($LASTEXITCODE -ne 0) { throw 'Le test de rétropropagation GPU a échoué. Le gros entraînement ne démarre pas.' }
