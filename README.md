# EDOF lens reproduction

This is a standalone project for reproducing the public EDOF hybrid-lens
training protocol described in Section 6.2 of the paper. It is intentionally
separate from the OptiResearch Agent project.

The implementation contains only the EDOF optical runner, public DeepLens
integration, Poly1D DOE, 16-level DOE quantization, cached ray-to-wave fields,
NAFNet reconstruction, local trace/artifact recording, configurations, tests,
and Windows/macOS environment scripts.

DeepLens is installed from the public repository at commit
`7df9613ca06be4093d094ad3095bd8712641a77d` so the optical API does not drift
between machines.

The paper does not release the complete Optolife prescription, trained DOE,
sensor response, exact DOE-to-sensor spacing, or exact NAFNet training details.
This project therefore reproduces the disclosed protocol with the public A489
DeepLens base lens and records that boundary in every run; it does not claim
bit-identical numerical results.

## Project layout

- `edof_reproduction/`: standalone runner and model code.
- `configs/edof_reproduction/`: A489 DOE, Mac smoke, and Windows full configs.
- `scripts/windows_edof_bootstrap.ps1`: installs uv-managed Python and CUDA
  dependencies entirely below the project directory.
- `scripts/windows_edof_start.ps1`: starts checkpoint-resumable training.
- `scripts/windows_edof_worker.ps1`: persistent scheduled-task worker used so
  training survives SSH disconnection.
- `tests/`: Poly1D, NAFNet, optics, and end-to-end smoke tests.

## Mac smoke test

```bash
uv venv .venv-edof --python 3.12
uv pip install --python .venv-edof/bin/python torch torchvision
uv pip install --python .venv-edof/bin/python -r requirements-edof.txt
.venv-edof/bin/python -m edof_reproduction \
  --config configs/edof_reproduction/mac_smoke.yaml \
  --output workspace/edof_reproduction/mac_smoke
```

The smoke configuration runs three joint epochs on a small synthetic dataset.

## Windows setup and full training

Clone this repository to the D drive, for example `D:\edof`, then run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows_edof_bootstrap.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows_edof_start.ps1
```

The bootstrap script puts uv, its cache, managed Python, and `.venv-edof` under
the repository directory. The full run writes checkpoints and logs to
`workspace\edof_reproduction\windows_full_actual` and resumes from
`checkpoints\latest.pt` when present. The start script registers the
`EDOFFullTraining` Windows scheduled task, so closing SSH does not stop Python.

Monitor the detached process with:

```powershell
Get-Content .\workspace\edof_reproduction\windows_full_actual\windows_train.stdout.log -Tail 30 -Wait
```

The Windows default uses a 256-point simulation grid and 5x5 field cache to fit
an RTX 5060 8 GB card. The paper's 6000/12000-point wave grids require much more
GPU memory and are not the default staged run.

## Tests

```bash
python -m pytest -q
```
