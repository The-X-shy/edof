# EDOF lens reproduction

This is a standalone project for reproducing the public EDOF hybrid-lens
training protocol described in Section 6.2 of the paper. It is intentionally
separate from the OptiResearch Agent project.

The implementation contains only the EDOF optical runner, public DeepLens
integration, Poly1D DOE, 16-level DOE quantization, cached ray-to-wave fields,
NAFNet reconstruction, independent DIV2K validation, local trace/artifact
recording, configurations, tests, and Windows/macOS environment scripts.

DeepLens is installed from the public repository at commit
`7df9613ca06be4093d094ad3095bd8712641a77d` so the optical API does not drift
between machines.

The paper does not release the complete Optolife prescription, trained DOE,
sensor response, or exact NAFNet training details.
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
  --config configs/edof_reproduction/mac_optimized_smoke.yaml \
  --output workspace/edof_reproduction/mac_optimized_smoke
```

The smoke configuration runs three joint epochs on a small synthetic dataset.

## Windows setup and full training

Clone this repository to the D drive, for example `D:\edof`, then run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows_edof_bootstrap.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows_edof_start.ps1
```

The bootstrap script puts uv, its cache, managed Python, and `.venv-edof` under
the repository directory. The recommended run writes checkpoints and logs to
`workspace\edof_reproduction\windows_optimized` and resumes from
`checkpoints\latest.pt` when present. The start script registers the
`EDOFOptimizedTraining` Windows scheduled task, so closing SSH does not stop Python.

Verify the real CUDA forward/backward path before starting the detached run:

```powershell
.\.venv-edof\Scripts\python.exe -m edof_reproduction `
  --config configs\edof_reproduction\windows_optimized_memory.yaml `
  --memory-smoke `
  --output workspace\edof_reproduction\windows_optimized_memory
```

Monitor the detached process with:

```powershell
Get-Content .\workspace\edof_reproduction\windows_optimized\windows_train.stdout.log -Tail 30 -Wait
```

The optimized configuration uses the paper-disclosed 5x5 EDoF PSF grid for
joint training, a 512-point double-precision wave grid, one million coherent
rays, 50 joint epochs, and up to 50 noisy network fine-tuning epochs. The fixed
optics stage interpolates the map to 40x40 as disclosed in the supplement.
Validation runs on all 100 DIV2K validation images every five epochs and records
mean PSNR, SSIM, and LPIPS. The best checkpoint is selected by validation PSNR
and fine-tuning stops after four validations without a meaningful improvement.

Training uses epoch-varying random resized crops, color jitter, and horizontal
flips. Its EDoF loss follows Equation 1 of the supplement: cross-depth RMSE plus
0.3 times reconstruction RMSE. LPIPS weights are stored below
`D:\edof\torch-cache` by the Windows scripts.

The paper's 6000-point wave grid cannot run on an RTX 5060 8 GB card. The
5x5/512 configuration keeps cached fields in host memory and computes one local
field patch per training sample so the disclosed model can run on this GPU.

## RTX 5060 practical fine-tune

`windows_practical_finetune.yaml` starts a new 50-epoch network fine-tune from
the best checkpoint in `windows_optimized`. It keeps the 512-point wave grid,
but traces every one of the 40x40 field positions instead of interpolating the
5x5 PSF map. The approximately 230 MB fixed PSF cache is saved every 100 field
positions and resumes after interruption.

The practical loss is reconstruction MSE plus VGG16 perceptual loss and a
smaller cross-depth consistency term. This targets visible reconstruction
quality on the 8 GB card and is intentionally not described as the paper's
strict loss.

Run the short two-by-two field and one-batch check first:

```powershell
.\.venv-edof\Scripts\python.exe -m edof_reproduction `
  --config configs\edof_reproduction\windows_practical_memory.yaml `
  --output workspace\edof_reproduction\windows_practical_memory
```

Start the full detached run with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\windows_edof_start.ps1 `
  -TaskName EDOFPracticalFinetune `
  -Config configs\edof_reproduction\windows_practical_finetune.yaml `
  -OutputName windows_practical_finetune
```

Evaluate an existing checkpoint without training:

```powershell
python -m edof_reproduction `
  --config configs\edof_reproduction\windows_baseline_eval.yaml `
  --resume workspace\edof_reproduction\windows_full_actual\checkpoints\latest.pt `
  --evaluate-only `
  --output workspace\edof_reproduction\windows_full_actual\baseline_validation
```

## Recommended exact-PSF sequence

The recommended sequence isolates the exact 40x40 PSF change before spending
another full run:

1. evaluate `windows_optimized/checkpoints/best.pt` on the existing exact PSF
   cache without training;
2. train 15 epochs with perceptual weights 0.02 and 0.05 from the same
   checkpoint and seed;
3. select a Pareto checkpoint with at most 0.1 dB PSNR loss and at least 0.02
   1-LPIPS gain when that guard is achievable;
4. train only the selected weight for 50 epochs;
5. evaluate both the PSNR-best and 1-LPIPS-best full-run checkpoints.

Start the resumable sequence as a Windows scheduled task:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\windows_edof_sequence_start.ps1
```

The task writes its current step to
`workspace\edof_reproduction\windows_recommended_sequence\sequence_state.json`.
The final comparison, selected checkpoint, and paper gaps are written to
`windows_recommended_sequence\final_summary.json`. Existing completed steps are
skipped and interrupted training resumes from `checkpoints\latest.pt`.

## Tests

```bash
python -m pytest -q
```
