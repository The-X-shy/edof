"""Execute the approved evaluation, smoothing, ablation, and DOE gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_step(
    project_root: Path,
    state: dict[str, Any],
    state_path: Path,
    name: str,
    script: str,
    summary_path: Path,
) -> dict[str, Any]:
    state["current_step"] = name
    state["updated_at"] = now()
    state["steps"][name] = {
        "status": "running",
        "updated_at": now(),
        "summary": str(summary_path),
    }
    write_json(state_path, state)
    if not summary_path.exists():
        process = subprocess.run(
            [
                sys.executable,
                "-u",
                str(project_root / "scripts" / script),
                "--project-root",
                str(project_root),
            ],
            cwd=project_root,
            check=False,
        )
        if process.returncode != 0:
            state["status"] = "failed"
            state["steps"][name].update(
                {
                    "status": "failed",
                    "exit_code": process.returncode,
                    "updated_at": now(),
                }
            )
            write_json(state_path, state)
            raise RuntimeError(f"{name} failed with exit code {process.returncode}")
    if not summary_path.exists():
        raise RuntimeError(f"{name} did not write its summary")
    summary = read_json(summary_path)
    state["steps"][name] = {
        "status": "completed",
        "updated_at": now(),
        "summary": str(summary_path),
        "result_status": summary.get("status"),
    }
    write_json(state_path, state)
    return summary


def run(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    workspace = project_root / "workspace" / "edof_reproduction"
    output = workspace / "windows_ordered_optimization"
    state_path = output / "state.json"
    summary_path = output / "summary.json"
    output.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        return read_json(summary_path)
    state = {
        "status": "running",
        "started_at": now(),
        "updated_at": now(),
        "current_step": "initializing",
        "steps": {},
    }
    write_json(state_path, state)

    unified = run_step(
        project_root,
        state,
        state_path,
        "01_unified_evaluation",
        "run_strict_unified_spatial_evaluation.py",
        workspace / "windows_strict_unified_evaluation" / "summary.json",
    )
    smooth = run_step(
        project_root,
        state,
        state_path,
        "02_smooth_psf_gate",
        "run_smooth_psf_gate.py",
        workspace / "windows_smooth_psf_gate" / "summary.json",
    )
    if not smooth.get("passed"):
        summary = {
            "status": "stopped_at_smooth_psf_gate",
            "completed_at": now(),
            "unified_evaluation": unified,
            "smooth_psf_gate": smooth,
        }
        write_json(summary_path, summary)
        state.update(
            {
                "status": summary["status"],
                "current_step": "smooth_psf_gate_failed",
                "updated_at": now(),
                "completed_at": now(),
            }
        )
        write_json(state_path, state)
        return summary

    ablation = run_step(
        project_root,
        state,
        state_path,
        "03_loss_ablation",
        "run_smooth_loss_ablation.py",
        workspace / "windows_smooth_loss_ablation" / "summary.json",
    )
    doe = None
    if ablation.get("doe_optimization_required"):
        doe = run_step(
            project_root,
            state,
            state_path,
            "04_doe_optimization",
            "run_doe_raw_optimization.py",
            workspace / "windows_doe_raw_optimization" / "summary.json",
        )
    summary = {
        "status": "completed",
        "completed_at": now(),
        "unified_evaluation": unified,
        "smooth_psf_gate": smooth,
        "loss_ablation": ablation,
        "doe_optimization": doe,
    }
    write_json(summary_path, summary)
    state.update(
        {
            "status": "completed",
            "current_step": "completed",
            "updated_at": now(),
            "completed_at": now(),
        }
    )
    write_json(state_path, state)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    run(args.project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
