"""Evaluate all retained checkpoints twice on the complete 40x40 field map."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path("configs/edof_reproduction/windows_strict_full_fov_eval.yaml")
METRIC_KEYS = ("psnr", "ssim", "lpips", "one_minus_lpips")


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


def completed(path: Path) -> bool:
    return path.exists() and read_json(path).get("status") == "completed"


def metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in METRIC_KEYS:
        if left.get(key) is not None and right.get(key) is not None:
            result[key] = abs(float(left[key]) - float(right[key]))
    return result


class FullFieldEvaluator:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.workspace = self.project_root / "workspace" / "edof_reproduction"
        self.output = self.workspace / "windows_strict_full_fov_evaluation"
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output / "state.json"
        self.state: dict[str, Any] = {
            "status": "running",
            "started_at": now(),
            "updated_at": now(),
            "current_step": "initializing",
            "steps": {},
        }
        if self.state_path.exists():
            previous = read_json(self.state_path)
            self.state["started_at"] = previous.get("started_at", self.state["started_at"])
            self.state["steps"] = previous.get("steps", {})
        self._save_state()

    def _save_state(self) -> None:
        self.state["updated_at"] = now()
        write_json(self.state_path, self.state)

    def _set_step(self, name: str, status: str, **details: Any) -> None:
        self.state["current_step"] = name
        row = self.state["steps"].setdefault(name, {})
        row.update({"status": status, "updated_at": now(), **details})
        self._save_state()
        print(
            json.dumps(
                {"strict_full_field": {"step": name, "status": status, **details}},
                ensure_ascii=False,
            ),
            flush=True,
        )

    def _evaluate(
        self,
        name: str,
        checkpoint: Path,
        repeat: int,
    ) -> dict[str, Any]:
        step = f"{name}_repeat_{repeat}"
        output = self.output / step
        summary_path = output / "validation_summary.json"
        if completed(summary_path):
            self._set_step(step, "completed", skipped=True, output=str(output))
            return read_json(summary_path)
        arguments = [
            sys.executable,
            "-u",
            "-m",
            "edof_reproduction",
            "--config",
            str(CONFIG),
            "--output",
            str(output),
            "--resume",
            str(checkpoint),
            "--evaluate-only",
        ]
        self._set_step(step, "running", command=arguments, output=str(output))
        result = subprocess.run(arguments, cwd=self.project_root, check=False)
        if result.returncode != 0:
            self._set_step(step, "failed", exit_code=result.returncode)
            raise RuntimeError(f"{step} failed with exit code {result.returncode}")
        if not completed(summary_path):
            self._set_step(step, "failed", missing=str(summary_path))
            raise RuntimeError(f"{step} did not write a completed summary")
        self._set_step(step, "completed", exit_code=0, output=str(output))
        return read_json(summary_path)

    def run(self) -> dict[str, Any]:
        checkpoints = {
            "original": self.workspace / "windows_optimized" / "checkpoints" / "best.pt",
            "epoch45_psnr": (
                self.workspace
                / "windows_practical_p002_full"
                / "checkpoints"
                / "best.pt"
            ),
            "epoch25_perceptual": (
                self.workspace
                / "windows_practical_p002_full"
                / "checkpoints"
                / "best_perceptual.pt"
            ),
        }
        for name, checkpoint in checkpoints.items():
            if not checkpoint.exists():
                raise FileNotFoundError(f"required checkpoint is missing: {name}: {checkpoint}")

        evaluations: dict[str, Any] = {}
        repeatability: dict[str, Any] = {}
        for name, checkpoint in checkpoints.items():
            first = self._evaluate(name, checkpoint, 1)
            second = self._evaluate(name, checkpoint, 2)
            first_validation = first["validation"]
            second_validation = second["validation"]
            mean_delta = metric_delta(
                first_validation["mean"],
                second_validation["mean"],
            )
            depth_deltas = [
                metric_delta(left, right)
                for left, right in zip(
                    first_validation["depth_metrics"],
                    second_validation["depth_metrics"],
                )
            ]
            maximum_delta = max(
                [*mean_delta.values(), *(value for row in depth_deltas for value in row.values())],
                default=0.0,
            )
            repeatability[name] = {
                "mean_delta": mean_delta,
                "depth_deltas": depth_deltas,
                "maximum_absolute_delta": maximum_delta,
                "threshold": 0.02,
                "passed": maximum_delta <= 0.02,
            }
            evaluations[name] = {
                "checkpoint": str(checkpoint),
                "checkpoint_epoch": first["checkpoint_epoch"],
                "validation": first_validation,
            }

        passed = all(row["passed"] for row in repeatability.values())
        summary = {
            "status": "completed" if passed else "failed_repeatability",
            "completed_at": now(),
            "protocol": {
                "sensor_crop": [1000, 1000],
                "field_grid": [40, 40],
                "field_coverage": "all_1600_cells",
                "validation_images": 100,
                "repeats": 2,
                "repeatability_threshold": 0.02,
            },
            "evaluations": evaluations,
            "repeatability": repeatability,
        }
        write_json(self.output / "summary.json", summary)
        self.state["status"] = summary["status"]
        self.state["current_step"] = "completed" if passed else "repeatability_failed"
        self.state["completed_at"] = now()
        self._save_state()
        print(json.dumps({"strict_full_field_summary": summary}, ensure_ascii=False), flush=True)
        if not passed:
            raise RuntimeError("full-field evaluation did not satisfy repeatability threshold")
        return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    FullFieldEvaluator(args.project_root).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
