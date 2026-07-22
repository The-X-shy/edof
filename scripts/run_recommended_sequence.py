"""Run the recommended exact-baseline, short-ablation, and full EDoF sequence."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PROJECT_ROOT))

from edof_reproduction.selection import (
    ValidationCandidate,
    best_perceptual_candidate,
    candidate_from_validation_row,
    select_balanced_candidate,
)


CONFIG_ROOT = Path("configs/edof_reproduction")
BASELINE_CONFIG = CONFIG_ROOT / "windows_exact_baseline_eval.yaml"
SHORT_CONFIGS = {
    0.02: CONFIG_ROOT / "windows_practical_p002_short.yaml",
    0.05: CONFIG_ROOT / "windows_practical_p005_short.yaml",
}
FULL_CONFIGS = {
    0.02: CONFIG_ROOT / "windows_practical_p002_full.yaml",
    0.05: CONFIG_ROOT / "windows_practical_p005_full.yaml",
}
SHORT_RUNS = {0.02: "windows_practical_p002_short", 0.05: "windows_practical_p005_short"}
FULL_RUNS = {0.02: "windows_practical_p002_full", 0.05: "windows_practical_p005_full"}
PAPER_TARGETS = [
    {"depth_mm": -200.0, "psnr": 27.5, "ssim": 0.821, "one_minus_lpips": 0.782},
    {"depth_mm": -300.0, "psnr": 28.9, "ssim": 0.869, "one_minus_lpips": 0.842},
    {"depth_mm": -10000.0, "psnr": 27.4, "ssim": 0.818, "one_minus_lpips": 0.787},
]


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


class SequenceRunner:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.workspace = self.project_root / "workspace" / "edof_reproduction"
        self.sequence_output = self.workspace / "windows_recommended_sequence"
        self.sequence_output.mkdir(parents=True, exist_ok=True)
        self.state_path = self.sequence_output / "sequence_state.json"
        self.state: dict[str, Any] = {
            "status": "running",
            "current_step": "initializing",
            "started_at": now(),
            "updated_at": now(),
            "steps": {},
        }
        if self.state_path.exists():
            previous = read_json(self.state_path)
            self.state["started_at"] = previous.get("started_at", self.state["started_at"])
            self.state["steps"] = previous.get("steps", {})
        self._write_state()

    def _write_state(self) -> None:
        self.state["updated_at"] = now()
        write_json(self.state_path, self.state)

    def _set_step(self, name: str, status: str, **details: Any) -> None:
        self.state["current_step"] = name
        row = self.state["steps"].setdefault(name, {})
        row.update({"status": status, "updated_at": now(), **details})
        self._write_state()
        print(json.dumps({"sequence": {"step": name, "status": status, **details}}, ensure_ascii=False), flush=True)

    def _run_process(
        self,
        step: str,
        arguments: list[str],
        completion_file: Path,
    ) -> dict[str, Any]:
        if completed(completion_file):
            self._set_step(step, "completed", skipped=True, completion_file=str(completion_file))
            return read_json(completion_file)
        self._set_step(step, "running", command=arguments, completion_file=str(completion_file))
        result = subprocess.run(arguments, cwd=self.project_root, check=False)
        if result.returncode != 0:
            self._set_step(step, "failed", exit_code=result.returncode)
            raise RuntimeError(f"{step} failed with exit code {result.returncode}")
        if not completed(completion_file):
            self._set_step(step, "failed", exit_code=result.returncode, missing=str(completion_file))
            raise RuntimeError(f"{step} did not write a completed summary")
        self._set_step(step, "completed", exit_code=result.returncode, completion_file=str(completion_file))
        return read_json(completion_file)

    def _module_arguments(self, config: Path, output: Path) -> list[str]:
        return [
            sys.executable,
            "-u",
            "-m",
            "edof_reproduction",
            "--config",
            str(config),
            "--output",
            str(output),
        ]

    def evaluate_checkpoint(
        self,
        step: str,
        checkpoint: Path,
        output: Path,
    ) -> dict[str, Any]:
        arguments = self._module_arguments(BASELINE_CONFIG, output)
        arguments.extend(["--resume", str(checkpoint), "--evaluate-only"])
        return self._run_process(step, arguments, output / "validation_summary.json")

    def train(self, step: str, config: Path, output: Path) -> dict[str, Any]:
        arguments = self._module_arguments(config, output)
        latest = output / "checkpoints" / "latest.pt"
        if latest.exists():
            arguments.extend(["--resume", str(latest)])
        return self._run_process(step, arguments, output / "summary.json")

    def validation_candidates(
        self,
        run_name: str,
        weight: float,
        output: Path,
    ) -> list[ValidationCandidate]:
        log_path = output / "validation_log.jsonl"
        rows = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        candidates = []
        for row in rows:
            epoch = int(row["epoch"])
            checkpoint = output / "checkpoints" / f"epoch_{epoch:03d}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"validation checkpoint is missing: {checkpoint}")
            candidates.append(
                candidate_from_validation_row(
                    row,
                    run_name=run_name,
                    weight=weight,
                    checkpoint=str(checkpoint),
                )
            )
        if not candidates:
            raise RuntimeError(f"no validation rows found for {run_name}")
        return candidates

    def name_perceptual_checkpoint(
        self,
        run_name: str,
        weight: float,
        output: Path,
    ) -> ValidationCandidate:
        candidates = self.validation_candidates(run_name, weight, output)
        best = best_perceptual_candidate(candidates)
        source = Path(best.checkpoint)
        target = output / "checkpoints" / "best_perceptual.pt"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        write_json(
            output / "best_perceptual.json",
            {"status": "completed", "selected": best.as_dict(), "checkpoint": str(target)},
        )
        return ValidationCandidate(
            run_name=best.run_name,
            weight=best.weight,
            epoch=best.epoch,
            psnr=best.psnr,
            ssim=best.ssim,
            one_minus_lpips=best.one_minus_lpips,
            checkpoint=str(target),
        )

    def run(self) -> dict[str, Any]:
        old_best = self.workspace / "windows_optimized" / "checkpoints" / "best.pt"
        if not old_best.exists():
            raise FileNotFoundError(f"required epoch-95 baseline checkpoint is missing: {old_best}")

        baseline_output = self.workspace / "windows_exact_baseline_eval"
        baseline_summary = self.evaluate_checkpoint(
            "01_exact_baseline",
            old_best,
            baseline_output,
        )
        baseline_mean = baseline_summary["validation"]["mean"]

        short_summaries: dict[str, Any] = {}
        short_candidates: list[ValidationCandidate] = []
        for index, weight in enumerate((0.02, 0.05), start=2):
            run_name = SHORT_RUNS[weight]
            output = self.workspace / run_name
            summary = self.train(
                f"0{index}_short_p{int(weight * 1000):03d}",
                SHORT_CONFIGS[weight],
                output,
            )
            perceptual_best = self.name_perceptual_checkpoint(run_name, weight, output)
            short_summaries[run_name] = {
                "summary": summary,
                "best_perceptual": perceptual_best.as_dict(),
            }
            short_candidates.extend(self.validation_candidates(run_name, weight, output))

        short_decision = select_balanced_candidate(baseline_mean, short_candidates)
        selected_weight = float(short_decision["selected"]["weight"])
        if not any(math.isclose(selected_weight, candidate) for candidate in FULL_CONFIGS):
            raise RuntimeError(f"unsupported selected perceptual weight: {selected_weight}")
        selected_weight = min(FULL_CONFIGS, key=lambda item: abs(item - selected_weight))
        current_summary_path = self.workspace / "windows_practical_finetune" / "summary.json"
        optimized_summary_path = self.workspace / "windows_optimized" / "summary.json"
        short_decision_payload = {
            "status": "completed",
            "baseline": baseline_mean,
            "decision": short_decision,
            "short_runs": short_summaries,
            "current_p010": read_json(current_summary_path) if current_summary_path.exists() else None,
        }
        write_json(self.sequence_output / "short_decision.json", short_decision_payload)
        self._set_step(
            "04_select_short_winner",
            "completed",
            selected_weight=selected_weight,
            reason=short_decision["reason"],
        )

        full_run_name = FULL_RUNS[selected_weight]
        full_output = self.workspace / full_run_name
        full_summary = self.train(
            "05_full_training",
            FULL_CONFIGS[selected_weight],
            full_output,
        )
        perceptual_best = self.name_perceptual_checkpoint(
            full_run_name,
            selected_weight,
            full_output,
        )

        psnr_evaluation = self.evaluate_checkpoint(
            "06_evaluate_psnr_best",
            full_output / "checkpoints" / "best.pt",
            full_output / "evaluation_psnr_best",
        )
        perceptual_evaluation = self.evaluate_checkpoint(
            "07_evaluate_perceptual_best",
            full_output / "checkpoints" / "best_perceptual.pt",
            full_output / "evaluation_perceptual_best",
        )
        final_candidates = [
            ValidationCandidate(
                run_name="psnr_best",
                weight=selected_weight,
                epoch=int(psnr_evaluation["checkpoint_epoch"]),
                checkpoint=str(full_output / "checkpoints" / "best.pt"),
                **{
                    key: float(psnr_evaluation["validation"]["mean"][key])
                    for key in ("psnr", "ssim", "one_minus_lpips")
                },
            ),
            ValidationCandidate(
                run_name="perceptual_best",
                weight=selected_weight,
                epoch=int(perceptual_evaluation["checkpoint_epoch"]),
                checkpoint=str(full_output / "checkpoints" / "best_perceptual.pt"),
                **{
                    key: float(perceptual_evaluation["validation"]["mean"][key])
                    for key in ("psnr", "ssim", "one_minus_lpips")
                },
            ),
        ]
        final_decision = select_balanced_candidate(baseline_mean, final_candidates)
        paper_mean = {
            metric: sum(float(row[metric]) for row in PAPER_TARGETS) / len(PAPER_TARGETS)
            for metric in ("psnr", "ssim", "one_minus_lpips")
        }
        selected_final = final_decision["selected"]
        final_summary = {
            "status": "completed",
            "completed_at": now(),
            "baseline_evaluation": baseline_summary,
            "short_decision": short_decision_payload,
            "selected_weight": selected_weight,
            "full_run": full_summary,
            "full_best_perceptual": perceptual_best.as_dict(),
            "full_evaluations": {
                "psnr_best": psnr_evaluation,
                "perceptual_best": perceptual_evaluation,
            },
            "final_decision": final_decision,
            "selected_checkpoint": selected_final["checkpoint"],
            "selected_metrics": {
                key: selected_final[key] for key in ("psnr", "ssim", "one_minus_lpips")
            },
            "paper_targets": PAPER_TARGETS,
            "paper_mean": paper_mean,
            "gap_to_paper": {
                key: float(selected_final[key]) - paper_mean[key]
                for key in ("psnr", "ssim", "one_minus_lpips")
            },
            "previous_optimized": (
                read_json(optimized_summary_path) if optimized_summary_path.exists() else None
            ),
            "current_p010": (
                read_json(current_summary_path) if current_summary_path.exists() else None
            ),
        }
        final_path = self.sequence_output / "final_summary.json"
        write_json(final_path, final_summary)
        self.state.update(
            {
                "status": "completed",
                "current_step": "completed",
                "completed_at": now(),
                "final_summary": str(final_path),
                "selected_checkpoint": selected_final["checkpoint"],
            }
        )
        self._write_state()
        print(json.dumps({"sequence_completed": final_summary}, ensure_ascii=False), flush=True)
        return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=SCRIPT_PROJECT_ROOT,
    )
    arguments = parser.parse_args()
    runner = SequenceRunner(arguments.project_root)
    try:
        result = runner.run()
    except Exception as error:
        runner.state.update(
            {
                "status": "failed",
                "failed_at": now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        runner._write_state()
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
