"""Decision rules for the staged practical EDoF fine-tune sequence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ValidationCandidate:
    """One validation checkpoint that can participate in a Pareto decision."""

    run_name: str
    weight: float
    epoch: int
    psnr: float
    ssim: float
    one_minus_lpips: float
    checkpoint: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def candidate_from_validation_row(
    row: Mapping[str, Any],
    *,
    run_name: str,
    weight: float,
    checkpoint: str,
) -> ValidationCandidate:
    mean = row["mean"]
    return ValidationCandidate(
        run_name=run_name,
        weight=float(weight),
        epoch=int(row["epoch"]),
        psnr=float(mean["psnr"]),
        ssim=float(mean["ssim"]),
        one_minus_lpips=float(mean["one_minus_lpips"]),
        checkpoint=checkpoint,
    )


def dominates(left: ValidationCandidate, right: ValidationCandidate) -> bool:
    """Return true when ``left`` is no worse on every metric and better on one."""

    no_worse = (
        left.psnr >= right.psnr
        and left.ssim >= right.ssim
        and left.one_minus_lpips >= right.one_minus_lpips
    )
    strictly_better = (
        left.psnr > right.psnr
        or left.ssim > right.ssim
        or left.one_minus_lpips > right.one_minus_lpips
    )
    return no_worse and strictly_better


def pareto_frontier(candidates: Iterable[ValidationCandidate]) -> list[ValidationCandidate]:
    rows = list(candidates)
    return [
        candidate
        for candidate in rows
        if not any(
            dominates(other, candidate)
            for other in rows
            if other is not candidate
        )
    ]


def best_perceptual_candidate(
    candidates: Iterable[ValidationCandidate],
) -> ValidationCandidate:
    rows = list(candidates)
    if not rows:
        raise ValueError("at least one validation candidate is required")
    return max(rows, key=lambda item: (item.one_minus_lpips, item.psnr, item.ssim))


def select_balanced_candidate(
    baseline: Mapping[str, Any],
    candidates: Iterable[ValidationCandidate],
    *,
    max_psnr_drop: float = 0.1,
    min_perceptual_gain: float = 0.02,
) -> dict[str, Any]:
    """Select a perceptual improvement without accepting a large PSNR regression.

    A candidate that satisfies both guards is preferred by 1-LPIPS and then by
    PSNR/SSIM.  If no candidate satisfies the guards, the choice is restricted
    to the Pareto frontier and falls back to the highest-PSNR checkpoint.
    """

    rows = list(candidates)
    if not rows:
        raise ValueError("at least one validation candidate is required")
    baseline_psnr = float(baseline["psnr"])
    baseline_ssim = float(baseline["ssim"])
    baseline_perceptual = float(baseline["one_minus_lpips"])
    eligible = [
        item
        for item in rows
        if item.psnr >= baseline_psnr - max_psnr_drop
        and item.one_minus_lpips >= baseline_perceptual + min_perceptual_gain
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda item: (
                item.one_minus_lpips,
                item.psnr,
                item.ssim,
                -item.weight,
            ),
        )
        reason = "quality_guard"
    else:
        frontier = pareto_frontier(rows)
        selected = max(
            frontier,
            key=lambda item: (
                item.psnr,
                item.ssim,
                item.one_minus_lpips,
                -item.weight,
            ),
        )
        reason = "conservative_pareto_fallback"
    return {
        "selected": selected.as_dict(),
        "reason": reason,
        "guard": {
            "baseline_psnr": baseline_psnr,
            "baseline_ssim": baseline_ssim,
            "baseline_one_minus_lpips": baseline_perceptual,
            "max_psnr_drop": max_psnr_drop,
            "min_perceptual_gain": min_perceptual_gain,
        },
        "deltas": {
            "psnr": selected.psnr - baseline_psnr,
            "ssim": selected.ssim - baseline_ssim,
            "one_minus_lpips": selected.one_minus_lpips - baseline_perceptual,
        },
        "eligible_count": len(eligible),
        "pareto_frontier": [item.as_dict() for item in pareto_frontier(rows)],
    }
