from __future__ import annotations

from edof_reproduction.selection import (
    ValidationCandidate,
    best_perceptual_candidate,
    pareto_frontier,
    select_balanced_candidate,
)


def candidate(
    name: str,
    weight: float,
    epoch: int,
    psnr: float,
    ssim: float,
    perceptual: float,
) -> ValidationCandidate:
    return ValidationCandidate(
        run_name=name,
        weight=weight,
        epoch=epoch,
        psnr=psnr,
        ssim=ssim,
        one_minus_lpips=perceptual,
        checkpoint=f"{name}/epoch_{epoch:03d}.pt",
    )


def test_balanced_selection_prefers_perceptual_gain_inside_psnr_guard() -> None:
    baseline = {"psnr": 17.2, "ssim": 0.39, "one_minus_lpips": 0.22}
    p002 = candidate("p002", 0.02, 10, 17.18, 0.388, 0.27)
    p005 = candidate("p005", 0.05, 15, 17.11, 0.382, 0.31)
    decision = select_balanced_candidate(baseline, [p002, p005])
    assert decision["reason"] == "quality_guard"
    assert decision["selected"]["run_name"] == "p005"
    assert decision["eligible_count"] == 2


def test_balanced_selection_uses_conservative_pareto_fallback() -> None:
    baseline = {"psnr": 17.2, "ssim": 0.39, "one_minus_lpips": 0.22}
    high_psnr = candidate("p002", 0.02, 10, 17.07, 0.385, 0.25)
    high_perceptual = candidate("p005", 0.05, 15, 16.9, 0.38, 0.34)
    dominated = candidate("dominated", 0.1, 15, 16.8, 0.37, 0.24)
    decision = select_balanced_candidate(baseline, [high_psnr, high_perceptual, dominated])
    assert decision["reason"] == "conservative_pareto_fallback"
    assert decision["selected"]["run_name"] == "p002"
    assert {item.run_name for item in pareto_frontier([high_psnr, high_perceptual, dominated])} == {
        "p002",
        "p005",
    }


def test_best_perceptual_candidate_breaks_tie_with_psnr() -> None:
    lower_psnr = candidate("a", 0.02, 5, 17.0, 0.38, 0.3)
    higher_psnr = candidate("b", 0.05, 10, 17.1, 0.37, 0.3)
    assert best_perceptual_candidate([lower_psnr, higher_psnr]) == higher_psnr
