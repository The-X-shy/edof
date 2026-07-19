import math

import pytest

torch = pytest.importorskip("torch")

from edof_reproduction.poly1d import Poly1DDOE


def test_poly1d_matches_public_deeplens_formula_and_has_six_parameters():
    coefficients = torch.tensor([0.2, -0.3, 0.4, -0.5, 0.6, -0.7])
    doe = Poly1DDOE(doe_radius=2.0, coefficients=coefficients)
    x = torch.tensor([1.0, -0.5])
    y = torch.tensor([0.5, 1.5])

    x_norm = x / 2.0
    y_norm = y / 2.0
    radius_squared = x_norm.square() + y_norm.square()
    expected = (
        coefficients[0] * radius_squared
        + coefficients[2] * radius_squared.pow(2)
        + coefficients[4] * radius_squared.pow(3)
        + coefficients[1] * (x_norm.pow(3) + y_norm.pow(3))
        + coefficients[3] * (x_norm.pow(5) + y_norm.pow(5))
        + coefficients[5] * (x_norm.pow(7) + y_norm.pow(7))
    )

    assert torch.allclose(doe.raw_phase(x, y), expected)
    assert [name for name, _ in doe.named_parameters()] == [
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
    ]
    assert all(parameter.requires_grad for parameter in doe.parameters())


def test_continuous_phase_wrap_and_wavelength_index_scaling():
    doe = Poly1DDOE(
        doe_radius=1.0,
        coefficients=[8.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        design_wavelength_um=0.55,
        design_refractive_index=1.46,
    )
    raw = doe.raw_phase(torch.tensor(1.0), torch.tensor(0.0))
    scale = (0.55 / 0.50) * ((1.50 - 1.0) / (1.46 - 1.0))

    unwrapped = doe.continuous_phase(
        torch.tensor(1.0),
        torch.tensor(0.0),
        wavelength_um=0.50,
        refractive_index=1.50,
        wrap=False,
    )
    wrapped = doe.continuous_phase(
        torch.tensor(1.0),
        torch.tensor(0.0),
        wavelength_um=0.50,
        refractive_index=1.50,
    )
    source_compatible = doe.continuous_phase(
        torch.tensor(1.0),
        torch.tensor(0.0),
        wavelength_um=0.50,
        refractive_index=1.50,
        wrap_before_scaling=True,
    )

    assert unwrapped.item() == pytest.approx(raw.item() * scale)
    assert wrapped.item() == pytest.approx((raw.item() * scale) % math.tau)
    assert source_compatible.item() == pytest.approx((raw.item() % math.tau) * scale)


def test_sixteen_level_quantization_has_discrete_forward_and_ste_gradient():
    doe = Poly1DDOE(doe_radius=1.0, coefficients=torch.zeros(6))
    step = math.tau / 16
    phase = torch.tensor(
        [-0.1, 0.49 * step, 1.51 * step, math.tau - 0.1 * step],
        requires_grad=True,
    )

    quantized = doe.quantize_phase(phase)
    expected_indices = torch.tensor([0.0, 0.0, 2.0, 0.0])
    assert torch.allclose(quantized, expected_indices * step, atol=1e-7)
    assert torch.all(quantized >= 0)
    assert torch.all(quantized < math.tau)

    quantized.sum().backward()
    assert torch.allclose(phase.grad, torch.ones_like(phase))


def test_quantized_phase_backpropagates_to_all_coefficients():
    doe = Poly1DDOE(
        doe_radius=1.0,
        coefficients=[0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
    )
    x = torch.tensor([0.2, 0.4, 0.6])
    y = torch.tensor([0.3, 0.5, 0.7])

    doe.quantized_phase(x, y).sum().backward()

    assert all(parameter.grad is not None for parameter in doe.parameters())
    assert all(torch.isfinite(parameter.grad) for parameter in doe.parameters())


def test_state_and_checkpoint_round_trip(tmp_path):
    doe = Poly1DDOE(
        doe_radius=3.5,
        coefficients=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float64),
        design_wavelength_um=0.532,
        design_refractive_index=1.47,
        quantization_levels=16,
    )
    x = torch.linspace(-2.0, 2.0, 7, dtype=torch.float64)
    y = torch.linspace(1.0, -1.0, 7, dtype=torch.float64)
    expected = doe.quantized_phase(x, y, straight_through=False)

    restored_from_state = Poly1DDOE.from_state(doe.export_state())
    assert restored_from_state.coefficients.dtype == torch.float64
    assert torch.equal(
        restored_from_state.quantized_phase(x, y, straight_through=False), expected
    )

    checkpoint = tmp_path / "poly1d.pt"
    doe.save_checkpoint(checkpoint)
    restored_from_file = Poly1DDOE.from_checkpoint(checkpoint)
    assert restored_from_file.quantization_levels == 16
    assert restored_from_file.doe_radius.item() == pytest.approx(3.5)
    assert torch.equal(
        restored_from_file.quantized_phase(x, y, straight_through=False), expected
    )

    existing = Poly1DDOE(doe_radius=1.0, coefficients=torch.zeros(6))
    existing.load_exported_state(doe.export_state())
    assert torch.equal(existing.coefficients, doe.coefficients.float())
    assert existing.doe_radius.item() == pytest.approx(3.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"doe_radius": 0.0},
        {"doe_radius": 1.0, "design_wavelength_um": 0.0},
        {"doe_radius": 1.0, "design_refractive_index": 1.0},
        {"doe_radius": 1.0, "quantization_levels": 1},
    ],
)
def test_invalid_physical_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Poly1DDOE(**kwargs)
