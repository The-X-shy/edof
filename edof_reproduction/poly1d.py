"""Pure PyTorch implementation of the historical DeepLens Poly1D DOE.

The polynomial follows the public DeepLens implementation: even orders are
radially symmetric and odd orders are separable in the normalized x/y axes.
This module intentionally has no dependency on DeepLens.
"""

from __future__ import annotations

import math
from os import PathLike
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import torch
from torch import Tensor, nn


PathType = Union[str, PathLike]
_ORDERS = tuple(range(2, 8))
_CHECKPOINT_FORMAT = "edof_reproduction.poly1d_doe"
_CHECKPOINT_VERSION = 1


class Poly1DDOE(nn.Module):
    """Trainable Poly1D phase mask used by the public EDoF example.

    Args:
        doe_radius: DOE radius in the same units as ``x`` and ``y``.
        coefficients: Values for ``a2`` through ``a7``. When omitted, the
            historical initialization, uniform in ``[0, 1e-3)``, is used.
        design_wavelength_um: Wavelength at which the coefficients are defined.
        design_refractive_index: DOE substrate index at the design wavelength.
        quantization_levels: Number of uniformly spaced phase fabrication levels.
        dtype: Parameter dtype. Defaults to PyTorch's default floating dtype.
        device: Parameter device.

    Coordinates are normalized by ``doe_radius``. No circular aperture is
    applied; the caller remains responsible for masking points outside the DOE.
    """

    def __init__(
        self,
        doe_radius: float,
        coefficients: Optional[Union[Sequence[float], Tensor]] = None,
        *,
        design_wavelength_um: float = 0.55,
        design_refractive_index: float = 1.4601,
        quantization_levels: int = 16,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self._validate_config(
            doe_radius=doe_radius,
            design_wavelength_um=design_wavelength_um,
            design_refractive_index=design_refractive_index,
            quantization_levels=quantization_levels,
        )

        factory_kwargs = {"dtype": dtype, "device": device}
        if coefficients is None:
            values = torch.rand(6, **factory_kwargs) * 1e-3
        else:
            values = torch.as_tensor(coefficients, **factory_kwargs)
            if not values.is_floating_point():
                values = values.to(dtype=dtype or torch.get_default_dtype())
        if values.shape != (6,):
            raise ValueError("coefficients must contain exactly a2 through a7")

        for index, order in enumerate(_ORDERS):
            self.register_parameter(
                f"a{order}", nn.Parameter(values[index].detach().clone())
            )

        buffer_kwargs = {"dtype": values.dtype, "device": values.device}
        self.register_buffer("doe_radius", torch.tensor(doe_radius, **buffer_kwargs))
        self.register_buffer(
            "design_wavelength_um",
            torch.tensor(design_wavelength_um, **buffer_kwargs),
        )
        self.register_buffer(
            "design_refractive_index",
            torch.tensor(design_refractive_index, **buffer_kwargs),
        )
        self.quantization_levels = int(quantization_levels)

    @staticmethod
    def _validate_config(
        *,
        doe_radius: float,
        design_wavelength_um: float,
        design_refractive_index: float,
        quantization_levels: int,
    ) -> None:
        if doe_radius <= 0:
            raise ValueError("doe_radius must be positive")
        if design_wavelength_um <= 0:
            raise ValueError("design_wavelength_um must be positive")
        if design_refractive_index <= 1:
            raise ValueError("design_refractive_index must be greater than one")
        if isinstance(quantization_levels, bool) or quantization_levels < 2:
            raise ValueError("quantization_levels must be an integer of at least two")
        if int(quantization_levels) != quantization_levels:
            raise ValueError("quantization_levels must be an integer of at least two")

    @property
    def coefficients(self) -> Tensor:
        """Return ``[a2, ..., a7]`` as a differentiable stacked tensor."""

        return torch.stack([getattr(self, f"a{order}") for order in _ORDERS])

    def set_coefficients(self, coefficients: Union[Sequence[float], Tensor]) -> None:
        """Copy six coefficient values into the existing trainable parameters."""

        values = torch.as_tensor(
            coefficients, dtype=self.a2.dtype, device=self.a2.device
        )
        if values.shape != (6,):
            raise ValueError("coefficients must contain exactly a2 through a7")
        with torch.no_grad():
            for index, order in enumerate(_ORDERS):
                getattr(self, f"a{order}").copy_(values[index])

    def normalized_coordinates(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Broadcast ``x`` and ``y`` and normalize them by the DOE radius."""

        x_tensor = torch.as_tensor(x, dtype=self.a2.dtype, device=self.a2.device)
        y_tensor = torch.as_tensor(y, dtype=self.a2.dtype, device=self.a2.device)
        x_tensor, y_tensor = torch.broadcast_tensors(x_tensor, y_tensor)
        return x_tensor / self.doe_radius, y_tensor / self.doe_radius

    def raw_phase(self, x: Tensor, y: Tensor) -> Tensor:
        """Evaluate the unwrapped Poly1D phase at the design wavelength.

        The exact public-source polynomial is

        ``a2*r^2 + a4*r^4 + a6*r^6``
        ``+ a3*(x^3+y^3) + a5*(x^5+y^5) + a7*(x^7+y^7)``,

        where all coordinates are normalized by the DOE radius.
        """

        x_norm, y_norm = self.normalized_coordinates(x, y)
        radius_squared = x_norm.square() + y_norm.square()
        even = (
            self.a2 * radius_squared
            + self.a4 * radius_squared.square()
            + self.a6 * radius_squared.pow(3)
        )
        odd = (
            self.a3 * (x_norm.pow(3) + y_norm.pow(3))
            + self.a5 * (x_norm.pow(5) + y_norm.pow(5))
            + self.a7 * (x_norm.pow(7) + y_norm.pow(7))
        )
        return even + odd

    @staticmethod
    def wrap_phase(phase: Tensor) -> Tensor:
        """Wrap a phase tensor to the half-open interval ``[0, 2*pi)``."""

        return torch.remainder(phase, math.tau)

    def wavelength_scale(
        self,
        wavelength_um: Optional[Union[float, Tensor]] = None,
        refractive_index: Optional[Union[float, Tensor]] = None,
    ) -> Tensor:
        """Return the DeepLens wavelength/material phase scale.

        The scale is ``(lambda0/lambda) * (n(lambda)-1)/(n0-1)``. The target
        refractive index defaults to the design index when it is not supplied.
        """

        wavelength = self.design_wavelength_um
        if wavelength_um is not None:
            wavelength = torch.as_tensor(
                wavelength_um, dtype=self.a2.dtype, device=self.a2.device
            )
        index = self.design_refractive_index
        if refractive_index is not None:
            index = torch.as_tensor(
                refractive_index, dtype=self.a2.dtype, device=self.a2.device
            )
        if torch.any(wavelength <= 0):
            raise ValueError("wavelength_um must be positive")
        if torch.any(index <= 1):
            raise ValueError("refractive_index must be greater than one")
        return (self.design_wavelength_um / wavelength) * (
            (index - 1) / (self.design_refractive_index - 1)
        )

    def continuous_phase(
        self,
        x: Tensor,
        y: Tensor,
        *,
        wavelength_um: Optional[Union[float, Tensor]] = None,
        refractive_index: Optional[Union[float, Tensor]] = None,
        wrap: bool = True,
        wrap_before_scaling: bool = False,
    ) -> Tensor:
        """Evaluate the wavelength-scaled, non-quantized phase.

        By default scaling precedes wrapping. Set ``wrap_before_scaling=True``
        to reproduce historical DeepLens ordering, which wrapped the design
        phase first and then applied wavelength/material scaling.
        """

        phase = self.raw_phase(x, y)
        if wrap_before_scaling:
            phase = self.wrap_phase(phase)
        phase = phase * self.wavelength_scale(wavelength_um, refractive_index)
        if wrap and not wrap_before_scaling:
            phase = self.wrap_phase(phase)
        return phase

    def quantize_phase(
        self,
        phase: Tensor,
        *,
        levels: Optional[int] = None,
        straight_through: bool = True,
    ) -> Tensor:
        """Uniformly quantize wrapped phase while optionally preserving gradients.

        ``levels=16`` creates exactly 16 states in ``[0, 2*pi)``. In
        straight-through mode the forward value is quantized and the backward
        derivative with respect to the input phase is one.
        """

        level_count = self.quantization_levels if levels is None else levels
        if isinstance(level_count, bool) or not isinstance(level_count, int):
            raise ValueError("levels must be an integer of at least two")
        if level_count < 2:
            raise ValueError("levels must be an integer of at least two")
        wrapped = self.wrap_phase(phase)
        step = math.tau / level_count
        indices = torch.remainder(torch.round(wrapped / step), level_count)
        quantized = indices * step
        if straight_through:
            return wrapped + (quantized - wrapped).detach()
        return quantized

    def quantized_phase(
        self,
        x: Tensor,
        y: Tensor,
        *,
        wavelength_um: Optional[Union[float, Tensor]] = None,
        refractive_index: Optional[Union[float, Tensor]] = None,
        levels: Optional[int] = None,
        straight_through: bool = True,
        wrap_before_scaling: bool = False,
    ) -> Tensor:
        """Evaluate wavelength-scaled phase and quantize it to fabrication levels."""

        phase = self.continuous_phase(
            x,
            y,
            wavelength_um=wavelength_um,
            refractive_index=refractive_index,
            wrap=True,
            wrap_before_scaling=wrap_before_scaling,
        )
        return self.quantize_phase(
            phase, levels=levels, straight_through=straight_through
        )

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        *,
        quantized: bool = False,
        wavelength_um: Optional[Union[float, Tensor]] = None,
        refractive_index: Optional[Union[float, Tensor]] = None,
        wrap: bool = True,
        straight_through: bool = True,
        wrap_before_scaling: bool = False,
    ) -> Tensor:
        """Evaluate either the continuous or quantized phase map."""

        if quantized:
            return self.quantized_phase(
                x,
                y,
                wavelength_um=wavelength_um,
                refractive_index=refractive_index,
                straight_through=straight_through,
                wrap_before_scaling=wrap_before_scaling,
            )
        return self.continuous_phase(
            x,
            y,
            wavelength_um=wavelength_um,
            refractive_index=refractive_index,
            wrap=wrap,
            wrap_before_scaling=wrap_before_scaling,
        )

    def export_state(self) -> Dict[str, Any]:
        """Export all parameters and physical configuration for a round-trip."""

        return {
            "format": _CHECKPOINT_FORMAT,
            "version": _CHECKPOINT_VERSION,
            "config": {
                "doe_radius": float(self.doe_radius.detach().cpu()),
                "design_wavelength_um": float(
                    self.design_wavelength_um.detach().cpu()
                ),
                "design_refractive_index": float(
                    self.design_refractive_index.detach().cpu()
                ),
                "quantization_levels": self.quantization_levels,
            },
            "coefficients": self.coefficients.detach().cpu().clone(),
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Poly1DDOE":
        """Construct a module from :meth:`export_state` output."""

        cls._validate_exported_state(state)
        config = state["config"]
        coefficients = torch.as_tensor(state["coefficients"])
        return cls(
            doe_radius=config["doe_radius"],
            coefficients=coefficients,
            design_wavelength_um=config["design_wavelength_um"],
            design_refractive_index=config["design_refractive_index"],
            quantization_levels=config["quantization_levels"],
            device=device,
            dtype=dtype or coefficients.dtype,
        )

    def load_exported_state(self, state: Mapping[str, Any]) -> None:
        """Restore a full exported state into this existing module."""

        self._validate_exported_state(state)
        config = state["config"]
        self._validate_config(**config)
        self.set_coefficients(state["coefficients"])
        with torch.no_grad():
            self.doe_radius.fill_(config["doe_radius"])
            self.design_wavelength_um.fill_(config["design_wavelength_um"])
            self.design_refractive_index.fill_(config["design_refractive_index"])
        self.quantization_levels = int(config["quantization_levels"])

    @staticmethod
    def _validate_exported_state(state: Mapping[str, Any]) -> None:
        if state.get("format") != _CHECKPOINT_FORMAT:
            raise ValueError("not an EDoF reproduction Poly1D DOE state")
        if state.get("version") != _CHECKPOINT_VERSION:
            raise ValueError("unsupported Poly1D DOE state version")
        if "config" not in state or "coefficients" not in state:
            raise ValueError("incomplete Poly1D DOE state")
        required_config = {
            "doe_radius",
            "design_wavelength_um",
            "design_refractive_index",
            "quantization_levels",
        }
        if set(state["config"]) != required_config:
            raise ValueError("invalid Poly1D DOE state config")
        if torch.as_tensor(state["coefficients"]).shape != (6,):
            raise ValueError("invalid Poly1D DOE coefficients")

    def save_checkpoint(self, path: PathType) -> None:
        """Save an explicit, device-independent Poly1D checkpoint."""

        torch.save(self.export_state(), path)

    @classmethod
    def from_checkpoint(
        cls,
        path: PathType,
        *,
        map_location: Optional[Union[str, torch.device]] = "cpu",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Poly1DDOE":
        """Load a checkpoint written by :meth:`save_checkpoint`."""

        try:
            state = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:  # PyTorch before the ``weights_only`` argument.
            state = torch.load(path, map_location=map_location)
        return cls.from_state(state, device=device, dtype=dtype)


__all__ = ["Poly1DDOE"]
