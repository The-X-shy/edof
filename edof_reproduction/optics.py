"""Cached DeepLens ray fields and differentiable Poly1D wave propagation."""

from __future__ import annotations

import importlib.metadata
import hashlib
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import OpticsConfig
from .poly1d import Poly1DDOE


_CACHE_VERSION = 2
_FIXED_PSF_CACHE_VERSION = 1
_LENS_SPACING_ABS_TOLERANCE_MM = 1e-5


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fused_silica_index(wavelength_um: float | Tensor) -> Tensor:
    """Malitson fused-silica Sellmeier equation in the transparent band."""

    wavelength = torch.as_tensor(wavelength_um)
    square = wavelength.square()
    index_square = 1.0
    for coefficient, pole in (
        (0.6961663, 0.0684043),
        (0.4079426, 0.1162414),
        (0.8974794, 9.896161),
    ):
        index_square = index_square + coefficient * square / (square - pole**2)
    return torch.sqrt(index_square)


def angular_spectrum(
    field: Tensor,
    *,
    distance_mm: float,
    wavelength_um: float,
    pixel_pitch_mm: float,
    pad: bool = True,
) -> Tensor:
    """Batch-capable angular-spectrum propagation equivalent to DeepLens ASM."""

    original_height, original_width = field.shape[-2:]
    if pad:
        pad_height, pad_width = original_height // 2, original_width // 2
        field = F.pad(field, (pad_width, pad_width, pad_height, pad_height))
    height, width = field.shape[-2:]
    real_dtype = field.real.dtype
    wavelength_mm = wavelength_um * 1e-3
    fx = torch.fft.fftfreq(width, d=pixel_pitch_mm, device=field.device, dtype=real_dtype)
    fy = torch.fft.fftfreq(height, d=pixel_pitch_mm, device=field.device, dtype=real_dtype)
    radial = 1.0 - wavelength_mm**2 * (
        fy[:, None].square() + fx[None, :].square()
    )
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    root = torch.sqrt(radial.to(complex_dtype))
    transfer = torch.exp(1j * (2.0 * math.pi / wavelength_mm) * distance_mm * root)
    propagated = torch.fft.ifft2(torch.fft.fft2(field) * transfer)
    if pad:
        propagated = propagated[
            ..., pad_height : pad_height + original_height, pad_width : pad_width + original_width
        ]
    return propagated


class CachedRayWaveOptics(nn.Module):
    """Apply trainable Poly1D phase to fixed aberrated fields from DeepLens."""

    def __init__(self, config: OpticsConfig, cache: dict[str, Any], device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.device_for_compute = device
        self.depths = tuple(float(value) for value in cache["depths_mm"])
        self.fields_xy = tuple(tuple(value) for value in cache["fields_xy"])
        self.wavelengths = tuple(float(value) for value in cache["wavelengths_um"])
        self.sensor_resolution = tuple(
            int(value) for value in cache.get("sensor_resolution", config.sensor_resolution)
        )
        self.propagation_distance_mm = float(
            cache.get("doe_sensor_distance_mm", config.doe_sensor_distance_mm)
        )
        # Keep the multi-field complex cache in host memory. Moving the full
        # 10x10 cache to an 8 GB GPU would consume several GB before FFT work
        # begins; each selected wavelength is transferred on demand instead.
        self.cached_fields = cache["fields"].cpu()
        self.register_buffer("centers", cache["centers"], persistent=False)
        propagation_dtype = (
            torch.float64 if config.propagation_precision == "float64" else torch.float32
        )
        normalization_radius = config.doe_normalization_radius_mm
        if normalization_radius is None:
            # The paper DOE is a 3 mm square.  Historical Poly1D normalizes
            # radius by the centre-to-corner distance, not the half width.
            normalization_radius = config.doe_size_mm / math.sqrt(2.0)
        self.doe = Poly1DDOE(
            doe_radius=normalization_radius,
            coefficients=None,
            design_wavelength_um=config.design_wavelength_um,
            design_refractive_index=config.design_refractive_index,
            quantization_levels=config.quantization_levels,
            dtype=propagation_dtype,
            device=device,
        )
        # Sample square DOE pixels at their centres.  Including both physical
        # endpoints makes the pitch size/(N-1), inconsistent with DeepLens.
        pitch = config.doe_size_mm / config.simulation_grid
        axis = (
            torch.arange(config.simulation_grid, dtype=propagation_dtype, device=device) + 0.5
        ) * pitch - config.doe_size_mm / 2.0
        yy, xx = torch.meshgrid(torch.flip(axis, dims=(0,)), axis, indexing="ij")
        self.register_buffer("grid_x", xx, persistent=False)
        self.register_buffer("grid_y", yy, persistent=False)

    @property
    def field_count(self) -> int:
        return len(self.fields_xy)

    def _phase(self, wavelength_um: float) -> Tensor:
        index = fused_silica_index(
            torch.tensor(wavelength_um, device=self.grid_x.device, dtype=self.grid_x.dtype)
        )
        raw_wrapped = self.doe.wrap_phase(self.doe.raw_phase(self.grid_x, self.grid_y))
        if self.config.quantize_during_training:
            design_phase = self.doe.quantize_phase(raw_wrapped, straight_through=True)
        else:
            design_phase = raw_wrapped
        phase = design_phase * self.doe.wavelength_scale(wavelength_um, index)
        return torch.flip(phase, dims=(-2, -1))

    def _crop_psfs(self, intensity: Tensor, centers: Tensor) -> Tensor:
        count = intensity.shape[0]
        size = self.config.psf_size
        offsets = torch.arange(
            -(size // 2), size // 2 + 1, device=intensity.device, dtype=intensity.dtype
        )
        sensor_height, sensor_width = self.sensor_resolution
        # DeepLens pads the DOE field to twice its size and then resamples that
        # full padded field to 2 * sensor_resolution.  Reproduce its integer
        # sensor-pixel centre rounding without materialising that large image.
        full_height, full_width = 2 * sensor_height, 2 * sensor_width
        center_i = torch.round((2.0 - centers[:, 1]) * full_height / 4.0)
        center_j = torch.round((2.0 + centers[:, 0]) * full_width / 4.0)
        sample_i = center_i[:, None, None] + offsets[None, :, None]
        sample_j = center_j[:, None, None] + offsets[None, None, :]
        grid = torch.empty((count, size, size, 2), device=intensity.device, dtype=intensity.dtype)
        grid[..., 0] = 2.0 * (sample_j + 0.5) / full_width - 1.0
        grid[..., 1] = 2.0 * (sample_i + 0.5) / full_height - 1.0
        sampled = F.grid_sample(
            intensity[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )[:, 0]
        sampled = sampled / sampled.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        # The paper keeps coherent tracing, phase modulation and ASM in double
        # precision, then converts intensity PSFs to float for the network.
        return sampled.float()

    def _propagate_chunk(
        self,
        fields: Tensor,
        phase: Tensor,
        centers: Tensor,
        wavelength: float,
    ) -> Tensor:
        modulated = fields * torch.exp(1j * phase)
        height, width = modulated.shape[-2:]
        padded = F.pad(modulated, (width // 2, width // 2, height // 2, height // 2))
        propagated = angular_spectrum(
            padded,
            distance_mm=self.propagation_distance_mm,
            wavelength_um=wavelength,
            pixel_pitch_mm=self.config.doe_size_mm / self.config.simulation_grid,
            pad=False,
        )
        return self._crop_psfs(propagated.abs().square(), centers)

    def _one_wavelength(
        self, wavelength_index: int, field_indices: tuple[int, ...] | None = None
    ) -> Tensor:
        wavelength = self.wavelengths[wavelength_index]
        fields = self.cached_fields[:, :, wavelength_index]
        centers = self.centers[:, :, wavelength_index]
        if field_indices is not None:
            fields = fields[:, list(field_indices)]
            centers = centers[:, list(field_indices)]
        depth_count, field_count = fields.shape[:2]
        fields = fields.reshape(depth_count * field_count, *fields.shape[-2:])
        target_complex = (
            torch.complex128
            if self.config.propagation_precision == "float64"
            else torch.complex64
        )
        phase = self._phase(wavelength).to(
            dtype=torch.float64 if target_complex == torch.complex128 else torch.float32
        )
        centers = centers.reshape(-1, 2)
        chunk_size = self.config.propagation_batch_size or fields.shape[0]
        psf_chunks = []
        for start in range(0, fields.shape[0], chunk_size):
            end = min(start + chunk_size, fields.shape[0])
            field_chunk = fields[start:end].to(device=self.device_for_compute, dtype=target_complex)
            center_chunk = centers[start:end].to(
                device=self.device_for_compute, dtype=phase.dtype
            )
            if torch.is_grad_enabled() and phase.requires_grad:
                psf_chunk = checkpoint(
                    lambda values, phase_map, points: self._propagate_chunk(
                        values, phase_map, points, wavelength
                    ),
                    field_chunk,
                    phase,
                    center_chunk,
                    use_reentrant=False,
                )
            else:
                psf_chunk = self._propagate_chunk(field_chunk, phase, center_chunk, wavelength)
            psf_chunks.append(psf_chunk)
        return torch.cat(psf_chunks).reshape(
            depth_count, field_count, self.config.psf_size, self.config.psf_size
        )

    def psfs(
        self,
        wavelength_choice: tuple[tuple[int, ...], ...],
        *,
        field_indices: tuple[int, ...] | None = None,
    ) -> Tensor:
        """Return PSFs as ``[depth, field, RGB, kernel, kernel]``."""

        channels = []
        for indices in wavelength_choice:
            stack = torch.stack(
                [self._one_wavelength(index, field_indices=field_indices) for index in indices]
            )
            channels.append(stack.mean(dim=0))
        return torch.stack(channels, dim=2)


def _field_coordinates(field_grid: int) -> list[tuple[float, float]]:
    if field_grid == 1:
        return [(0.0, 0.0)]
    half_bin = 1.0 / (2.0 * (field_grid - 1))
    x_axis = torch.linspace(-1.0 + half_bin, 1.0 - half_bin, field_grid).tolist()
    y_axis = torch.linspace(1.0 - half_bin, -1.0 + half_bin, field_grid).tolist()
    return [(float(x), float(y)) for y in y_axis for x in x_axis]


def _all_wavelengths(config: OpticsConfig) -> tuple[float, ...]:
    return tuple(value for channel in config.wavelengths_rgb_um for value in channel)


def _call_doe_field(
    lens: Any,
    *,
    point: Tensor,
    wavelength: float,
    coherent_rays: int,
) -> tuple[Tensor, Any]:
    """Call DeepLens while keeping its field grid at the configured DOE grid.

    DeepLens 2.5 adds an automatic upsampling mode that targets roughly
    4000x4000 samples.  The staged 8 GB run must explicitly select factor 1;
    older DeepLens releases do not expose this keyword and already return the
    configured DOE resolution.
    """

    kwargs: dict[str, Any] = {
        "point": point,
        "wvln": wavelength,
        "spp": coherent_rays,
    }
    if "upsample_factor" in inspect.signature(lens.doe_field).parameters:
        kwargs["upsample_factor"] = 1
    return lens.doe_field(**kwargs)


def _analytic_cache(config: OpticsConfig) -> dict[str, Any]:
    grid = config.simulation_grid
    axis = torch.linspace(-1.0, 1.0, grid, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    radius2 = xx.square() + yy.square()
    aperture = (radius2 <= 1.0).double()
    fields_xy = _field_coordinates(config.field_grid)
    wavelengths = _all_wavelengths(config)
    fields = torch.empty(
        len(config.depths_mm), len(fields_xy), len(wavelengths), grid, grid, dtype=torch.complex128
    )
    centers = torch.empty(len(config.depths_mm), len(fields_xy), len(wavelengths), 2)
    for depth_index, depth in enumerate(config.depths_mm):
        defocus = (1.0 / abs(depth) - 1.0 / abs(config.focus_depth_mm)) * 4000.0
        for field_index, (field_x, field_y) in enumerate(fields_xy):
            for wavelength_index, wavelength in enumerate(wavelengths):
                aberration = defocus * radius2 / wavelength
                aberration += 0.08 * (field_x * xx + field_y * yy) * radius2 / wavelength
                fields[depth_index, field_index, wavelength_index] = aperture * torch.exp(1j * aberration)
                centers[depth_index, field_index, wavelength_index] = torch.tensor((field_x, field_y))
    return {
        "format": "edof_reproduction.cached_fields",
        "version": _CACHE_VERSION,
        "backend": "analytic",
        "depths_mm": config.depths_mm,
        "fields_xy": fields_xy,
        "wavelengths_um": wavelengths,
        "fields": fields,
        "centers": centers,
        "sensor_resolution": config.sensor_resolution,
        "doe_sensor_distance_mm": config.doe_sensor_distance_mm,
        "focus_depth_mm": config.focus_depth_mm,
        "f_number": config.f_number,
        "lens_file_sha256": _sha256(config.lens_file) if Path(config.lens_file).exists() else None,
    }


def _deeplens_cache(config: OpticsConfig) -> dict[str, Any]:
    try:
        from deeplens import HybridLens
    except ImportError as exc:
        raise RuntimeError("DeepLens is required for the deeplens optical backend") from exc

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        cache_device = config.cache_device
        if cache_device == "auto":
            cache_device = "cuda" if torch.cuda.is_available() else "cpu"
        if cache_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA cache generation was requested but is unavailable")
        lens = HybridLens(filename=config.lens_file, device=cache_device)
        lens.refocus(config.focus_depth_mm)
        if hasattr(lens.geolens, "set_fnum"):
            lens.geolens.set_fnum(config.f_number)
        lens.doe.res = (config.simulation_grid, config.simulation_grid)
        lens.doe.ps = config.doe_size_mm / config.simulation_grid
        lens.doe.w = config.doe_size_mm
        lens.doe.h = config.doe_size_mm
        fields_xy = _field_coordinates(config.field_grid)
        wavelengths = _all_wavelengths(config)
        complex_dtype = torch.complex128 if config.cache_complex_dtype == "complex128" else torch.complex64
        fields = torch.empty(
            len(config.depths_mm), len(fields_xy), len(wavelengths),
            config.simulation_grid, config.simulation_grid, dtype=complex_dtype,
        )
        centers = torch.empty(len(config.depths_mm), len(fields_xy), len(wavelengths), 2)
        with torch.no_grad():
            total = len(config.depths_mm) * len(fields_xy) * len(wavelengths)
            completed = 0
            for depth_index, depth in enumerate(config.depths_mm):
                for field_index, (field_x, field_y) in enumerate(fields_xy):
                    point = torch.tensor([field_x, field_y, depth])
                    for wavelength_index, wavelength in enumerate(wavelengths):
                        wavefront, center = _call_doe_field(
                            lens,
                            point=point,
                            wavelength=wavelength,
                            coherent_rays=config.coherent_rays,
                        )
                        wavefront = wavefront.squeeze()
                        expected_grid = (config.simulation_grid, config.simulation_grid)
                        if tuple(wavefront.shape) != expected_grid:
                            raise RuntimeError(
                                f"DeepLens returned field shape {tuple(wavefront.shape)}; "
                                f"expected {expected_grid}. Check DOE resolution and upsampling."
                            )
                        fields[depth_index, field_index, wavelength_index].copy_(
                            wavefront.cpu().to(complex_dtype)
                        )
                        centers[depth_index, field_index, wavelength_index] = torch.as_tensor(center).cpu()
                        completed += 1
                        print(
                            f"CACHE {completed}/{total} depth={depth:g} "
                            f"field=({field_x:g},{field_y:g}) wavelength={wavelength:g}",
                            flush=True,
                        )
        return {
            "format": "edof_reproduction.cached_fields",
            "version": _CACHE_VERSION,
            "backend": "deeplens",
            "cache_device": cache_device,
            "deeplens_version": _deeplens_version(),
            "depths_mm": config.depths_mm,
            "fields_xy": fields_xy,
            "wavelengths_um": wavelengths,
            "fields": fields,
            "centers": centers,
            "sensor_resolution": tuple(int(value) for value in lens.geolens.sensor_res),
            "doe_sensor_distance_mm": float(lens.geolens.d_sensor - lens.doe.d),
            "f_number": config.f_number,
            "focus_depth_mm": config.focus_depth_mm,
            "lens_file_sha256": _sha256(config.lens_file),
        }
    finally:
        torch.set_default_dtype(previous_dtype)


def load_or_build_cache(config: OpticsConfig, output_dir: Path, *, force: bool = False) -> tuple[dict[str, Any], Path]:
    path = output_dir / config.cache_file
    if path.exists() and not force:
        cache = torch.load(path, map_location="cpu", weights_only=False)
    else:
        cache = _deeplens_cache(config) if config.backend == "deeplens" else _analytic_cache(config)
        torch.save(cache, path)
    expected = (
        len(config.depths_mm), config.field_grid**2, 9,
        config.simulation_grid, config.simulation_grid,
    )
    if cache.get("version") != _CACHE_VERSION:
        raise ValueError("cached field format is outdated; rebuild the optical cache")
    if cache.get("backend") != config.backend:
        raise ValueError("cached optical backend does not match the configuration")
    if tuple(cache["fields"].shape) != expected:
        raise ValueError(f"cached field shape {tuple(cache['fields'].shape)} does not match {expected}")
    if tuple(float(value) for value in cache["depths_mm"]) != tuple(config.depths_mm):
        raise ValueError("cached depths do not match the configuration")
    if tuple(float(value) for value in cache["wavelengths_um"]) != _all_wavelengths(config):
        raise ValueError("cached wavelengths do not match the configuration")
    expected_fields = torch.tensor(_field_coordinates(config.field_grid), dtype=torch.float64)
    cached_fields = torch.tensor(cache["fields_xy"], dtype=torch.float64)
    if not torch.allclose(cached_fields, expected_fields, atol=1e-6, rtol=0.0):
        raise ValueError("cached field coordinates do not match the DeepLens PSF-map grid")
    if float(cache.get("focus_depth_mm", float("nan"))) != float(config.focus_depth_mm):
        raise ValueError("cached focus depth does not match the configuration")
    if tuple(int(value) for value in cache.get("sensor_resolution", ())) != tuple(config.sensor_resolution):
        raise ValueError("cached sensor resolution does not match the configuration")
    if not math.isclose(float(cache.get("f_number", float("nan"))), config.f_number, abs_tol=1e-8):
        raise ValueError("cached f-number does not match the configuration")
    if cache.get("backend") == "deeplens":
        if cache.get("lens_file_sha256") != _sha256(config.lens_file):
            raise ValueError("cached lens prescription hash does not match the configuration")
        if "doe_sensor_distance_mm" not in cache:
            raise ValueError("cached propagation distance is missing; rebuild the optical cache")
    return cache, path


def _deeplens_version() -> str:
    for distribution in ("deeplens-core", "deeplens"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def cache_description(cache: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": cache["backend"],
        "deeplens_version": cache.get("deeplens_version"),
        "field_shape": list(cache["fields"].shape),
        "field_dtype": str(cache["fields"].dtype),
        "bytes": cache["fields"].nelement() * cache["fields"].element_size(),
        "depths_mm": list(cache["depths_mm"]),
        "wavelengths_um": list(cache["wavelengths_um"]),
        "sensor_resolution": list(cache.get("sensor_resolution", ())),
        "doe_sensor_distance_mm": cache.get("doe_sensor_distance_mm"),
    }


def _fixed_psf_signature(
    config: OpticsConfig,
    optics: CachedRayWaveOptics,
    target_grid: int,
) -> dict[str, Any]:
    lens_path = Path(config.lens_file)
    return {
        "backend": config.backend,
        "lens_file_sha256": _sha256(lens_path) if lens_path.exists() else None,
        "depths_mm": list(config.depths_mm),
        "wavelengths_rgb_um": [list(group) for group in config.wavelengths_rgb_um],
        "target_field_grid": target_grid,
        "simulation_grid": config.simulation_grid,
        "psf_size": config.psf_size,
        "coherent_rays": config.coherent_rays,
        "doe_size_mm": config.doe_size_mm,
        "f_number": config.f_number,
        "sensor_resolution": list(config.sensor_resolution),
        "focus_depth_mm": config.focus_depth_mm,
        "propagation_distance_mm": optics.propagation_distance_mm,
        "doe_normalization_radius_mm": float(optics.doe.doe_radius.detach().cpu()),
        "design_wavelength_um": config.design_wavelength_um,
        "design_refractive_index": config.design_refractive_index,
        "quantization_levels": config.quantization_levels,
        "quantize_during_training": config.quantize_during_training,
        "propagation_precision": config.propagation_precision,
        "doe_coefficients": [
            float(value) for value in optics.doe.coefficients.detach().cpu()
        ],
    }


def _save_fixed_psf_payload(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "format": payload["format"],
                "version": payload["version"],
                "signature": payload["signature"],
                "fields_total": int(payload["completed_fields"].numel()),
                "fields_completed": int(payload["completed_fields"].sum()),
                "complete": bool(payload["completed_fields"].all()),
                "psf_shape": list(payload["psfs"].shape),
                "psf_dtype": str(payload["psfs"].dtype),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _validate_fixed_psf_payload(
    payload: dict[str, Any],
    signature: dict[str, Any],
    target_grid: int,
    config: OpticsConfig,
) -> None:
    if payload.get("format") != "edof_reproduction.fixed_psfs":
        raise ValueError("fixed PSF cache has an unknown format")
    if payload.get("version") != _FIXED_PSF_CACHE_VERSION:
        raise ValueError("fixed PSF cache is outdated; rebuild it")
    if payload.get("signature") != signature:
        raise ValueError("fixed PSF cache does not match the lens, DOE, or configuration")
    expected = (
        len(config.depths_mm),
        target_grid**2,
        len(config.wavelengths_rgb_um),
        config.psf_size,
        config.psf_size,
    )
    if tuple(payload["psfs"].shape) != expected:
        raise ValueError(f"fixed PSF shape {tuple(payload['psfs'].shape)} does not match {expected}")
    if payload["psfs"].dtype != torch.float32:
        raise ValueError("fixed PSFs must use float32")
    if tuple(payload["completed_fields"].shape) != (target_grid**2,):
        raise ValueError("fixed PSF completion map has the wrong shape")


def _exact_analytic_psfs(
    config: OpticsConfig,
    optics: CachedRayWaveOptics,
    target_grid: int,
) -> Tensor:
    exact_config = replace(config, field_grid=target_grid)
    exact_cache = _analytic_cache(exact_config)
    exact_optics = CachedRayWaveOptics(
        exact_config, exact_cache, optics.device_for_compute
    ).to(optics.device_for_compute)
    exact_optics.doe.set_coefficients(optics.doe.coefficients.detach())
    return exact_optics.psfs(((0, 1, 2), (3, 4, 5), (6, 7, 8))).detach().cpu()


def _fill_exact_deeplens_psfs(
    config: OpticsConfig,
    optics: CachedRayWaveOptics,
    payload: dict[str, Any],
    path: Path,
    target_grid: int,
) -> None:
    try:
        from deeplens import HybridLens
    except ImportError as exc:
        raise RuntimeError("DeepLens is required to compute the exact fixed PSF map") from exc

    previous_dtype = torch.get_default_dtype()
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.set_default_dtype(torch.float64)
    try:
        cache_device = config.cache_device
        if cache_device == "auto":
            cache_device = "cuda" if torch.cuda.is_available() else "cpu"
        if cache_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA PSF generation was requested but is unavailable")
        lens = HybridLens(filename=config.lens_file, device=cache_device)
        lens.refocus(config.focus_depth_mm)
        if hasattr(lens.geolens, "set_fnum"):
            lens.geolens.set_fnum(config.f_number)
        lens.doe.res = (config.simulation_grid, config.simulation_grid)
        lens.doe.ps = config.doe_size_mm / config.simulation_grid
        lens.doe.w = config.doe_size_mm
        lens.doe.h = config.doe_size_mm
        # DeepLens refocus estimates the sensor plane from sampled rays, so a
        # second HybridLens can differ by a fraction of a micron.  The PSF-map
        # pass must use the sensor plane recorded with the base optical cache.
        pinned_sensor_distance = float(lens.doe.d) + optics.propagation_distance_mm
        lens.geolens.d_sensor.copy_(
            torch.as_tensor(
                pinned_sensor_distance,
                device=lens.geolens.d_sensor.device,
                dtype=lens.geolens.d_sensor.dtype,
            )
        )
        lens.geolens.post_computation()
        propagation_distance = float(lens.geolens.d_sensor - lens.doe.d)
        if not math.isclose(
            propagation_distance,
            optics.propagation_distance_mm,
            rel_tol=0.0,
            abs_tol=_LENS_SPACING_ABS_TOLERANCE_MM,
        ):
            raise ValueError(
                "exact PSF lens spacing does not match the training optical cache: "
                f"live={propagation_distance:.12g} mm, "
                f"cached={optics.propagation_distance_mm:.12g} mm"
            )

        target_complex = (
            torch.complex128
            if config.propagation_precision == "float64"
            else torch.complex64
        )
        phase_dtype = torch.float64 if target_complex == torch.complex128 else torch.float32
        phase_maps = {
            float(wavelength): optics._phase(float(wavelength)).detach().to(dtype=phase_dtype)
            for group in config.wavelengths_rgb_um
            for wavelength in group
        }
        fields_xy = _field_coordinates(target_grid)
        completed = payload["completed_fields"]
        psfs = payload["psfs"]
        save_every = config.finetune_psf_save_every_fields
        total_fields = len(fields_xy)
        try:
            with torch.no_grad():
                for field_index, (field_x, field_y) in enumerate(fields_xy):
                    if bool(completed[field_index]):
                        continue
                    for depth_index, depth in enumerate(config.depths_mm):
                        point = torch.tensor([field_x, field_y, depth])
                        for channel_index, wavelengths in enumerate(config.wavelengths_rgb_um):
                            channel_psf = torch.zeros(
                                config.psf_size, config.psf_size, dtype=torch.float32
                            )
                            for wavelength in wavelengths:
                                # Make every expensive ray trace independent of
                                # loop history, so an interrupted map resumes
                                # with the same per-field Monte Carlo samples.
                                seed_material = (
                                    f"{field_index}:{depth_index}:{float(wavelength):.12g}:"
                                    f"{config.coherent_rays}:{payload['signature']['lens_file_sha256']}"
                                )
                                ray_seed = int(
                                    hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16],
                                    16,
                                ) % (2**63 - 1)
                                torch.manual_seed(ray_seed)
                                if torch.cuda.is_available():
                                    torch.cuda.manual_seed_all(ray_seed)
                                wavefront, center = _call_doe_field(
                                    lens,
                                    point=point,
                                    wavelength=float(wavelength),
                                    coherent_rays=config.coherent_rays,
                                )
                                wavefront = wavefront.squeeze()
                                expected_grid = (config.simulation_grid, config.simulation_grid)
                                if tuple(wavefront.shape) != expected_grid:
                                    raise RuntimeError(
                                        f"DeepLens returned field shape {tuple(wavefront.shape)}; "
                                        f"expected {expected_grid}"
                                    )
                                field = wavefront[None].to(
                                    device=optics.device_for_compute, dtype=target_complex
                                )
                                center_tensor = torch.as_tensor(
                                    center,
                                    device=optics.device_for_compute,
                                    dtype=phase_dtype,
                                ).reshape(1, 2)
                                propagated = optics._propagate_chunk(
                                    field,
                                    phase_maps[float(wavelength)],
                                    center_tensor,
                                    float(wavelength),
                                )[0]
                                channel_psf.add_(propagated.detach().cpu() / len(wavelengths))
                            psfs[depth_index, field_index, channel_index].copy_(channel_psf)
                    completed[field_index] = True
                    completed_count = int(completed.sum())
                    print(
                        json.dumps(
                            {
                                "fixed_psf_progress": {
                                    "fields_completed": completed_count,
                                    "fields_total": total_fields,
                                    "field": [field_x, field_y],
                                }
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    if completed_count % save_every == 0 or completed_count == total_fields:
                        _save_fixed_psf_payload(payload, path)
        except BaseException:
            _save_fixed_psf_payload(payload, path)
            raise
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


def load_or_build_fixed_psf_map(
    config: OpticsConfig,
    optics: CachedRayWaveOptics,
    output_dir: Path,
    *,
    force: bool = False,
) -> tuple[Tensor, Path]:
    """Load or compute a true field-sampled fixed PSF map for network fine-tuning."""

    target_grid = config.finetune_field_grid or config.field_grid
    path = output_dir / config.finetune_psf_cache_file
    signature = _fixed_psf_signature(config, optics, target_grid)
    if path.exists() and not force:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _validate_fixed_psf_payload(payload, signature, target_grid, config)
    else:
        payload = {
            "format": "edof_reproduction.fixed_psfs",
            "version": _FIXED_PSF_CACHE_VERSION,
            "signature": signature,
            "fields_xy": _field_coordinates(target_grid),
            "psfs": torch.zeros(
                len(config.depths_mm),
                target_grid**2,
                len(config.wavelengths_rgb_um),
                config.psf_size,
                config.psf_size,
                dtype=torch.float32,
            ),
            "completed_fields": torch.zeros(target_grid**2, dtype=torch.bool),
        }
        _save_fixed_psf_payload(payload, path)

    if not bool(payload["completed_fields"].all()):
        if config.backend == "analytic":
            payload["psfs"].copy_(_exact_analytic_psfs(config, optics, target_grid))
            payload["completed_fields"].fill_(True)
            _save_fixed_psf_payload(payload, path)
        else:
            _fill_exact_deeplens_psfs(config, optics, payload, path, target_grid)
    _validate_fixed_psf_payload(payload, signature, target_grid, config)
    if not bool(payload["completed_fields"].all()):
        raise RuntimeError("fixed PSF map is incomplete")
    return payload["psfs"], path
