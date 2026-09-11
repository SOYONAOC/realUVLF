"""HMF weighting and Monte-Carlo UVLF estimation.

The production halo mass function is the Reed et al. (2007) fit, DOI:
10.1111/j.1365-2966.2006.11204.x, arXiv:astro-ph/0607150, evaluated with the
``hmf`` implementation described by Murray, Power & Robotham (2013), DOI:
10.1016/j.ascom.2013.11.001, arXiv:1306.6721.  AuroraLF's mass interpolation,
two-level Monte-Carlo sampling, binning, and weighting are numerical estimator
choices and should not be attributed to Reed et al.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from hmf import MassFunction

from auroralf.constants import AB_ZEROPOINT_LNU, PLANCK18_NS, PLANCK18_SIGMA8
from auroralf.model_options import (
    DEFAULT_IMF_TRANSITION_PARAMETERS,
    DEFAULT_MASS_FUNCTION_MODEL,
    IMF_MODE_CANONICAL,
    IMFTransitionParameters,
    MASS_FUNCTION_MODEL_HMF_REED07,
    MASS_FUNCTION_MODELS,
    validate_imf_mode,
    validate_mass_function_model,
)
from auroralf.chemistry import MZRBirthMetallicityParameters, RegulatorMetallicityParameters
from auroralf.seeding import (
    PipelineRandomSeeds,
    derive_hmf_mass_seed,
    derive_pipeline_random_seeds,
)
from auroralf.mah import (
    MAH_BACKEND_MCBRIDE,
    MAH_BACKEND_THESAN,
    MAH_BACKEND_TNG,
    THESAN_TIME_GRID_SNAPSHOT,
    validate_thesan_time_grid_mode,
    TNG_TIME_GRID_SNAPSHOT,
    Cosmology,
    validate_mah_backend,
    validate_tng_time_grid_mode,
)
from auroralf.sfr import (
    DEFAULT_POPIII_SFR_PARAMETERS,
    DEFAULT_SFR_MODEL_PARAMETERS,
    PopIIISFRParameters,
    SFRModelParameters,
)
from auroralf.cooling import (
    ATOMIC_COOLING_MU,
    ATOMIC_COOLING_TEMPERATURE_K,
    DEFAULT_LW_BACKGROUND_J21,
    POPIII_LW_FEEDBACK_COEFFICIENT,
    POPIII_LW_FEEDBACK_EXPONENT,
    POPIII_MOLECULAR_COOLING_M0_NORMALIZATION_MSUN,
    POPIII_MOLECULAR_COOLING_REDSHIFT_EXPONENT,
    STELLAR_CHANNEL_BELOW_POPIII_MIN,
    STELLAR_CHANNEL_POPII,
    STELLAR_CHANNEL_POPIII,
    STELLAR_CHANNELS,
    classify_halo_stellar_channels,
    compute_atomic_cooling_mass_msun,
    compute_popiii_lw_minimum_mass_msun,
)
from .pipeline import (
    DEFAULT_BURST_SCATTER_TIMESCALE_MYR,
    DEFAULT_POPIII_SSP_FILE,
    DEFAULT_SSP_FILE,
    DEFAULT_TOPHEAVY_SSP_FILE,
    DEFAULT_TOPHEAVY_SSP_METALLICITY,
    default_worker_count,
    run_halo_uv_pipeline,
)


LOGM_MIN = 9.0
LOGM_MAX = 13.0
DEFAULT_HMF_DLOG10M = 0.005
MASS_FUNCTION_NS = PLANCK18_NS
MASS_FUNCTION_SIGMA8 = PLANCK18_SIGMA8
HMF_REED07_FITTING_FUNCTION = "Reed07"


@dataclass(frozen=True)
class UVLFSamplingResult:
    samples: dict[str, np.ndarray]
    uvlf: dict[str, np.ndarray]
    metadata: dict[str, Any]


def uv_luminosity_to_muv(luminosity_nu: np.ndarray | float) -> np.ndarray | float:
    luminosity = np.asarray(luminosity_nu, dtype=float)
    muv = np.full_like(luminosity, np.nan, dtype=float)
    positive = luminosity > 0.0
    muv[positive] = -2.5 * np.log10(luminosity[positive]) + AB_ZEROPOINT_LNU
    if np.ndim(luminosity_nu) == 0:
        return float(muv)
    return muv


def _strict_finite_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-boolean value")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_real_mass_values(name: str, values: object) -> None:
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.bool_) or np.issubdtype(
            values.dtype,
            np.complexfloating,
        ):
            raise TypeError(f"{name} must contain real non-boolean values")
        if np.issubdtype(values.dtype, np.integer) or np.issubdtype(
            values.dtype,
            np.floating,
        ):
            return
        if values.dtype == np.dtype(object) and all(
            isinstance(item, Real) and not isinstance(item, (bool, np.bool_))
            for item in values.flat
        ):
            return
        raise TypeError(f"{name} must contain real non-boolean values")
    if isinstance(values, (list, tuple)) and all(
        isinstance(item, Real) and not isinstance(item, (bool, np.bool_))
        for item in values
    ):
        return
    if isinstance(values, Real) and not isinstance(values, (bool, np.bool_)):
        return
    raise TypeError(f"{name} must contain real non-boolean values")


def _immutable_float_vector(name: str, values: object) -> np.ndarray:
    _validate_real_mass_values(name, values)
    vector = np.array(values, dtype=float, copy=True)
    if vector.ndim != 1 or vector.size < 2 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 1D array with at least two values")
    immutable = np.frombuffer(vector.tobytes(order="C"), dtype=vector.dtype).reshape(
        vector.shape
    )
    immutable.flags.writeable = False
    return immutable


@dataclass(frozen=True, slots=True)
class Reed07HMFInterpolator:
    log10_halo_mass_min_msun: float
    log10_halo_mass_max_msun: float
    redshift: float
    cosmology: Cosmology
    hmf_dlog10m: float
    log_mass_grid: np.ndarray
    log_dndm_grid: np.ndarray

    def __post_init__(self) -> None:
        log_min = _strict_finite_real(
            "log10_halo_mass_min_msun",
            self.log10_halo_mass_min_msun,
        )
        log_max = _strict_finite_real(
            "log10_halo_mass_max_msun",
            self.log10_halo_mass_max_msun,
        )
        redshift = _strict_finite_real("redshift", self.redshift)
        step = _strict_finite_real("hmf_dlog10m", self.hmf_dlog10m)
        if log_max <= log_min:
            raise ValueError(
                "log10_halo_mass_max_msun must exceed log10_halo_mass_min_msun"
            )
        if redshift < 0.0:
            raise ValueError("redshift must be non-negative")
        if step <= 0.0:
            raise ValueError("hmf_dlog10m must be positive")
        if type(self.cosmology) is not Cosmology:
            raise TypeError("cosmology must be exactly Cosmology")
        log_mass_grid = _immutable_float_vector(
            "log_mass_grid",
            self.log_mass_grid,
        )
        log_dndm_grid = _immutable_float_vector(
            "log_dndm_grid",
            self.log_dndm_grid,
        )
        if log_mass_grid.size != log_dndm_grid.size:
            raise ValueError("HMF interpolation grids must have equal length")
        if np.any(np.diff(log_mass_grid) <= 0.0):
            raise ValueError("log_mass_grid must be strictly increasing")
        configured_log_mass = np.log(10.0) * np.array([log_min, log_max])
        if (
            configured_log_mass[0] < log_mass_grid[0]
            or configured_log_mass[1] > log_mass_grid[-1]
        ):
            raise ValueError("HMF interpolation grid must cover the configured mass range")
        for name, value in (
            ("log10_halo_mass_min_msun", log_min),
            ("log10_halo_mass_max_msun", log_max),
            ("redshift", redshift),
            ("hmf_dlog10m", step),
            ("log_mass_grid", log_mass_grid),
            ("log_dndm_grid", log_dndm_grid),
        ):
            object.__setattr__(self, name, value)

    def evaluate(self, halo_mass_msun: object) -> np.ndarray | float:
        _validate_real_mass_values("halo mass", halo_mass_msun)
        mass = np.array(halo_mass_msun, dtype=float, copy=True)
        if mass.size == 0:
            raise ValueError("halo mass must be non-empty")
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
            raise ValueError("halo mass must be finite and positive")
        log10_mass = np.log10(mass)
        if np.any(log10_mass < self.log10_halo_mass_min_msun) or np.any(
            log10_mass > self.log10_halo_mass_max_msun
        ):
            raise ValueError("halo mass must lie within the configured mass range")
        result = np.exp(
            np.interp(np.log(mass), self.log_mass_grid, self.log_dndm_grid)
        )
        if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
            raise RuntimeError("Reed07 HMF interpolation returned invalid dn/dM values")
        if np.ndim(halo_mass_msun) == 0:
            return float(result)
        return np.asarray(result, dtype=float).reshape(mass.shape)


def prepare_reed07_hmf_interpolator(
    *,
    log10_halo_mass_min_msun: float,
    log10_halo_mass_max_msun: float,
    z_obs: float,
    cosmology: Cosmology,
    hmf_dlog10m: float = DEFAULT_HMF_DLOG10M,
) -> Reed07HMFInterpolator:
    """Build an interpolator for the direct Reed et al. (2007) HMF fit.

    ``hmf`` evaluates the fitting function; the explicit ``h`` conversions
    below convert its native mass and density units to ``Msun`` and
    ``Mpc^-3 Msun^-1``.  Grid padding/interpolation is AuroraLF numerics.
    """
    log_min = _strict_finite_real(
        "log10_halo_mass_min_msun",
        log10_halo_mass_min_msun,
    )
    log_max = _strict_finite_real(
        "log10_halo_mass_max_msun",
        log10_halo_mass_max_msun,
    )
    redshift = _strict_finite_real("z_obs", z_obs)
    step = _strict_finite_real("hmf_dlog10m", hmf_dlog10m)
    if log_max <= log_min:
        raise ValueError(
            "log10_halo_mass_max_msun must exceed log10_halo_mass_min_msun"
        )
    if redshift < 0.0:
        raise ValueError("z_obs must be non-negative")
    if step <= 0.0:
        raise ValueError("hmf_dlog10m must be positive")
    if type(cosmology) is not Cosmology:
        raise TypeError("cosmology must be exactly Cosmology")

    h = cosmology.h0_km_s_mpc / 100.0
    log_h = np.log10(h)
    grid_min = np.floor((log_min + log_h - 2.0 * step) / step) * step
    grid_max = np.ceil((log_max + log_h + 2.0 * step) / step) * step
    grid_max += step

    mass_function = MassFunction(
        Mmin=grid_min,
        Mmax=grid_max,
        dlog10m=step,
        z=redshift,
        hmf_model=HMF_REED07_FITTING_FUNCTION,
        sigma_8=MASS_FUNCTION_SIGMA8,
        n=MASS_FUNCTION_NS,
        cosmo_params={
            "H0": cosmology.h0_km_s_mpc,
            "Om0": cosmology.omega_m,
            "Ob0": cosmology.omega_b,
        },
        transfer_params={"extrapolate_with_eh": True},
    )

    grid_mass_msun = np.asarray(mass_function.m, dtype=float) / h
    grid_dndm = np.asarray(mass_function.dndm, dtype=float) * h**4
    valid = np.isfinite(grid_mass_msun) & np.isfinite(grid_dndm) & (grid_mass_msun > 0.0) & (grid_dndm > 0.0)
    if np.count_nonzero(valid) < 2:
        raise RuntimeError(f"{MASS_FUNCTION_MODEL_HMF_REED07} returned too few positive mass-function samples")

    log_grid_mass = np.log(grid_mass_msun[valid])
    log_grid_dndm = np.log(grid_dndm[valid])
    return Reed07HMFInterpolator(
        log10_halo_mass_min_msun=log_min,
        log10_halo_mass_max_msun=log_max,
        redshift=redshift,
        cosmology=cosmology,
        hmf_dlog10m=step,
        log_mass_grid=log_grid_mass,
        log_dndm_grid=log_grid_dndm,
    )


def _hmf_reed07_dndm(
    halo_mass_msun: np.ndarray,
    z_obs: float,
    *,
    cosmology: Cosmology,
    hmf_dlog10m: float,
) -> np.ndarray:
    log_mass = np.log10(halo_mass_msun)
    log_min = float(np.min(log_mass))
    log_max = float(np.max(log_mass))
    if log_max == log_min:
        log_max = float(np.nextafter(log_min, np.inf))
    interpolator = prepare_reed07_hmf_interpolator(
        log10_halo_mass_min_msun=log_min,
        log10_halo_mass_max_msun=log_max,
        z_obs=z_obs,
        cosmology=cosmology,
        hmf_dlog10m=hmf_dlog10m,
    )
    return np.asarray(interpolator.evaluate(halo_mass_msun), dtype=float)


def compute_reed07_halo_mass_function_dndm(
    halo_mass_msun: np.ndarray | float,
    z_obs: float,
    *,
    cosmology: Cosmology,
    hmf_dlog10m: float = DEFAULT_HMF_DLOG10M,
) -> np.ndarray | float:
    """Evaluate ``dn/dM`` from Reed et al. (2007) through ``hmf``."""
    if not isinstance(cosmology, Cosmology):
        raise TypeError("cosmology must be an instance of auroralf.mah.models.Cosmology")
    mass = np.asarray(halo_mass_msun, dtype=float)
    if not np.all(np.isfinite(mass)):
        raise ValueError("halo masses must be finite")
    if np.any(mass <= 0.0):
        raise ValueError("halo masses must be positive")

    mass_1d = np.atleast_1d(mass)
    dndm = _hmf_reed07_dndm(
        mass_1d,
        float(z_obs),
        cosmology=cosmology,
        hmf_dlog10m=float(hmf_dlog10m),
    )
    if not np.all(np.isfinite(dndm)):
        raise RuntimeError(f"{MASS_FUNCTION_MODEL_HMF_REED07} returned non-finite dn/dM values")
    if np.any(dndm < 0.0):
        raise RuntimeError(f"{MASS_FUNCTION_MODEL_HMF_REED07} returned negative dn/dM values")
    dndm = np.asarray(dndm, dtype=float).reshape(mass_1d.shape)
    if np.ndim(halo_mass_msun) == 0:
        return float(dndm[0])
    return dndm.reshape(mass.shape)


def compute_halo_mass_function_dndm(
    halo_mass_msun: np.ndarray | float,
    z_obs: float,
    *,
    cosmology: Cosmology,
    mass_function_model: str = DEFAULT_MASS_FUNCTION_MODEL,
    hmf_dlog10m: float = DEFAULT_HMF_DLOG10M,
) -> np.ndarray | float:
    model = validate_mass_function_model(mass_function_model)
    if model != MASS_FUNCTION_MODEL_HMF_REED07:
        raise RuntimeError(f"unsupported mass function model after validation: {model}")
    return compute_reed07_halo_mass_function_dndm(
        halo_mass_msun,
        z_obs,
        cosmology=cosmology,
        hmf_dlog10m=hmf_dlog10m,
    )


def _resolve_bin_edges(values: np.ndarray, quantity: str, bins: int | np.ndarray) -> np.ndarray:
    if isinstance(bins, np.ndarray):
        if bins.ndim != 1 or bins.size < 2:
            raise ValueError("bins array must be 1D with at least two edges")
        return np.asarray(bins, dtype=float)

    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer or a 1D numpy array")

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("no finite samples available to build histogram edges")

    if quantity == "luminosity":
        positive = finite[finite > 0.0]
        if positive.size == 0:
            raise RuntimeError("no positive luminosity samples available to build histogram edges")
        return np.logspace(np.log10(np.min(positive)), np.log10(np.max(positive)), bins + 1)

    return np.linspace(np.min(finite), np.max(finite), bins + 1)


def _format_progress(completed: int, total: int, elapsed_seconds: float) -> str:
    fraction = completed / total
    filled = int(round(30 * fraction))
    bar = "#" * filled + "-" * (30 - filled)
    rate = completed / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    remaining = total - completed
    eta_seconds = remaining / rate if rate > 0.0 else float("inf")
    eta_text = f"{eta_seconds:.1f}s" if np.isfinite(eta_seconds) else "inf"
    return (
        f"[{bar}] {completed}/{total} "
        f"({fraction * 100.0:.2f}%) "
        f"elapsed={elapsed_seconds:.1f}s "
        f"eta={eta_text}\n"
    )


def _write_progress(progress_path: Path, completed: int, total: int, elapsed_seconds: float) -> str:
    text = _format_progress(completed=completed, total=total, elapsed_seconds=elapsed_seconds)
    progress_path.write_text(text, encoding="utf-8")
    return text


def _finite_median_or_none(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


class _MassSampleTask(NamedTuple):
    """Named, spawn-picklable inputs for the compatibility sampler."""

    mass_index: int
    log_mass: float
    mass: float
    weight: float
    z_obs: float
    n_tracks: int
    z_start_max: float
    n_grid: int
    sampler: str
    mah_backend: str
    tng_mah_cache_path: str | Path | None
    tng_mass_bin_width_dex: float
    tng_min_candidates: int
    tng_smoothing_myr: float
    tng_time_grid_mode: str
    thesan_mah_cache_path: str | Path | None
    thesan_mass_bin_width_dex: float
    thesan_min_candidates: int
    thesan_smoothing_myr: float
    thesan_time_grid_mode: str
    enable_time_delay: bool
    ssp_file: str
    topheavy_ssp_file: str | None
    topheavy_ssp_metallicity: float | None
    imf_mode: str
    imf_transition_parameters: IMFTransitionParameters
    random_seeds: PipelineRandomSeeds
    sfr_model_parameters: SFRModelParameters
    mzr_metallicity_parameters: MZRBirthMetallicityParameters | None
    regulator_metallicity_parameters: RegulatorMetallicityParameters | None
    burst_scatter_dex: float
    burst_scatter_timescale_myr: float
    burst_scatter_preserve_mean: bool
    enable_popiii: bool
    popiii_sfr_parameters: PopIIISFRParameters
    popiii_ssp_file: str
    cosmology: Cosmology


def _run_single_mass_sample(task: _MassSampleTask) -> tuple[Any, ...]:
    t0 = time.perf_counter()
    pipeline_result = run_halo_uv_pipeline(
        n_tracks=task.n_tracks,
        z_final=task.z_obs,
        Mh_final=float(task.mass),
        cosmology=task.cosmology,
        random_seeds=task.random_seeds,
        z_start_max=task.z_start_max,
        n_grid=task.n_grid,
        sampler=task.sampler,
        mah_backend=task.mah_backend,
        tng_mah_cache_path=task.tng_mah_cache_path,
        tng_mass_bin_width_dex=task.tng_mass_bin_width_dex,
        tng_min_candidates=task.tng_min_candidates,
        tng_smoothing_myr=task.tng_smoothing_myr,
        tng_time_grid_mode=task.tng_time_grid_mode,
        thesan_mah_cache_path=task.thesan_mah_cache_path,
        thesan_mass_bin_width_dex=task.thesan_mass_bin_width_dex,
        thesan_min_candidates=task.thesan_min_candidates,
        thesan_smoothing_myr=task.thesan_smoothing_myr,
        thesan_time_grid_mode=task.thesan_time_grid_mode,
        enable_time_delay=task.enable_time_delay,
        workers=1,
        ssp_file=task.ssp_file,
        topheavy_ssp_file=task.topheavy_ssp_file,
        topheavy_ssp_metallicity=task.topheavy_ssp_metallicity,
        imf_mode=task.imf_mode,
        imf_transition_parameters=task.imf_transition_parameters,
        sfr_model_parameters=task.sfr_model_parameters,
        mzr_metallicity_parameters=task.mzr_metallicity_parameters,
        regulator_metallicity_parameters=task.regulator_metallicity_parameters,
        burst_scatter_dex=task.burst_scatter_dex,
        burst_scatter_timescale_myr=task.burst_scatter_timescale_myr,
        burst_scatter_preserve_mean=task.burst_scatter_preserve_mean,
        enable_popiii=task.enable_popiii,
        popiii_sfr_parameters=task.popiii_sfr_parameters,
        popiii_ssp_file=task.popiii_ssp_file,
    )
    duration = time.perf_counter() - t0
    luminosity = np.asarray(pipeline_result.uv_luminosities, dtype=float)
    topheavy_luminosity = np.asarray(pipeline_result.uv_luminosities_topheavy, dtype=float)
    popiii_luminosity = np.asarray(pipeline_result.uv_luminosities_popiii, dtype=float)
    topheavy_light_fraction = np.zeros_like(luminosity, dtype=float)
    popiii_light_fraction = np.zeros_like(luminosity, dtype=float)
    positive_light = luminosity > 0.0
    topheavy_light_fraction[positive_light] = topheavy_luminosity[positive_light] / luminosity[positive_light]
    popiii_light_fraction[positive_light] = popiii_luminosity[positive_light] / luminosity[positive_light]
    active_shape = np.asarray(pipeline_result.active_grid, dtype=bool).shape
    sfr_grid = np.asarray(pipeline_result.sfr_tracks["SFR"], dtype=float)
    if sfr_grid.size != int(np.prod(active_shape)):
        raise RuntimeError("run_halo_uv_pipeline returned an unexpected SFR grid size")
    sfr = sfr_grid.reshape(active_shape)[:, -1]
    if sfr.size != task.n_tracks:
        raise RuntimeError("run_halo_uv_pipeline returned an unexpected number of SFR samples")
    popiii_sfr_grid = np.asarray(pipeline_result.sfr_tracks["SFR_popiii"], dtype=float)
    if popiii_sfr_grid.size != int(np.prod(active_shape)):
        raise RuntimeError("run_halo_uv_pipeline returned an unexpected Pop III SFR grid size")
    popiii_sfr = popiii_sfr_grid.reshape(active_shape)[:, -1]
    if popiii_sfr.size != task.n_tracks:
        raise RuntimeError("run_halo_uv_pipeline returned an unexpected number of Pop III SFR samples")
    return (
        task.mass_index,
        task.log_mass,
        luminosity,
        sfr,
        topheavy_light_fraction,
        popiii_luminosity,
        popiii_light_fraction,
        popiii_sfr,
        duration,
        int(pipeline_result.metadata["topheavy_source_count"]),
        int(pipeline_result.metadata["starforming_source_count"]),
        int(pipeline_result.metadata["popiii_source_count"]),
        int(pipeline_result.metadata["active_source_count"]),
        float(pipeline_result.metadata["final_gas_metallicity_zsun_median"])
        if pipeline_result.metadata["final_gas_metallicity_zsun_median"] is not None
        else np.nan,
        float(pipeline_result.metadata["birth_metallicity_zsun_starforming_median"])
        if pipeline_result.metadata["birth_metallicity_zsun_starforming_median"] is not None
        else np.nan,
    )


def _build_sample_histogram(
    histogram_values: np.ndarray, sample_weight: np.ndarray,
    *, quantity: str, bin_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    """Bin samples with the historical raw, weighted and squared-weight estimator."""
    valid_mask = np.isfinite(histogram_values) & np.isfinite(sample_weight)
    if quantity == "luminosity":
        valid_mask &= histogram_values > 0.0

    weighted_counts, used_edges = np.histogram(
        histogram_values[valid_mask],
        bins=bin_edges,
        weights=sample_weight[valid_mask],
    )
    raw_counts, raw_edges = np.histogram(
        histogram_values[valid_mask],
        bins=bin_edges,
    )
    if not np.allclose(used_edges, raw_edges, rtol=0.0, atol=0.0):
        raise RuntimeError("weighted and raw histogram bin edges differ")
    weight_squared_counts, squared_edges = np.histogram(
        histogram_values[valid_mask],
        bins=bin_edges,
        weights=np.square(sample_weight[valid_mask]),
    )
    if not np.allclose(used_edges, squared_edges, rtol=0.0, atol=0.0):
        raise RuntimeError("weighted and squared-weight histogram bin edges differ")
    bin_width = np.diff(used_edges)
    phi = weighted_counts / bin_width
    weighted_count_sigma = np.sqrt(weight_squared_counts)
    phi_sigma = weighted_count_sigma / bin_width
    effective_counts = np.divide(
        np.square(weighted_counts),
        weight_squared_counts,
        out=np.zeros_like(weighted_counts, dtype=float),
        where=weight_squared_counts > 0.0,
    )
    bin_centers = 0.5 * (used_edges[:-1] + used_edges[1:])
    return {
        "quantity": np.array([quantity]),
        "bin_edges": used_edges,
        "bin_centers": bin_centers,
        "bin_width": bin_width,
        "raw_counts": raw_counts.astype(np.int64),
        "weighted_counts": weighted_counts,
        "weight_squared_counts": weight_squared_counts,
        "weighted_count_sigma": weighted_count_sigma,
        "effective_counts": effective_counts,
        "phi": phi,
        "phi_sigma": phi_sigma,
    }

def sample_uvlf_from_hmf(
    z_obs: float,
    N_mass: int = 3000,
    n_tracks: int = 1000,
    *,
    cosmology: Cosmology,
    base_seed: int,
    quantity: str = "Muv",
    bins: int | np.ndarray = 40,
    logM_min: float = LOGM_MIN,
    logM_max: float = LOGM_MAX,
    z_start_max: float = 50.0,
    n_grid: int = 240,
    sampler: str = "mcbride",
    mah_backend: str = MAH_BACKEND_MCBRIDE,
    tng_mah_cache_path: str | Path | None = None,
    tng_mass_bin_width_dex: float = 0.15,
    tng_min_candidates: int = 5,
    tng_smoothing_myr: float = 0.0,
    tng_time_grid_mode: str = TNG_TIME_GRID_SNAPSHOT,
    thesan_mah_cache_path: str | Path | None = None,
    thesan_mass_bin_width_dex: float = 0.15,
    thesan_min_candidates: int = 5,
    thesan_smoothing_myr: float = 0.0,
    thesan_time_grid_mode: str = THESAN_TIME_GRID_SNAPSHOT,
    enable_time_delay: bool = False,
    pipeline_workers: int | None = None,
    ssp_file: str = DEFAULT_SSP_FILE,
    topheavy_ssp_file: str | None = None,
    topheavy_ssp_metallicity: float | None = DEFAULT_TOPHEAVY_SSP_METALLICITY,
    enable_popiii: bool = False,
    popiii_sfr_parameters: PopIIISFRParameters = DEFAULT_POPIII_SFR_PARAMETERS,
    popiii_ssp_file: str = DEFAULT_POPIII_SSP_FILE,
    imf_mode: str = "canonical",
    imf_transition_parameters: IMFTransitionParameters = DEFAULT_IMF_TRANSITION_PARAMETERS,
    progress_path: str | Path | None = None,
    print_progress: bool = False,
    sfr_model_parameters: SFRModelParameters = DEFAULT_SFR_MODEL_PARAMETERS,
    mass_function_model: str = DEFAULT_MASS_FUNCTION_MODEL,
    hmf_dlog10m: float = DEFAULT_HMF_DLOG10M,
    mzr_metallicity_parameters: MZRBirthMetallicityParameters | None = None,
    regulator_metallicity_parameters: RegulatorMetallicityParameters | None = None,
    burst_scatter_dex: float = 0.0,
    burst_scatter_timescale_myr: float = DEFAULT_BURST_SCATTER_TIMESCALE_MYR,
    burst_scatter_preserve_mean: bool = True,
) -> UVLFSamplingResult:
    """Sample a UVLF by Monte Carlo integration over a halo mass function."""

    hmf_mass_seed = derive_hmf_mass_seed(base_seed, z_obs)
    z_obs = float(z_obs)
    if not isinstance(cosmology, Cosmology):
        raise TypeError("cosmology must be an instance of auroralf.mah.models.Cosmology")
    if not isinstance(popiii_sfr_parameters, PopIIISFRParameters):
        raise TypeError("popiii_sfr_parameters must be an instance of PopIIISFRParameters")
    popiii_lw_array = np.asarray(popiii_sfr_parameters.lw_background_j21, dtype=float)
    if popiii_lw_array.ndim != 0:
        raise ValueError("lw_background_j21 must be scalar, finite, and non-negative")
    popiii_lw_background_j21 = float(popiii_lw_array)
    if not np.isfinite(popiii_lw_background_j21) or popiii_lw_background_j21 < 0.0:
        raise ValueError("lw_background_j21 must be scalar, finite, and non-negative")

    if quantity not in {"Muv", "luminosity"}:
        raise ValueError("quantity must be either 'Muv' or 'luminosity'")
    if N_mass < 1 or n_tracks < 1:
        raise ValueError("N_mass and n_tracks must both be positive")
    if logM_max <= logM_min:
        raise ValueError("logM_max must be larger than logM_min")
    if float(burst_scatter_dex) < 0.0:
        raise ValueError("burst_scatter_dex must be non-negative")
    if float(burst_scatter_timescale_myr) <= 0.0:
        raise ValueError("burst_scatter_timescale_myr must be positive")
    imf_mode = validate_imf_mode(imf_mode)
    mah_backend = validate_mah_backend(mah_backend)
    tng_time_grid_mode = validate_tng_time_grid_mode(tng_time_grid_mode)
    thesan_time_grid_mode = validate_thesan_time_grid_mode(thesan_time_grid_mode)
    if mah_backend == MAH_BACKEND_TNG and tng_mah_cache_path is None:
        raise ValueError("tng_mah_cache_path is required when mah_backend='tng'")
    if mah_backend == MAH_BACKEND_THESAN and thesan_mah_cache_path is None:
        raise ValueError("thesan_mah_cache_path is required when mah_backend='thesan'")
    if float(tng_mass_bin_width_dex) <= 0.0:
        raise ValueError("tng_mass_bin_width_dex must be positive")
    if int(tng_min_candidates) <= 0:
        raise ValueError("tng_min_candidates must be positive")
    if float(tng_smoothing_myr) < 0.0:
        raise ValueError("tng_smoothing_myr must be non-negative")
    if float(thesan_mass_bin_width_dex) <= 0.0:
        raise ValueError("thesan_mass_bin_width_dex must be positive")
    if int(thesan_min_candidates) <= 0:
        raise ValueError("thesan_min_candidates must be positive")
    if float(thesan_smoothing_myr) < 0.0:
        raise ValueError("thesan_smoothing_myr must be non-negative")
    source_count = sum(
        source is not None
        for source in (
            mzr_metallicity_parameters,
            regulator_metallicity_parameters,
        )
    )
    if source_count > 1:
        raise ValueError(
            "provide only one birth metallicity source: "
            "mzr_metallicity_parameters or regulator_metallicity_parameters"
        )
    birth_metallicity_source_enabled = source_count == 1
    if (
        imf_mode != IMF_MODE_CANONICAL
        and imf_transition_parameters.metallicity_topheavy_max_zsun is not None
        and not birth_metallicity_source_enabled
    ):
        raise ValueError(
            "a birth metallicity source must be provided when metallicity_topheavy_max_zsun is set"
        )
    mass_function_model = validate_mass_function_model(mass_function_model)
    if topheavy_ssp_file is None:
        topheavy_ssp_file = DEFAULT_TOPHEAVY_SSP_FILE

    if pipeline_workers is None:
        pipeline_workers = default_worker_count()
    elif isinstance(pipeline_workers, (bool, np.bool_)) or not isinstance(
        pipeline_workers,
        Integral,
    ):
        raise TypeError("pipeline_workers must be an integer non-boolean value")
    pipeline_workers = int(pipeline_workers)
    if pipeline_workers <= 0:
        raise ValueError("pipeline_workers must be positive")
    progress_file = None if progress_path is None else Path(progress_path).expanduser().resolve()
    if progress_file is not None:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_text = _write_progress(progress_file, completed=0, total=N_mass, elapsed_seconds=0.0)
        if print_progress:
            print(progress_text.strip(), flush=True)
    rng = np.random.default_rng(hmf_mass_seed)

    t0 = time.perf_counter()
    logMh = rng.uniform(logM_min, logM_max, size=N_mass)
    Mh = np.power(10.0, logMh)
    atomic_cooling_mass_msun = float(
        compute_atomic_cooling_mass_msun(z_obs, cosmology=cosmology)
    )
    popiii_minimum_mass_msun = float(
        compute_popiii_lw_minimum_mass_msun(z_obs, lw_background_j21=popiii_lw_background_j21)
    )
    stellar_channel_by_mass = np.asarray(
        classify_halo_stellar_channels(
            Mh,
            z_obs=z_obs,
            cosmology=cosmology,
            lw_background_j21=popiii_lw_background_j21,
        ),
        dtype=f"<U{max(len(channel) for channel in STELLAR_CHANNELS)}",
    )
    dndm = np.asarray(
        compute_halo_mass_function_dndm(
            Mh,
            z_obs,
            cosmology=cosmology,
            mass_function_model=mass_function_model,
            hmf_dlog10m=hmf_dlog10m,
        ),
        dtype=float,
    )
    dndlogM = Mh * np.log(10.0) * dndm
    mass_weight = (logM_max - logM_min) * dndlogM / N_mass

    total_samples = N_mass * n_tracks
    sample_logMh = np.empty(total_samples, dtype=float)
    sample_Mh = np.empty(total_samples, dtype=float)
    sample_mass_weight = np.empty(total_samples, dtype=float)
    sample_track_index = np.empty(total_samples, dtype=int)
    sample_luminosity = np.empty(total_samples, dtype=float)
    sample_sfr = np.empty(total_samples, dtype=float)
    sample_topheavy_light_fraction = np.empty(total_samples, dtype=float)
    sample_popiii_luminosity = np.empty(total_samples, dtype=float)
    sample_popiii_light_fraction = np.empty(total_samples, dtype=float)
    sample_popiii_sfr = np.empty(total_samples, dtype=float)
    sample_stellar_channel = np.empty(total_samples, dtype=stellar_channel_by_mass.dtype)
    sample_atomic_cooling_mass_msun = np.empty(total_samples, dtype=float)
    sample_popiii_minimum_mass_msun = np.empty(total_samples, dtype=float)
    sample_sample_weight = np.empty(total_samples, dtype=float)
    sample_Muv = np.empty(total_samples, dtype=float)
    per_mass_pipeline_seconds = np.empty(N_mass, dtype=float)
    topheavy_source_count_by_mass = np.empty(N_mass, dtype=np.int64)
    starforming_source_count_by_mass = np.empty(N_mass, dtype=np.int64)
    popiii_source_count_by_mass = np.empty(N_mass, dtype=np.int64)
    active_source_count_by_mass = np.empty(N_mass, dtype=np.int64)
    final_gas_metallicity_zsun_median_by_mass = np.full(N_mass, np.nan, dtype=float)
    birth_metallicity_zsun_starforming_median_by_mass = np.full(N_mass, np.nan, dtype=float)
    pipeline_random_seeds_by_mass = [
        derive_pipeline_random_seeds(
            base_seed,
            redshift=float(z_obs),
            mass_index=mass_index,
        )
        for mass_index in range(N_mass)
    ]

    progress_stride = max(1, N_mass // 100)
    tasks = [
        _MassSampleTask(
            mass_index=mass_index,
            log_mass=float(log_mass),
            mass=float(mass),
            weight=float(weight),
            z_obs=float(z_obs),
            n_tracks=int(n_tracks),
            z_start_max=float(z_start_max),
            n_grid=int(n_grid),
            sampler=sampler,
            mah_backend=mah_backend,
            tng_mah_cache_path=None if tng_mah_cache_path is None else str(tng_mah_cache_path),
            tng_mass_bin_width_dex=float(tng_mass_bin_width_dex),
            tng_min_candidates=int(tng_min_candidates),
            tng_smoothing_myr=float(tng_smoothing_myr),
            tng_time_grid_mode=str(tng_time_grid_mode),
            thesan_mah_cache_path=None if thesan_mah_cache_path is None else str(thesan_mah_cache_path),
            thesan_mass_bin_width_dex=float(thesan_mass_bin_width_dex),
            thesan_min_candidates=int(thesan_min_candidates),
            thesan_smoothing_myr=float(thesan_smoothing_myr),
            thesan_time_grid_mode=str(thesan_time_grid_mode),
            enable_time_delay=bool(enable_time_delay),
            ssp_file=ssp_file,
            topheavy_ssp_file=str(topheavy_ssp_file),
            topheavy_ssp_metallicity=topheavy_ssp_metallicity,
            imf_mode=imf_mode,
            imf_transition_parameters=imf_transition_parameters,
            random_seeds=pipeline_random_seeds_by_mass[mass_index],
            sfr_model_parameters=sfr_model_parameters,
            mzr_metallicity_parameters=mzr_metallicity_parameters,
            regulator_metallicity_parameters=regulator_metallicity_parameters,
            burst_scatter_dex=float(burst_scatter_dex),
            burst_scatter_timescale_myr=float(burst_scatter_timescale_myr),
            burst_scatter_preserve_mean=bool(burst_scatter_preserve_mean),
            enable_popiii=bool(enable_popiii),
            popiii_sfr_parameters=popiii_sfr_parameters,
            popiii_ssp_file=str(popiii_ssp_file),
            cosmology=cosmology,
        )
        for mass_index, (log_mass, mass, weight) in enumerate(zip(logMh, Mh, mass_weight, strict=True))
    ]

    executor_context = (
        nullcontext(None)
        if pipeline_workers == 1
        else ProcessPoolExecutor(
            max_workers=pipeline_workers,
            mp_context=mp.get_context("spawn"),
        )
    )
    with executor_context as executor:
        if executor is None:
            results_iter = (_run_single_mass_sample(task) for task in tasks)
        else:
            futures = [executor.submit(_run_single_mass_sample, task) for task in tasks]
            results_iter = (future.result() for future in as_completed(futures))

        completed = 0
        for (
            mass_index,
            log_mass,
            luminosity,
            sfr,
            topheavy_light_fraction,
            popiii_luminosity,
            popiii_light_fraction,
            popiii_sfr,
            duration,
            topheavy_source_count,
            starforming_source_count,
            popiii_source_count,
            active_source_count,
            final_gas_metallicity_zsun_median,
            birth_metallicity_zsun_starforming_median,
        ) in results_iter:
            if luminosity.size != n_tracks:
                raise RuntimeError("run_halo_uv_pipeline returned an unexpected number of luminosity samples")
            if sfr.size != n_tracks:
                raise RuntimeError("run_halo_uv_pipeline returned an unexpected number of SFR samples")
            if topheavy_light_fraction.size != n_tracks:
                raise RuntimeError("run_halo_uv_pipeline returned an unexpected number of top-heavy fractions")
            if popiii_luminosity.size != n_tracks:
                raise RuntimeError("run_halo_uv_pipeline returned an unexpected number of Pop III luminosity samples")
            if popiii_light_fraction.size != n_tracks:
                raise RuntimeError("run_halo_uv_pipeline returned an unexpected number of Pop III fractions")
            if popiii_sfr.size != n_tracks:
                raise RuntimeError("run_halo_uv_pipeline returned an unexpected number of Pop III SFR samples")

            start = mass_index * n_tracks
            stop = start + n_tracks
            sample_logMh[start:stop] = log_mass
            sample_Mh[start:stop] = Mh[mass_index]
            sample_mass_weight[start:stop] = mass_weight[mass_index]
            sample_track_index[start:stop] = np.arange(n_tracks, dtype=int)
            sample_luminosity[start:stop] = luminosity
            sample_sfr[start:stop] = sfr
            sample_topheavy_light_fraction[start:stop] = topheavy_light_fraction
            sample_popiii_luminosity[start:stop] = popiii_luminosity
            sample_popiii_light_fraction[start:stop] = popiii_light_fraction
            sample_popiii_sfr[start:stop] = popiii_sfr
            sample_stellar_channel[start:stop] = stellar_channel_by_mass[mass_index]
            sample_atomic_cooling_mass_msun[start:stop] = atomic_cooling_mass_msun
            sample_popiii_minimum_mass_msun[start:stop] = popiii_minimum_mass_msun
            sample_sample_weight[start:stop] = mass_weight[mass_index] / n_tracks
            sample_Muv[start:stop] = np.asarray(uv_luminosity_to_muv(luminosity), dtype=float)
            per_mass_pipeline_seconds[mass_index] = duration
            topheavy_source_count_by_mass[mass_index] = topheavy_source_count
            starforming_source_count_by_mass[mass_index] = starforming_source_count
            popiii_source_count_by_mass[mass_index] = popiii_source_count
            active_source_count_by_mass[mass_index] = active_source_count
            final_gas_metallicity_zsun_median_by_mass[mass_index] = final_gas_metallicity_zsun_median
            birth_metallicity_zsun_starforming_median_by_mass[mass_index] = (
                birth_metallicity_zsun_starforming_median
            )

            completed += 1
            if progress_file is not None and (completed == N_mass or completed % progress_stride == 0):
                progress_text = _write_progress(
                    progress_file,
                    completed=completed,
                    total=N_mass,
                    elapsed_seconds=time.perf_counter() - t0,
                )
                if print_progress:
                    print(progress_text.strip(), flush=True)

    if quantity == "luminosity":
        histogram_values = sample_luminosity
    else:
        histogram_values = sample_Muv

    bin_edges = _resolve_bin_edges(histogram_values, quantity=quantity, bins=bins)
    uvlf = _build_sample_histogram(
        histogram_values, sample_sample_weight, quantity=quantity, bin_edges=bin_edges,
    )
    total_seconds = time.perf_counter() - t0

    samples = {
        "logMh": sample_logMh,
        "Mh": sample_Mh,
        "mass_weight": sample_mass_weight,
        "track_index": sample_track_index,
        "luminosity": sample_luminosity,
        "sfr": sample_sfr,
        "topheavy_light_fraction": sample_topheavy_light_fraction,
        "popiii_luminosity": sample_popiii_luminosity,
        "popiii_light_fraction": sample_popiii_light_fraction,
        "popiii_sfr": sample_popiii_sfr,
        "stellar_channel": sample_stellar_channel,
        "atomic_cooling_mass_msun": sample_atomic_cooling_mass_msun,
        "popiii_minimum_mass_msun": sample_popiii_minimum_mass_msun,
        "Muv": sample_Muv,
        "sample_weight": sample_sample_weight,
    }
    metadata = {
        "z_obs": z_obs,
        "N_mass": N_mass,
        "n_tracks": n_tracks,
        "base_seed": base_seed,
        "hmf_mass_seed": hmf_mass_seed,
        "pipeline_random_seeds_by_mass": {
            "mah": np.asarray([seeds.mah for seeds in pipeline_random_seeds_by_mass], dtype=np.uint64),
            "metallicity": np.asarray(
                [seeds.metallicity for seeds in pipeline_random_seeds_by_mass],
                dtype=np.uint64,
            ),
            "burst": np.asarray([seeds.burst for seeds in pipeline_random_seeds_by_mass], dtype=np.uint64),
        },
        "logM_min": logM_min,
        "logM_max": logM_max,
        "halo_mass_by_mass": Mh,
        "log_halo_mass_by_mass": logMh,
        "mass_weight_by_mass": mass_weight,
        "atomic_cooling_temperature_k": ATOMIC_COOLING_TEMPERATURE_K,
        "atomic_cooling_mu": ATOMIC_COOLING_MU,
        "atomic_cooling_mass_msun": atomic_cooling_mass_msun,
        "popiii_molecular_cooling_m0_normalization_msun": POPIII_MOLECULAR_COOLING_M0_NORMALIZATION_MSUN,
        "popiii_molecular_cooling_redshift_exponent": POPIII_MOLECULAR_COOLING_REDSHIFT_EXPONENT,
        "popiii_lw_feedback_coefficient": POPIII_LW_FEEDBACK_COEFFICIENT,
        "popiii_lw_feedback_exponent": POPIII_LW_FEEDBACK_EXPONENT,
        "lw_background_j21": popiii_lw_background_j21,
        "popiii_minimum_mass_msun": popiii_minimum_mass_msun,
        "stellar_channel_by_mass": stellar_channel_by_mass,
        "stellar_channel_counts": {
            channel: int(np.count_nonzero(stellar_channel_by_mass == channel))
            for channel in STELLAR_CHANNELS
        },
        "mass_function_model": mass_function_model,
        "hmf_dlog10m": hmf_dlog10m,
        "mass_function_parameters": {
            "ns": MASS_FUNCTION_NS,
            "sigma8": MASS_FUNCTION_SIGMA8,
            "h": cosmology.h0_km_s_mpc / 100.0,
            "h0_km_s_mpc": cosmology.h0_km_s_mpc,
            "omega_m": cosmology.omega_m,
            "omega_b": cosmology.omega_b,
            "omega_lambda": cosmology.omega_lambda,
        },
        "pipeline_workers": pipeline_workers,
        "mah_backend": mah_backend,
        "sampler": sampler,
        "tng_mah_cache_path": None if tng_mah_cache_path is None else str(Path(tng_mah_cache_path).expanduser().resolve()),
        "tng_mass_bin_width_dex": None if mah_backend != MAH_BACKEND_TNG else float(tng_mass_bin_width_dex),
        "tng_min_candidates": None if mah_backend != MAH_BACKEND_TNG else int(tng_min_candidates),
        "tng_smoothing_myr": None if mah_backend != MAH_BACKEND_TNG else float(tng_smoothing_myr),
        "tng_time_grid_mode": None if mah_backend != MAH_BACKEND_TNG else str(tng_time_grid_mode),
        "thesan_mah_cache_path": None
        if thesan_mah_cache_path is None
        else str(Path(thesan_mah_cache_path).expanduser().resolve()),
        "thesan_mass_bin_width_dex": None
        if mah_backend != MAH_BACKEND_THESAN
        else float(thesan_mass_bin_width_dex),
        "thesan_min_candidates": None if mah_backend != MAH_BACKEND_THESAN else int(thesan_min_candidates),
        "thesan_smoothing_myr": None if mah_backend != MAH_BACKEND_THESAN else float(thesan_smoothing_myr),
        "thesan_time_grid_mode": None if mah_backend != MAH_BACKEND_THESAN else str(thesan_time_grid_mode),
        "quantity": quantity,
        "ssp_file": ssp_file,
        "topheavy_ssp_file": topheavy_ssp_file,
        "topheavy_ssp_metallicity": topheavy_ssp_metallicity,
        "popiii_enabled": bool(enable_popiii),
        "popiii_ssp_file": str(popiii_ssp_file),
        "popiii_sfr_parameters": popiii_sfr_parameters.as_metadata(),
        "imf_mode": imf_mode,
        "imf_transition_parameters": {
            "z_topheavy_min": float(imf_transition_parameters.z_topheavy_min),
            "source_redshift_gate_enabled": bool(imf_transition_parameters.source_redshift_gate_enabled),
            "growth_time_threshold_myr": float(imf_transition_parameters.growth_time_threshold_myr),
            "metallicity_topheavy_max_zsun": None
            if imf_transition_parameters.metallicity_topheavy_max_zsun is None
            else float(imf_transition_parameters.metallicity_topheavy_max_zsun),
        },
        "enable_time_delay": enable_time_delay,
        "burst_scatter_enabled": float(burst_scatter_dex) > 0.0,
        "burst_scatter_dex": float(burst_scatter_dex),
        "burst_scatter_timescale_myr": float(burst_scatter_timescale_myr),
        "burst_scatter_preserve_mean": bool(burst_scatter_preserve_mean),
        "burst_scatter_mass_conserving": bool(burst_scatter_preserve_mean),
        "sfr_model_parameters": {
            "epsilon_0": sfr_model_parameters.epsilon_0,
            "characteristic_mass": sfr_model_parameters.characteristic_mass,
            "beta_star": sfr_model_parameters.beta_star,
            "gamma_star": sfr_model_parameters.gamma_star,
        },
        "sampling_seconds": total_seconds,
        "per_mass_pipeline_seconds": per_mass_pipeline_seconds,
        "topheavy_source_count_by_mass": topheavy_source_count_by_mass,
        "starforming_source_count_by_mass": starforming_source_count_by_mass,
        "popiii_source_count_by_mass": popiii_source_count_by_mass,
        "active_source_count_by_mass": active_source_count_by_mass,
        "topheavy_source_fraction": float(np.sum(topheavy_source_count_by_mass) / np.sum(starforming_source_count_by_mass))
        if np.sum(starforming_source_count_by_mass) > 0
        else 0.0,
        "topheavy_light_fraction_median": float(
            np.median(sample_topheavy_light_fraction[np.isfinite(sample_topheavy_light_fraction)])
        )
        if np.any(np.isfinite(sample_topheavy_light_fraction))
        else 0.0,
        "popiii_source_fraction": float(np.sum(popiii_source_count_by_mass) / np.sum(active_source_count_by_mass))
        if np.sum(active_source_count_by_mass) > 0
        else 0.0,
        "popiii_light_fraction_median": float(
            np.median(sample_popiii_light_fraction[np.isfinite(sample_popiii_light_fraction)])
        )
        if np.any(np.isfinite(sample_popiii_light_fraction))
        else 0.0,
        "sfrd_msun_yr_mpc3": float(np.sum(sample_sfr * sample_sample_weight)),
        "popiii_sfrd_msun_yr_mpc3": float(np.sum(sample_popiii_sfr * sample_sample_weight)),
        "mzr_metallicity_enabled": mzr_metallicity_parameters is not None,
        "regulator_metallicity_enabled": regulator_metallicity_parameters is not None,
        "metallicity_source": "mzr"
        if mzr_metallicity_parameters is not None
        else "regulator"
        if regulator_metallicity_parameters is not None
        else "none",
        "mzr_metallicity_parameters": mzr_metallicity_parameters.as_metadata()
        if mzr_metallicity_parameters is not None
        else None,
        "regulator_metallicity_parameters": regulator_metallicity_parameters.as_metadata()
        if regulator_metallicity_parameters is not None
        else None,
        "final_gas_metallicity_zsun_median_by_mass": final_gas_metallicity_zsun_median_by_mass,
        "birth_metallicity_zsun_starforming_median_by_mass": birth_metallicity_zsun_starforming_median_by_mass,
        "final_gas_metallicity_zsun_median": _finite_median_or_none(final_gas_metallicity_zsun_median_by_mass)
        if regulator_metallicity_parameters is not None
        else None,
        "birth_metallicity_zsun_starforming_median": _finite_median_or_none(
            birth_metallicity_zsun_starforming_median_by_mass
        )
        if birth_metallicity_source_enabled
        else None,
        "progress_path": None if progress_file is None else str(progress_file),
    }
    return UVLFSamplingResult(samples=samples, uvlf=uvlf, metadata=metadata)
