from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from auroralf.constants import YEARS_PER_GYR
from auroralf.seeding import PipelineRandomSeeds
from auroralf.chemistry import (
    MZRBirthMetallicityParameters,
    RegulatorMetallicityParameters,
    compute_mzr_birth_metallicity,
    compute_regulator_metallicity,
)
from auroralf.cooling import compute_popiii_lw_minimum_mass_msun
from auroralf.mah import (
    MAH_BACKEND_MCBRIDE,
    MAH_BACKEND_THESAN,
    MAH_BACKEND_TNG,
    THESAN_TIME_GRID_SNAPSHOT,
    THESAN_TIME_GRID_UNIFORM_IN_T,
    TNG_TIME_GRID_SNAPSHOT,
    TNG_TIME_GRID_UNIFORM_IN_T,
    Cosmology,
    HaloHistoryResult,
    generate_halo_histories,
    generate_thesan_halo_histories,
    generate_tng_halo_histories,
    validate_mah_backend,
)
from auroralf.sfr import (
    DEFAULT_POPIII_SFR_PARAMETERS,
    DEFAULT_SFR_MODEL_PARAMETERS,
    EXTENDED_BURST_LOOKBACK_MAX_MYR,
    PopIIISFRParameters,
    SFRModelParameters,
    compute_popiii_sfr_from_grids,
    compute_sfr_from_tracks,
)
from auroralf.ssp import (
    DEFAULT_POPIII_UV_SSP_FILE,
    SSP_UV_LOOKBACK_MAX_MYR,
    compute_final_ssp_observable_from_sfr_grid,
    load_popiii_uv_luminosity_table,
    load_uv1600_table,
)
from auroralf.ssp.convolution import _reject_boolean_scalar, _reject_boolean_values
from .imf import (
    DEFAULT_CANONICAL_SSP_FILE,
    DEFAULT_IMF_TRANSITION_PARAMETERS,
    DEFAULT_MILD_TOPHEAVY_SSP_FILE,
    DEFAULT_MILD_TOPHEAVY_SSP_METALLICITY,
    IMF_MODE_CANONICAL,
    IMFTransitionParameters,
    compute_topheavy_source_flags,
    resolve_ssp_path,
    validate_imf_mode,
)


DEFAULT_SSP_FILE = DEFAULT_CANONICAL_SSP_FILE
DEFAULT_TOPHEAVY_SSP_FILE = DEFAULT_MILD_TOPHEAVY_SSP_FILE
DEFAULT_TOPHEAVY_SSP_METALLICITY = DEFAULT_MILD_TOPHEAVY_SSP_METALLICITY
DEFAULT_POPIII_SSP_FILE = DEFAULT_POPIII_UV_SSP_FILE
DEFAULT_BURST_SCATTER_TIMESCALE_MYR = 20.0


from .pipeline_models import (
    HaloModeEvaluation,
    HaloUVPipelineResult,
    LoadedSSPKernels,
    SharedHaloBatch,
)


def _build_astropy_cosmology(cosmology: Cosmology) -> FlatLambdaCDM:
    return FlatLambdaCDM(H0=cosmology.h0_km_s_mpc, Om0=cosmology.omega_m, Ob0=cosmology.omega_b)


def compute_uv_luminosities_parallel(
    t_grid: np.ndarray,
    mh_grid: np.ndarray,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    ssp_age_grid: np.ndarray,
    ssp_luv_grid: np.ndarray,
    n_workers: int,
    ssp_lookback_max_myr: float,
) -> np.ndarray:
    """Compatibility adapter for the legacy Gyr-age UV convolution API.

    The shared SSP kernel now evaluates every halo row in one vectorized call, so
    the retained ``n_workers`` argument is validated but no process pool is needed.
    """

    for name, values in (
        ("t_grid", t_grid),
        ("mh_grid", mh_grid),
        ("sfr_grid", sfr_grid),
        ("ssp_age_grid", ssp_age_grid),
        ("ssp_luv_grid", ssp_luv_grid),
    ):
        _reject_boolean_values(name, values)
    _reject_boolean_scalar("ssp_lookback_max_myr", ssp_lookback_max_myr)
    if isinstance(n_workers, (bool, np.bool_)) or not isinstance(
        n_workers,
        (int, np.integer),
    ):
        raise ValueError("n_workers must be a positive non-boolean integer")
    if int(n_workers) <= 0:
        raise ValueError("n_workers must be a positive non-boolean integer")

    time = np.asarray(t_grid, dtype=float)
    halo_mass = np.asarray(mh_grid, dtype=float)
    sfr = np.asarray(sfr_grid, dtype=float)
    active = np.asarray(active_grid)
    if sfr.ndim != 2 or sfr.size == 0:
        raise ValueError("mh_grid, sfr_grid, and active_grid must be non-empty 2D arrays")
    if halo_mass.shape != sfr.shape or active.shape != sfr.shape:
        raise ValueError("mh_grid, sfr_grid, and active_grid must have identical shapes")
    if active.dtype != np.dtype(bool):
        raise ValueError("active_grid must have boolean dtype")
    if time.ndim == 1 and time.size == sfr.shape[1]:
        time_rows = np.broadcast_to(time, sfr.shape)
    elif time.ndim == 2 and time.shape == sfr.shape:
        time_rows = time
    else:
        raise ValueError("t_grid must be shared 1D or match the 2D SFR grid")
    time_deltas = np.diff(time_rows, axis=1)
    if not np.all(np.isfinite(time_rows)) or not np.all(np.isfinite(time_deltas)):
        raise ValueError("t_grid must contain only finite values")
    if np.any(time_deltas <= 0.0):
        raise ValueError("each shared or row t_grid must be strictly increasing")

    legacy_ssp_age_gyr = np.asarray(ssp_age_grid, dtype=float)
    ssp_luv = np.asarray(ssp_luv_grid, dtype=float)
    if legacy_ssp_age_gyr.ndim != 1 or ssp_luv.ndim != 1:
        raise ValueError("legacy SSP age and luminosity grids must be 1D")
    if legacy_ssp_age_gyr.size != ssp_luv.size:
        raise ValueError("legacy SSP age and luminosity grids must have the same length")
    order = np.argsort(legacy_ssp_age_gyr, kind="stable")
    return compute_final_ssp_observable_from_sfr_grid(
        t_grid_gyr=time_rows,
        sfr_grid=sfr,
        active_grid=active,
        ssp_age_myr=legacy_ssp_age_gyr[order] * 1.0e3,
        ssp_observable_per_msun=ssp_luv[order],
        lookback_max_myr=ssp_lookback_max_myr,
        time_unit_in_years=YEARS_PER_GYR,
    )


def default_worker_count() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


from .burst_scatter import (
    _apply_burst_scatter_to_sfr_grid as _apply_burst_scatter_core,
    _draw_burst_multiplier_for_segments,
)


def _apply_burst_scatter_to_sfr_grid(
    *,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    t_grid: np.ndarray,
    scatter_dex: float,
    correlation_timescale_myr: float,
    random_seed: int | None,
    preserve_mean: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility boundary for the archived burst-scatter diagnostic."""

    return _apply_burst_scatter_core(
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        t_grid=t_grid,
        scatter_dex=scatter_dex,
        correlation_timescale_myr=correlation_timescale_myr,
        random_seed=random_seed,
        preserve_mean=preserve_mean,
        draw_multiplier=_draw_burst_multiplier_for_segments,
    )


def _resolve_regular_time_grid(t_grid: np.ndarray) -> np.ndarray | None:
    if t_grid.ndim != 2 or t_grid.shape[0] == 0:
        return None
    time_row = np.asarray(t_grid[0], dtype=float)
    if not np.all(np.isfinite(time_row)):
        return None
    if not np.allclose(t_grid, time_row[None, :], rtol=0.0, atol=0.0):
        return None
    return time_row


def _compute_final_uv_luminosity_components_vectorized(
    t_grid: np.ndarray,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    topheavy_source_flag_grid: np.ndarray,
    ssp_age_grid: np.ndarray,
    ssp_luv_grid: np.ndarray,
    topheavy_ssp_age_grid: np.ndarray | None,
    topheavy_ssp_luv_grid: np.ndarray | None,
    ssp_lookback_max_myr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convolve canonical and top-heavy source masks with SSP ages in Myr."""

    for name, values in (
        ("t_grid", t_grid),
        ("sfr_grid", sfr_grid),
        ("ssp_age_grid", ssp_age_grid),
        ("ssp_luv_grid", ssp_luv_grid),
    ):
        _reject_boolean_values(name, values)
    if topheavy_ssp_age_grid is not None:
        _reject_boolean_values("topheavy_ssp_age_grid", topheavy_ssp_age_grid)
    if topheavy_ssp_luv_grid is not None:
        _reject_boolean_values("topheavy_ssp_luv_grid", topheavy_ssp_luv_grid)
    _reject_boolean_scalar("ssp_lookback_max_myr", ssp_lookback_max_myr)

    time = np.asarray(t_grid, dtype=float)
    sfr = np.asarray(sfr_grid, dtype=float)
    canonical_ssp_age = np.asarray(ssp_age_grid, dtype=float)
    canonical_ssp_luv = np.asarray(ssp_luv_grid, dtype=float)
    topheavy_ssp_age = (
        None
        if topheavy_ssp_age_grid is None
        else np.asarray(topheavy_ssp_age_grid, dtype=float)
    )
    topheavy_ssp_luv = (
        None
        if topheavy_ssp_luv_grid is None
        else np.asarray(topheavy_ssp_luv_grid, dtype=float)
    )

    time_row = _resolve_regular_time_grid(time)
    if time_row is None:
        raise ValueError("vectorized final UV convolution requires a shared regular time grid")
    del time_row
    active = np.asarray(active_grid)
    topheavy_source = np.asarray(topheavy_source_flag_grid)
    if active.dtype != np.dtype(bool) or topheavy_source.dtype != np.dtype(bool):
        raise ValueError("active_grid and topheavy_source_flag_grid must have boolean dtype")
    if time.shape != sfr.shape or time.shape != active.shape or time.shape != topheavy_source.shape:
        raise ValueError(
            "t_grid, sfr_grid, active_grid, and topheavy_source_flag_grid must have identical shapes"
        )
    if (topheavy_ssp_age is None) != (topheavy_ssp_luv is None):
        raise ValueError("top-heavy SSP age and luminosity grids must be provided together")

    if topheavy_ssp_age is None:
        canonical_source = active
    else:
        canonical_source = active & ~topheavy_source
    canonical_result = compute_final_ssp_observable_from_sfr_grid(
        t_grid_gyr=time,
        sfr_grid=sfr,
        active_grid=canonical_source,
        ssp_age_myr=canonical_ssp_age,
        ssp_observable_per_msun=canonical_ssp_luv,
        lookback_max_myr=ssp_lookback_max_myr,
        time_unit_in_years=YEARS_PER_GYR,
    )

    if topheavy_ssp_age is None or topheavy_ssp_luv is None:
        topheavy_result = np.zeros_like(canonical_result, dtype=float)
    else:
        topheavy_result = compute_final_ssp_observable_from_sfr_grid(
            t_grid_gyr=time,
            sfr_grid=sfr,
            active_grid=active & topheavy_source,
            ssp_age_myr=topheavy_ssp_age,
            ssp_observable_per_msun=topheavy_ssp_luv,
            lookback_max_myr=ssp_lookback_max_myr,
            time_unit_in_years=YEARS_PER_GYR,
        )
    return canonical_result, topheavy_result


def prepare_shared_halo_batch(
    n_tracks: int,
    z_final: float,
    Mh_final: float,
    *,
    cosmology: Cosmology,
    random_seeds: PipelineRandomSeeds,
    mass_index: int = 0,
    z_start_max: float = 50.0,
    n_grid: int = 240,
    enable_popiii: bool = False,
    popiii_sfr_parameters: PopIIISFRParameters = DEFAULT_POPIII_SFR_PARAMETERS,
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
    burst_lookback_max_myr: float = EXTENDED_BURST_LOOKBACK_MAX_MYR,
    sfr_model_parameters: SFRModelParameters = DEFAULT_SFR_MODEL_PARAMETERS,
    mzr_metallicity_parameters: MZRBirthMetallicityParameters | None = None,
    regulator_metallicity_parameters: RegulatorMetallicityParameters | None = None,
    burst_scatter_dex: float = 0.0,
    burst_scatter_timescale_myr: float = DEFAULT_BURST_SCATTER_TIMESCALE_MYR,
    burst_scatter_preserve_mean: bool = True,
    _mutable_result_sources: dict[str, object] | None = None,
) -> SharedHaloBatch:
    """Generate mode-independent MAH, SFR, Pop III, and metallicity grids once."""

    del mass_index
    mah_backend = validate_mah_backend(mah_backend)
    if type(cosmology) is not Cosmology:
        raise TypeError("cosmology must be exactly Cosmology")
    if type(random_seeds) is not PipelineRandomSeeds:
        raise TypeError("random_seeds must be exactly PipelineRandomSeeds")
    if int(n_grid) < 2:
        raise ValueError("n_grid must be at least 2")
    source_count = sum(
        source is not None
        for source in (mzr_metallicity_parameters, regulator_metallicity_parameters)
    )
    if source_count > 1:
        raise ValueError("provide only one birth metallicity source")
    astro = _build_astropy_cosmology(cosmology)
    t_start_gyr = float(astro.age(z_start_max).value)
    t_end_gyr = float(astro.age(z_final).value)
    dt_gyr = (t_end_gyr - t_start_gyr) / float(int(n_grid) - 1)

    started = time.perf_counter()
    if mah_backend == MAH_BACKEND_MCBRIDE:
        mah_mass_floor = None
        if enable_popiii:
            mah_mass_floor = lambda redshift: compute_popiii_lw_minimum_mass_msun(
                redshift,
                lw_background_j21=float(popiii_sfr_parameters.lw_background_j21),
            )
        histories = generate_halo_histories(
            n_tracks=n_tracks,
            z_final=z_final,
            Mh_final=Mh_final,
            z_start_max=z_start_max,
            M_min=mah_mass_floor,
            cosmology=cosmology,
            random_seed=random_seeds.mah,
            time_grid_mode="uniform_in_t",
            dt=dt_gyr,
            store_inactive_history=True,
            sampler=sampler,
        )
    elif mah_backend == MAH_BACKEND_TNG:
        if tng_mah_cache_path is None:
            raise ValueError("tng_mah_cache_path is required when mah_backend='tng'")
        histories = generate_tng_halo_histories(
            n_tracks=n_tracks,
            z_final=z_final,
            Mh_final=Mh_final,
            cosmology=cosmology,
            cache_path=tng_mah_cache_path,
            z_start_max=z_start_max,
            mass_bin_width_dex=tng_mass_bin_width_dex,
            min_candidates=tng_min_candidates,
            smoothing_myr=tng_smoothing_myr,
            random_seed=random_seeds.mah,
            time_grid_mode=tng_time_grid_mode,
            target_n_grid=int(n_grid)
            if str(tng_time_grid_mode).strip().lower() == TNG_TIME_GRID_UNIFORM_IN_T
            else None,
        )
    elif mah_backend == MAH_BACKEND_THESAN:
        if thesan_mah_cache_path is None:
            raise ValueError("thesan_mah_cache_path is required when mah_backend='thesan'")
        histories = generate_thesan_halo_histories(
            n_tracks=n_tracks,
            z_final=z_final,
            Mh_final=Mh_final,
            cosmology=cosmology,
            cache_path=thesan_mah_cache_path,
            z_start_max=z_start_max,
            mass_bin_width_dex=thesan_mass_bin_width_dex,
            min_candidates=thesan_min_candidates,
            smoothing_myr=thesan_smoothing_myr,
            random_seed=random_seeds.mah,
            time_grid_mode=thesan_time_grid_mode,
            target_n_grid=int(n_grid)
            if str(thesan_time_grid_mode).strip().lower() == THESAN_TIME_GRID_UNIFORM_IN_T
            else None,
        )
    else:  # pragma: no cover
        raise RuntimeError(f"unsupported mah_backend after validation: {mah_backend}")
    after_mah = time.perf_counter()

    redshift_grid = np.unique(np.asarray(histories.tracks["z"], dtype=float))[::-1]
    sfr_tracks = compute_sfr_from_tracks(
        histories.tracks,
        cosmology=cosmology,
        enable_time_delay=enable_time_delay,
        burst_lookback_max_myr=burst_lookback_max_myr,
        model_parameters=sfr_model_parameters,
    )
    halo_ids = np.asarray(sfr_tracks["halo_id"], dtype=int)
    n_halos = np.unique(halo_ids).size
    steps_per_halo = redshift_grid.size
    shape = (n_halos, steps_per_halo)
    t_grid = np.asarray(sfr_tracks["t_gyr"], dtype=float).reshape(shape)
    mh_grid = np.asarray(sfr_tracks["Mh"], dtype=float).reshape(shape)
    dmhdt_sfr_grid = np.asarray(sfr_tracks["dMh_dt_sfr"], dtype=float).reshape(shape)
    sfr_grid = np.asarray(sfr_tracks["SFR"], dtype=float).reshape(shape)
    active_grid = np.asarray(sfr_tracks["active_flag"], dtype=bool).reshape(shape)
    z_grid = np.asarray(sfr_tracks["z"], dtype=float).reshape(shape)
    sfr_grid, burst_multiplier = _apply_burst_scatter_to_sfr_grid(
        sfr_grid=sfr_grid,
        active_grid=active_grid,
        t_grid=t_grid,
        scatter_dex=float(burst_scatter_dex),
        correlation_timescale_myr=float(burst_scatter_timescale_myr),
        random_seed=random_seeds.burst,
        preserve_mean=bool(burst_scatter_preserve_mean),
    )
    sfr_tracks["SFR"] = sfr_grid.reshape(-1)

    if enable_popiii:
        popiii = compute_popiii_sfr_from_grids(
            mh_grid=mh_grid,
            dmhdt_sfr_grid=dmhdt_sfr_grid,
            z_grid=z_grid,
            active_grid=active_grid,
            cosmology=cosmology,
            parameters=popiii_sfr_parameters,
        )
        popiii_sfr = np.asarray(popiii.sfr_grid, dtype=float)
        popiii_fstar = np.asarray(popiii.fstar_grid, dtype=float)
        popiii_duty = np.asarray(popiii.duty_cycle_grid, dtype=float)
        popiii_lower = np.asarray(popiii.lower_mass_msun_grid, dtype=float)
        popiii_upper = np.asarray(popiii.upper_mass_msun_grid, dtype=float)
        popiii_source = active_grid & np.isfinite(popiii_sfr) & (popiii_sfr > 0.0)
    else:
        popiii_sfr = np.zeros_like(sfr_grid)
        popiii_source = np.zeros_like(active_grid)
        popiii_fstar = np.zeros_like(sfr_grid)
        popiii_duty = np.zeros_like(sfr_grid)
        popiii_lower = np.full_like(sfr_grid, np.nan)
        popiii_upper = np.full_like(sfr_grid, np.nan)
    sfr_tracks["SFR_popiii"] = popiii_sfr.reshape(-1)
    sfr_tracks["fstar_popiii"] = popiii_fstar.reshape(-1)
    sfr_tracks["popiii_duty_cycle"] = popiii_duty.reshape(-1)
    sfr_tracks["popiii_lower_mass_msun"] = popiii_lower.reshape(-1)
    sfr_tracks["popiii_upper_mass_msun"] = popiii_upper.reshape(-1)
    sfr_tracks["popiii_source_flag"] = popiii_source.reshape(-1)
    starforming = active_grid & np.isfinite(sfr_grid) & (sfr_grid > 0.0)

    birth_metallicity = None
    gas_metallicity = None
    metal_mass = None
    gas_mass = None
    metallicity_source = "none"
    if mzr_metallicity_parameters is not None:
        mzr = compute_mzr_birth_metallicity(
            t_grid_gyr=t_grid,
            z_grid=z_grid,
            sfr_grid=sfr_grid,
            active_grid=starforming,
            parameters=mzr_metallicity_parameters,
            random_seed=random_seeds.metallicity,
        )
        birth_metallicity = np.asarray(mzr.birth_metallicity_zsun_grid, dtype=float)
        metallicity_source = "mzr"
    elif regulator_metallicity_parameters is not None:
        regulator = compute_regulator_metallicity(
            t_grid_gyr=t_grid,
            z_grid=z_grid,
            mh_grid=mh_grid,
            sfr_grid=sfr_grid,
            active_grid=starforming,
            cosmology=cosmology,
            parameters=regulator_metallicity_parameters,
            random_seed=random_seeds.metallicity,
        )
        birth_metallicity = np.asarray(regulator.birth_metallicity_zsun_grid, dtype=float)
        gas_metallicity = np.asarray(regulator.gas_metallicity_zsun_grid, dtype=float)
        metal_mass = np.asarray(regulator.metal_mass_grid, dtype=float)
        gas_mass = np.asarray(regulator.gas_mass_grid, dtype=float)
        metallicity_source = "regulator"

    floor_mass = np.zeros_like(redshift_grid)
    active_flat = active_grid.reshape(-1)
    if np.any(active_flat):
        active_mh = np.asarray(sfr_tracks["Mh"], dtype=float)[active_flat]
        active_z = np.asarray(sfr_tracks["z"], dtype=float)[active_flat]
        for index, z_value in enumerate(redshift_grid):
            selection = np.isclose(active_z, z_value)
            if np.any(selection):
                floor_mass[index] = float(np.min(active_mh[selection]))
    if not np.any(floor_mass > 0.0):
        raise RuntimeError("could not infer an effective M_min(z) floor from active histories")
    finished = time.perf_counter()
    shared = SharedHaloBatch(
        popiii_enabled=enable_popiii,
        redshift_grid=redshift_grid,
        floor_mass_msun=floor_mass,
        time_gyr_grid=t_grid,
        redshift_history_grid=z_grid,
        halo_mass_msun_grid=mh_grid,
        dmh_dt_sfr_msun_per_gyr_grid=dmhdt_sfr_grid,
        sfr_msun_per_yr_grid=sfr_grid,
        active_grid=active_grid,
        starforming_grid=starforming,
        popiii_sfr_msun_per_yr_grid=popiii_sfr,
        popiii_source_grid=popiii_source,
        popiii_fstar_grid=popiii_fstar,
        popiii_duty_cycle_grid=popiii_duty,
        popiii_lower_mass_msun_grid=popiii_lower,
        popiii_upper_mass_msun_grid=popiii_upper,
        burst_sfr_multiplier_grid=burst_multiplier,
        birth_metallicity_zsun_grid=birth_metallicity,
        gas_metallicity_zsun_grid=gas_metallicity,
        metal_mass_msun_grid=metal_mass,
        gas_mass_msun_grid=gas_mass,
        metallicity_source=metallicity_source,
        timing_mah_generation_seconds=after_mah - started,
        timing_sfr_and_chemistry_seconds=finished - after_mah,
        _array_policy="mutable" if _mutable_result_sources is not None else "view",
    )
    if _mutable_result_sources is not None:
        if _mutable_result_sources:
            raise ValueError("_mutable_result_sources must be empty")
        _mutable_result_sources["histories"] = histories
        _mutable_result_sources["sfr_tracks"] = sfr_tracks
    return shared


def load_ssp_kernels(
    *,
    ssp_file: str | Path,
    imf_modes: tuple[str, ...],
    topheavy_ssp_file: str | Path,
    topheavy_ssp_metallicity: float | None,
    enable_popiii: bool,
    popiii_ssp_file: str | Path,
) -> LoadedSSPKernels:
    canonical_path = resolve_ssp_path(ssp_file)
    topheavy_path = resolve_ssp_path(topheavy_ssp_file)
    popiii_path = resolve_ssp_path(popiii_ssp_file)
    canonical_age, canonical_luminosity = load_uv1600_table(canonical_path)
    has_variant = any(validate_imf_mode(mode) != IMF_MODE_CANONICAL for mode in imf_modes)
    if has_variant:
        topheavy_age, topheavy_luminosity = load_uv1600_table(
            topheavy_path,
            metallicity=topheavy_ssp_metallicity,
        )
    else:
        topheavy_age = None
        topheavy_luminosity = None
    if enable_popiii:
        popiii_age, popiii_luminosity = load_popiii_uv_luminosity_table(popiii_path)
    else:
        popiii_age = None
        popiii_luminosity = None
    return LoadedSSPKernels(
        canonical_age_myr=canonical_age,
        canonical_luminosity_per_msun=canonical_luminosity,
        topheavy_age_myr=topheavy_age,
        topheavy_luminosity_per_msun=topheavy_luminosity,
        popiii_age_myr=popiii_age,
        popiii_luminosity_per_msun=popiii_luminosity,
        canonical_ssp_path=canonical_path,
        topheavy_ssp_path=topheavy_path,
        popiii_ssp_path=popiii_path,
        topheavy_ssp_template_metallicity_zsun=topheavy_ssp_metallicity,
    )


def evaluate_shared_halo_batch(
    shared: SharedHaloBatch,
    *,
    imf_mode: str,
    transition_parameters: IMFTransitionParameters,
    kernels: LoadedSSPKernels,
    ssp_lookback_max_myr: float,
) -> HaloModeEvaluation:
    """Evaluate one IMF mode without mutating mode-independent shared grids."""

    if type(shared) is not SharedHaloBatch:
        raise TypeError("shared must be exactly SharedHaloBatch")
    if type(kernels) is not LoadedSSPKernels:
        raise TypeError("kernels must be exactly LoadedSSPKernels")
    mode = validate_imf_mode(imf_mode)
    started = time.perf_counter()
    candidate = compute_topheavy_source_flags(
        imf_mode=mode,
        z_grid=shared.redshift_history_grid,
        mh_grid=shared.halo_mass_msun_grid,
        dmhdt_sfr_grid=shared.dmh_dt_sfr_msun_per_gyr_grid,
        active_grid=shared.starforming_grid,
        transition_parameters=replace(
            transition_parameters,
            metallicity_topheavy_max_zsun=None,
        ),
    )
    topheavy = compute_topheavy_source_flags(
        imf_mode=mode,
        z_grid=shared.redshift_history_grid,
        mh_grid=shared.halo_mass_msun_grid,
        dmhdt_sfr_grid=shared.dmh_dt_sfr_msun_per_gyr_grid,
        active_grid=shared.starforming_grid,
        birth_metallicity_zsun_grid=shared.birth_metallicity_zsun_grid,
        transition_parameters=transition_parameters,
    )
    if mode != IMF_MODE_CANONICAL and kernels.topheavy_age_myr is None:
        raise ValueError("top-heavy SSP kernels are required for non-canonical IMF modes")
    canonical_uv, topheavy_uv = _compute_final_uv_luminosity_components_vectorized(
        t_grid=shared.time_gyr_grid,
        sfr_grid=shared.sfr_msun_per_yr_grid,
        active_grid=shared.active_grid,
        topheavy_source_flag_grid=topheavy,
        ssp_age_grid=kernels.canonical_age_myr,
        ssp_luv_grid=kernels.canonical_luminosity_per_msun,
        topheavy_ssp_age_grid=kernels.topheavy_age_myr if mode != IMF_MODE_CANONICAL else None,
        topheavy_ssp_luv_grid=(
            kernels.topheavy_luminosity_per_msun if mode != IMF_MODE_CANONICAL else None
        ),
        ssp_lookback_max_myr=ssp_lookback_max_myr,
    )
    if np.any(shared.popiii_source_grid):
        if kernels.popiii_age_myr is None or kernels.popiii_luminosity_per_msun is None:
            raise ValueError("Pop III SSP kernels are required when Pop III sources are enabled")
        popiii_uv, _ = _compute_final_uv_luminosity_components_vectorized(
            t_grid=shared.time_gyr_grid,
            sfr_grid=shared.popiii_sfr_msun_per_yr_grid,
            active_grid=shared.popiii_source_grid,
            topheavy_source_flag_grid=np.zeros_like(shared.popiii_source_grid),
            ssp_age_grid=kernels.popiii_age_myr,
            ssp_luv_grid=kernels.popiii_luminosity_per_msun,
            topheavy_ssp_age_grid=None,
            topheavy_ssp_luv_grid=None,
            ssp_lookback_max_myr=ssp_lookback_max_myr,
        )
    else:
        popiii_uv = np.zeros_like(canonical_uv)
    total = canonical_uv + topheavy_uv + popiii_uv
    positive = total > 0.0
    top_fraction = np.zeros_like(total)
    popiii_fraction = np.zeros_like(total)
    top_fraction[positive] = topheavy_uv[positive] / total[positive]
    popiii_fraction[positive] = popiii_uv[positive] / total[positive]
    return HaloModeEvaluation(
        imf_mode=mode,
        uv_luminosity_erg_per_s_hz=total,
        canonical_uv_luminosity_erg_per_s_hz=canonical_uv,
        topheavy_uv_luminosity_erg_per_s_hz=topheavy_uv,
        popiii_uv_luminosity_erg_per_s_hz=popiii_uv,
        topheavy_source_grid=topheavy,
        candidate_topheavy_source_grid=candidate,
        topheavy_light_fraction=top_fraction,
        popiii_light_fraction=popiii_fraction,
        topheavy_source_count=int(np.count_nonzero(topheavy & shared.starforming_grid)),
        starforming_source_count=int(np.count_nonzero(shared.starforming_grid)),
        popiii_source_count=int(np.count_nonzero(shared.popiii_source_grid)),
        active_source_count=int(np.count_nonzero(shared.active_grid)),
        uv_convolution_seconds=time.perf_counter() - started,
    )


def _population_metadata(
    shared: SharedHaloBatch,
    evaluation: HaloModeEvaluation,
    kernels: LoadedSSPKernels,
    *,
    worker_count: int,
    topheavy_ssp_metallicity: float | None,
    enable_popiii: bool,
    popiii_sfr_parameters: PopIIISFRParameters,
    mode: str,
    imf_transition_parameters: IMFTransitionParameters,
) -> dict[str, object]:
    """Describe population selection and luminosity diagnostics for a prepared batch."""
    starforming = shared.starforming_grid
    positive_light = evaluation.uv_luminosity_erg_per_s_hz > 0.0
    candidate_count = int(
        np.count_nonzero(evaluation.candidate_topheavy_source_grid & starforming)
    )
    return {
        "n_tracks": shared.time_gyr_grid.shape[0],
        "steps_per_halo": shared.time_gyr_grid.shape[1],
        "workers": max(1, worker_count),
        "ssp_file": str(kernels.canonical_ssp_path),
        "canonical_ssp_file": str(kernels.canonical_ssp_path),
        "topheavy_ssp_file": str(kernels.topheavy_ssp_path),
        "topheavy_ssp_metallicity": topheavy_ssp_metallicity,
        "popiii_enabled": bool(enable_popiii),
        "popiii_ssp_file": str(kernels.popiii_ssp_path),
        "popiii_sfr_parameters": popiii_sfr_parameters.as_metadata(),
        "imf_mode": mode,
        "imf_transition_parameters": {
            "z_topheavy_min": float(imf_transition_parameters.z_topheavy_min),
            "source_redshift_gate_enabled": bool(
                imf_transition_parameters.source_redshift_gate_enabled
            ),
            "growth_time_threshold_myr": float(
                imf_transition_parameters.growth_time_threshold_myr
            ),
            "metallicity_topheavy_max_zsun": (
                None
                if imf_transition_parameters.metallicity_topheavy_max_zsun is None
                else float(imf_transition_parameters.metallicity_topheavy_max_zsun)
            ),
        },
        "metallicity_topheavy_gate_applied": (
            mode != IMF_MODE_CANONICAL
            and imf_transition_parameters.metallicity_topheavy_max_zsun is not None
        ),
        "topheavy_candidate_source_fraction": (
            float(np.mean(evaluation.candidate_topheavy_source_grid[starforming]))
            if np.any(starforming)
            else 0.0
        ),
        "topheavy_candidate_source_count": candidate_count,
        "topheavy_source_fraction": (
            evaluation.topheavy_source_count / evaluation.starforming_source_count
            if evaluation.starforming_source_count > 0
            else 0.0
        ),
        "topheavy_source_count": evaluation.topheavy_source_count,
        "starforming_source_count": evaluation.starforming_source_count,
        "topheavy_light_fraction_median": (
            float(np.median(evaluation.topheavy_light_fraction[positive_light]))
            if np.any(positive_light)
            else 0.0
        ),
        "popiii_source_fraction": (
            evaluation.popiii_source_count / evaluation.active_source_count
            if evaluation.active_source_count > 0
            else 0.0
        ),
        "popiii_source_count": evaluation.popiii_source_count,
        "active_source_count": evaluation.active_source_count,
        "popiii_light_fraction_median": (
            float(np.median(evaluation.popiii_light_fraction[positive_light]))
            if np.any(positive_light)
            else 0.0
        ),
        "popiii_luminosity_median": float(
            np.median(evaluation.popiii_uv_luminosity_erg_per_s_hz)
        ),
    }


def _build_compatibility_result(
    shared: SharedHaloBatch,
    evaluation: HaloModeEvaluation,
    mutable_histories: HaloHistoryResult,
    mutable_sfr_tracks: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> HaloUVPipelineResult:
    """Return independently mutable arrays at the historical public API boundary."""
    return HaloUVPipelineResult(
        histories=mutable_histories,
        sfr_tracks=mutable_sfr_tracks,
        uv_luminosities=np.array(evaluation.uv_luminosity_erg_per_s_hz, copy=True),
        uv_luminosities_canonical=np.array(
            evaluation.canonical_uv_luminosity_erg_per_s_hz,
            copy=True,
        ),
        uv_luminosities_topheavy=np.array(
            evaluation.topheavy_uv_luminosity_erg_per_s_hz,
            copy=True,
        ),
        uv_luminosities_popiii=np.array(
            evaluation.popiii_uv_luminosity_erg_per_s_hz,
            copy=True,
        ),
        redshift_grid=np.array(shared.redshift_grid, copy=True),
        floor_mass=np.array(shared.floor_mass_msun, copy=True),
        active_grid=np.array(shared.active_grid, copy=True),
        imf_topheavy_source_grid=np.array(evaluation.topheavy_source_grid, copy=True),
        popiii_source_grid=np.array(shared.popiii_source_grid, copy=True),
        metadata=metadata,
        gas_metallicity_zsun_grid=(
            None
            if shared.gas_metallicity_zsun_grid is None
            else np.array(shared.gas_metallicity_zsun_grid, copy=True)
        ),
        birth_metallicity_zsun_grid=(
            None
            if shared.birth_metallicity_zsun_grid is None
            else np.array(shared.birth_metallicity_zsun_grid, copy=True)
        ),
        metal_mass_grid=(
            None
            if shared.metal_mass_msun_grid is None
            else np.array(shared.metal_mass_msun_grid, copy=True)
        ),
        gas_mass_grid=(
            None
            if shared.gas_mass_msun_grid is None
            else np.array(shared.gas_mass_msun_grid, copy=True)
        ),
    )


def run_halo_uv_pipeline(
    n_tracks: int,
    z_final: float,
    Mh_final: float,
    *,
    cosmology: Cosmology,
    random_seeds: PipelineRandomSeeds,
    z_start_max: float = 50.0,
    n_grid: int = 240,
    ssp_file: str | Path = DEFAULT_SSP_FILE,
    topheavy_ssp_file: str | Path = DEFAULT_TOPHEAVY_SSP_FILE,
    topheavy_ssp_metallicity: float | None = DEFAULT_TOPHEAVY_SSP_METALLICITY,
    enable_popiii: bool = False,
    popiii_sfr_parameters: PopIIISFRParameters = DEFAULT_POPIII_SFR_PARAMETERS,
    popiii_ssp_file: str | Path = DEFAULT_POPIII_SSP_FILE,
    imf_mode: str = "canonical",
    imf_transition_parameters: IMFTransitionParameters = DEFAULT_IMF_TRANSITION_PARAMETERS,
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
    workers: int | None = None,
    burst_lookback_max_myr: float = EXTENDED_BURST_LOOKBACK_MAX_MYR,
    ssp_lookback_max_myr: float = SSP_UV_LOOKBACK_MAX_MYR,
    sfr_model_parameters: SFRModelParameters = DEFAULT_SFR_MODEL_PARAMETERS,
    mzr_metallicity_parameters: MZRBirthMetallicityParameters | None = None,
    regulator_metallicity_parameters: RegulatorMetallicityParameters | None = None,
    burst_scatter_dex: float = 0.0,
    burst_scatter_timescale_myr: float = DEFAULT_BURST_SCATTER_TIMESCALE_MYR,
    burst_scatter_preserve_mean: bool = True,
) -> HaloUVPipelineResult:
    """Compatibility wrapper over shared preparation, SSP loading, and mode evaluation."""

    mode = validate_imf_mode(imf_mode)
    worker_count = default_worker_count() if workers is None else int(workers)
    if worker_count <= 0:
        raise ValueError("workers must be positive")
    if (
        mode != IMF_MODE_CANONICAL
        and imf_transition_parameters.metallicity_topheavy_max_zsun is not None
        and mzr_metallicity_parameters is None
        and regulator_metallicity_parameters is None
    ):
        raise ValueError(
            "a birth metallicity source must be provided when metallicity_topheavy_max_zsun is set"
        )
    wrapper_started = time.perf_counter()
    mutable_result_sources: dict[str, object] = {}
    shared = prepare_shared_halo_batch(
        n_tracks=n_tracks,
        z_final=z_final,
        Mh_final=Mh_final,
        cosmology=cosmology,
        random_seeds=random_seeds,
        z_start_max=z_start_max,
        n_grid=n_grid,
        enable_popiii=enable_popiii,
        popiii_sfr_parameters=popiii_sfr_parameters,
        sampler=sampler,
        mah_backend=mah_backend,
        tng_mah_cache_path=tng_mah_cache_path,
        tng_mass_bin_width_dex=tng_mass_bin_width_dex,
        tng_min_candidates=tng_min_candidates,
        tng_smoothing_myr=tng_smoothing_myr,
        tng_time_grid_mode=tng_time_grid_mode,
        thesan_mah_cache_path=thesan_mah_cache_path,
        thesan_mass_bin_width_dex=thesan_mass_bin_width_dex,
        thesan_min_candidates=thesan_min_candidates,
        thesan_smoothing_myr=thesan_smoothing_myr,
        thesan_time_grid_mode=thesan_time_grid_mode,
        enable_time_delay=enable_time_delay,
        burst_lookback_max_myr=burst_lookback_max_myr,
        sfr_model_parameters=sfr_model_parameters,
        mzr_metallicity_parameters=mzr_metallicity_parameters,
        regulator_metallicity_parameters=regulator_metallicity_parameters,
        burst_scatter_dex=burst_scatter_dex,
        burst_scatter_timescale_myr=burst_scatter_timescale_myr,
        burst_scatter_preserve_mean=burst_scatter_preserve_mean,
        _mutable_result_sources=mutable_result_sources,
    )
    mutable_histories = mutable_result_sources.pop("histories", None)
    mutable_sfr_tracks = mutable_result_sources.pop("sfr_tracks", None)
    if mutable_result_sources:
        raise RuntimeError("unexpected mutable pipeline result sources")
    if type(mutable_histories) is not HaloHistoryResult:
        raise RuntimeError("missing mutable halo history result")
    if type(mutable_sfr_tracks) is not dict:
        raise RuntimeError("missing mutable SFR track result")
    kernels = load_ssp_kernels(
        ssp_file=ssp_file,
        imf_modes=(mode,),
        topheavy_ssp_file=topheavy_ssp_file,
        topheavy_ssp_metallicity=topheavy_ssp_metallicity,
        enable_popiii=enable_popiii,
        popiii_ssp_file=popiii_ssp_file,
    )
    evaluation = evaluate_shared_halo_batch(
        shared,
        imf_mode=mode,
        transition_parameters=imf_transition_parameters,
        kernels=kernels,
        ssp_lookback_max_myr=ssp_lookback_max_myr,
    )
    histories_metadata = mutable_histories.metadata
    starforming = shared.starforming_grid
    gas_metallicity = shared.gas_metallicity_zsun_grid
    birth_metallicity = shared.birth_metallicity_zsun_grid
    metadata = {
        **_population_metadata(
            shared, evaluation, kernels,
            worker_count=worker_count,
            topheavy_ssp_metallicity=topheavy_ssp_metallicity,
            enable_popiii=enable_popiii,
            popiii_sfr_parameters=popiii_sfr_parameters,
            mode=mode,
            imf_transition_parameters=imf_transition_parameters,
        ),
        "metallicity_source": shared.metallicity_source,
        "mah_backend": validate_mah_backend(mah_backend),
        "sampler": sampler,
        "negative_dmhdt_clip_count": histories_metadata["negative_dmhdt_clip_count"],
        "negative_dmhdt_total_count": histories_metadata["negative_dmhdt_total_count"],
        "negative_dmhdt_clip_fraction": histories_metadata["negative_dmhdt_clip_fraction"],
        "tng_mah_cache_path": histories_metadata.get("tng_mah_cache_path"),
        "tng_source_simulation": histories_metadata.get("source_simulation")
        if mah_backend == MAH_BACKEND_TNG
        else None,
        "tng_mass_bin_width_dex": None
        if mah_backend != MAH_BACKEND_TNG
        else float(tng_mass_bin_width_dex),
        "tng_min_candidates": None if mah_backend != MAH_BACKEND_TNG else int(tng_min_candidates),
        "tng_candidate_count": histories_metadata.get("candidate_count")
        if mah_backend == MAH_BACKEND_TNG
        else None,
        "tng_smoothing_myr": None if mah_backend != MAH_BACKEND_TNG else float(tng_smoothing_myr),
        "tng_time_grid_mode": None
        if mah_backend != MAH_BACKEND_TNG
        else histories_metadata.get("tng_time_grid_mode"),
        "thesan_mah_cache_path": histories_metadata.get("thesan_mah_cache_path"),
        "thesan_source_simulation": histories_metadata.get("source_simulation")
        if mah_backend == MAH_BACKEND_THESAN
        else None,
        "thesan_source_tree": histories_metadata.get("source_tree")
        if mah_backend == MAH_BACKEND_THESAN
        else None,
        "thesan_mass_bin_width_dex": None
        if mah_backend != MAH_BACKEND_THESAN
        else float(thesan_mass_bin_width_dex),
        "thesan_min_candidates": None
        if mah_backend != MAH_BACKEND_THESAN
        else int(thesan_min_candidates),
        "thesan_candidate_count": histories_metadata.get("candidate_count")
        if mah_backend == MAH_BACKEND_THESAN
        else None,
        "thesan_smoothing_myr": None
        if mah_backend != MAH_BACKEND_THESAN
        else float(thesan_smoothing_myr),
        "thesan_time_grid_mode": None
        if mah_backend != MAH_BACKEND_THESAN
        else histories_metadata.get("thesan_time_grid_mode"),
        "mzr_metallicity_enabled": mzr_metallicity_parameters is not None,
        "regulator_metallicity_enabled": regulator_metallicity_parameters is not None,
        "random_seeds": random_seeds.as_metadata(),
        "mzr_metallicity_parameters": (
            mzr_metallicity_parameters.as_metadata()
            if mzr_metallicity_parameters is not None
            else None
        ),
        "regulator_metallicity_parameters": (
            regulator_metallicity_parameters.as_metadata()
            if regulator_metallicity_parameters is not None
            else None
        ),
        "final_gas_metallicity_zsun_median": (
            float(np.nanmedian(gas_metallicity[:, -1]))
            if gas_metallicity is not None
            else None
        ),
        "birth_metallicity_zsun_starforming_median": (
            float(np.nanmedian(birth_metallicity[starforming]))
            if birth_metallicity is not None and np.any(starforming)
            else None
        ),
        "enable_time_delay": enable_time_delay,
        "time_grid_mode": histories_metadata.get("time_grid_mode", "uniform_in_t"),
        "dt_gyr": float(histories_metadata.get("dt_gyr_median", np.nan)),
        "burst_lookback_max_myr": float(burst_lookback_max_myr),
        "burst_scatter_enabled": float(burst_scatter_dex) > 0.0,
        "burst_scatter_dex": float(burst_scatter_dex),
        "burst_scatter_timescale_myr": float(burst_scatter_timescale_myr),
        "burst_scatter_preserve_mean": bool(burst_scatter_preserve_mean),
        "burst_scatter_mass_conserving": bool(burst_scatter_preserve_mean),
        "burst_sfr_multiplier_median": (
            float(np.median(shared.burst_sfr_multiplier_grid[starforming]))
            if np.any(starforming)
            else 1.0
        ),
        "burst_sfr_multiplier_p16": (
            float(np.percentile(shared.burst_sfr_multiplier_grid[starforming], 16.0))
            if np.any(starforming)
            else 1.0
        ),
        "burst_sfr_multiplier_p84": (
            float(np.percentile(shared.burst_sfr_multiplier_grid[starforming], 84.0))
            if np.any(starforming)
            else 1.0
        ),
        "ssp_lookback_max_myr": float(ssp_lookback_max_myr),
        "sfr_model_parameters": {
            "epsilon_0": sfr_model_parameters.epsilon_0,
            "characteristic_mass": sfr_model_parameters.characteristic_mass,
            "beta_star": sfr_model_parameters.beta_star,
            "gamma_star": sfr_model_parameters.gamma_star,
        },
        "timing_seconds": {
            "mah_generation": shared.timing_mah_generation_seconds,
            "sfr": shared.timing_sfr_and_chemistry_seconds,
            "uv_convolution": evaluation.uv_convolution_seconds,
            "total_without_plotting": time.perf_counter() - wrapper_started,
        },
        "uv_convolution_method": "shared_prepared_batch_final_ssp_observable_v2",
    }
    return _build_compatibility_result(
        shared, evaluation, mutable_histories, mutable_sfr_tracks, metadata,
    )
