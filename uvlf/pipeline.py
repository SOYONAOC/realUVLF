from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from mah import Cosmology, HaloHistoryResult, generate_halo_histories
from sfr import (
    DEFAULT_SFR_MODEL_PARAMETERS,
    EXTENDED_BURST_LOOKBACK_MAX_MYR,
    SFRModelParameters,
    compute_sfr_from_tracks,
)
from ssp import SSP_UV_LOOKBACK_MAX_MYR, compute_halo_uv_luminosity, interpolate_ssp_luminosity, load_uv1600_table


DEFAULT_SSP_FILE = "data_save/ssp_uv1600_topheavy_imf100_300_z0005.npz"
YEARS_PER_GYR = 1.0e9


@dataclass(frozen=True)
class HaloUVPipelineResult:
    histories: HaloHistoryResult
    sfr_tracks: dict[str, np.ndarray]
    uv_luminosities: np.ndarray
    redshift_grid: np.ndarray
    floor_mass: np.ndarray
    active_grid: np.ndarray
    metadata: dict[str, Any]


_UV_WORKER_STATE: dict[str, np.ndarray] = {}


def _build_astropy_cosmology(cosmology: Cosmology) -> FlatLambdaCDM:
    return FlatLambdaCDM(H0=cosmology.h0_km_s_mpc, Om0=cosmology.omega_m, Ob0=cosmology.omega_b)


def _init_uv_worker(ssp_luv_grid: np.ndarray) -> None:
    _UV_WORKER_STATE["ssp_luv_grid"] = np.asarray(ssp_luv_grid, dtype=float)


def _compute_uv_chunk(
    args: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
) -> np.ndarray:
    t_grid, mh_chunk, sfr_chunk, active_chunk, ssp_age_grid, ssp_lookback_max_myr = args
    ssp_luv_grid = _UV_WORKER_STATE["ssp_luv_grid"]

    result = np.empty(mh_chunk.shape[0], dtype=float)
    for row_index in range(mh_chunk.shape[0]):
        active = np.asarray(active_chunk[row_index], dtype=bool)
        if not np.any(active):
            result[row_index] = 0.0
            continue

        t_used = np.asarray(t_grid[active], dtype=float)
        mh_used = np.asarray(mh_chunk[row_index][active], dtype=float)
        sfr_used = np.asarray(sfr_chunk[row_index][active], dtype=float)

        result[row_index] = compute_halo_uv_luminosity(
            t_obs=float(t_used[-1]),
            t_history=t_used,
            mh_history=mh_used,
            sfr_history=sfr_used,
            ssp_age_grid=ssp_age_grid,
            ssp_luv_grid=ssp_luv_grid,
            M_min=0.0,
            t_z50=float(t_used[0]),
            time_unit_in_years=YEARS_PER_GYR,
            ssp_lookback_max_myr=ssp_lookback_max_myr,
        )
    return result


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
    if n_workers <= 1:
        _init_uv_worker(ssp_luv_grid)
        return _compute_uv_chunk((t_grid, mh_grid, sfr_grid, active_grid, ssp_age_grid, ssp_lookback_max_myr))

    chunk_count = min(n_workers, mh_grid.shape[0])
    mh_chunks = np.array_split(mh_grid, chunk_count, axis=0)
    sfr_chunks = np.array_split(sfr_grid, chunk_count, axis=0)
    active_chunks = np.array_split(active_grid, chunk_count, axis=0)
    tasks = [
        (t_grid, mh_chunk, sfr_chunk, active_chunk, ssp_age_grid, ssp_lookback_max_myr)
        for mh_chunk, sfr_chunk, active_chunk in zip(mh_chunks, sfr_chunks, active_chunks, strict=True)
    ]

    outputs: list[np.ndarray] = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_uv_worker,
        initargs=(np.asarray(ssp_luv_grid, dtype=float),),
    ) as executor:
        for chunk_output in executor.map(_compute_uv_chunk, tasks):
            outputs.append(np.asarray(chunk_output, dtype=float))
    return np.concatenate(outputs)


def default_worker_count() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


def _resolve_regular_time_grid(t_grid: np.ndarray) -> np.ndarray | None:
    if t_grid.ndim != 2 or t_grid.shape[0] == 0:
        return None
    time_row = np.asarray(t_grid[0], dtype=float)
    if not np.all(np.isfinite(time_row)):
        return None
    if not np.allclose(t_grid, time_row[None, :], rtol=0.0, atol=0.0):
        return None
    return time_row


def _integrate_final_uv_single_halo_regular_grid(
    time_row: np.ndarray,
    sfr_row: np.ndarray,
    active_row: np.ndarray,
    ssp_age_grid: np.ndarray,
    ssp_luv_grid: np.ndarray,
    ssp_lookback_max_myr: float,
) -> float:
    active = np.asarray(active_row, dtype=bool)
    if not np.any(active):
        return 0.0

    t_obs = float(time_row[-1])
    max_lookback_gyr = float(ssp_lookback_max_myr) / 1.0e3
    first_active = int(np.argmax(active))
    lower = max(float(time_row[first_active]), t_obs - max_lookback_gyr)
    if lower >= t_obs:
        return 0.0

    start = int(np.searchsorted(time_row, lower, side="left"))
    t_used = np.asarray(time_row[start:], dtype=float)
    sfr_used = np.asarray(sfr_row[start:], dtype=float)
    active_used = np.asarray(active[start:], dtype=bool)

    if t_used.size == 0:
        return 0.0

    if lower < float(t_used[0]):
        left = start - 1
        right = start
        t_left = float(time_row[left])
        t_right = float(time_row[right])
        sfr_left = float(sfr_row[left])
        sfr_right = float(sfr_row[right])
        weight = 0.0 if t_right <= t_left else (lower - t_left) / (t_right - t_left)
        sfr_lower = sfr_left + weight * (sfr_right - sfr_left)
        t_used = np.concatenate((np.array([lower], dtype=float), t_used))
        sfr_used = np.concatenate((np.array([sfr_lower], dtype=float), sfr_used))
        active_used = np.concatenate((np.array([True], dtype=bool), active_used))

    if np.count_nonzero(active_used) < 2:
        return 0.0

    age_used = np.maximum(t_obs - t_used, 0.0)
    kernel_used = np.asarray(
        interpolate_ssp_luminosity(age_used, ssp_age_grid=ssp_age_grid, ssp_luv_grid=ssp_luv_grid),
        dtype=float,
    )
    integrand = np.where(active_used, sfr_used, 0.0) * kernel_used
    return float(np.trapezoid(integrand, x=t_used * YEARS_PER_GYR))


def _compute_final_uv_luminosities_vectorized(
    t_grid: np.ndarray,
    sfr_grid: np.ndarray,
    active_grid: np.ndarray,
    ssp_age_grid: np.ndarray,
    ssp_luv_grid: np.ndarray,
    ssp_lookback_max_myr: float,
) -> np.ndarray:
    time_row = _resolve_regular_time_grid(t_grid)
    if time_row is None:
        raise ValueError("vectorized final UV convolution requires a shared regular time grid")
    result = np.empty(sfr_grid.shape[0], dtype=float)
    for halo_index in range(sfr_grid.shape[0]):
        result[halo_index] = _integrate_final_uv_single_halo_regular_grid(
            time_row=time_row,
            sfr_row=np.asarray(sfr_grid[halo_index], dtype=float),
            active_row=np.asarray(active_grid[halo_index], dtype=bool),
            ssp_age_grid=ssp_age_grid,
            ssp_luv_grid=ssp_luv_grid,
            ssp_lookback_max_myr=ssp_lookback_max_myr,
        )
    return result


def run_halo_uv_pipeline(
    n_tracks: int,
    z_final: float,
    Mh_final: float,
    *,
    z_start_max: float = 50.0,
    n_grid: int = 240,
    ssp_file: str | Path = DEFAULT_SSP_FILE,
    cosmology: Cosmology | None = None,
    random_seed: int | None = 42,
    sampler: str = "mcbride",
    enable_time_delay: bool = False,
    workers: int | None = None,
    burst_lookback_max_myr: float = EXTENDED_BURST_LOOKBACK_MAX_MYR,
    ssp_lookback_max_myr: float = SSP_UV_LOOKBACK_MAX_MYR,
    sfr_model_parameters: SFRModelParameters = DEFAULT_SFR_MODEL_PARAMETERS,
) -> HaloUVPipelineResult:
    """Run the main mah -> sfr -> UV pipeline and return per-halo UV luminosities."""

    cosmology = Cosmology() if cosmology is None else cosmology
    workers = default_worker_count() if workers is None else int(workers)
    if int(n_grid) < 2:
        raise ValueError("n_grid must be at least 2")
    astro = _build_astropy_cosmology(cosmology)
    t_start_gyr = float(astro.age(z_start_max).value)
    t_end_gyr = float(astro.age(z_final).value)
    dt_gyr = (t_end_gyr - t_start_gyr) / float(int(n_grid) - 1)

    t0 = time.perf_counter()
    histories = generate_halo_histories(
        n_tracks=n_tracks,
        z_final=z_final,
        Mh_final=Mh_final,
        z_start_max=z_start_max,
        cosmology=cosmology,
        random_seed=random_seed,
        time_grid_mode="uniform_in_t",
        dt=dt_gyr,
        store_inactive_history=True,
        sampler=sampler,
    )
    t1 = time.perf_counter()
    redshift_grid = np.unique(np.asarray(histories.tracks["z"], dtype=float))[::-1]

    sfr_tracks = compute_sfr_from_tracks(
        histories.tracks,
        enable_time_delay=enable_time_delay,
        burst_lookback_max_myr=burst_lookback_max_myr,
        model_parameters=sfr_model_parameters,
    )
    t2 = time.perf_counter()

    ages_myr, luv_per_msun = load_uv1600_table(ssp_file)
    ssp_age_grid_gyr = ages_myr / 1.0e3

    halo_ids = np.asarray(sfr_tracks["halo_id"], dtype=int)
    n_halos = np.unique(halo_ids).size
    steps_per_halo = redshift_grid.size
    t_grid = np.asarray(sfr_tracks["t_gyr"], dtype=float).reshape(n_halos, steps_per_halo)
    mh_grid = np.asarray(sfr_tracks["Mh"], dtype=float).reshape(n_halos, steps_per_halo)
    sfr_grid = np.asarray(sfr_tracks["SFR"], dtype=float).reshape(n_halos, steps_per_halo)
    active_grid = np.asarray(sfr_tracks["active_flag"], dtype=bool).reshape(n_halos, steps_per_halo)

    floor_mass = np.zeros_like(redshift_grid, dtype=float)
    active_flat = active_grid.reshape(-1)
    if np.any(active_flat):
        active_mh = np.asarray(sfr_tracks["Mh"], dtype=float)[active_flat]
        active_z = np.asarray(sfr_tracks["z"], dtype=float)[active_flat]
        for index, z_value in enumerate(redshift_grid):
            mask = np.isclose(active_z, z_value)
            if np.any(mask):
                floor_mass[index] = float(np.min(active_mh[mask]))
    positive_floor = floor_mass[floor_mass > 0.0]
    if positive_floor.size == 0:
        raise RuntimeError("could not infer an effective M_min(z) floor from active histories")

    time_row = _resolve_regular_time_grid(t_grid)
    if time_row is not None:
        uv_luminosities = _compute_final_uv_luminosities_vectorized(
            t_grid=t_grid,
            sfr_grid=sfr_grid,
            active_grid=active_grid,
            ssp_age_grid=ssp_age_grid_gyr,
            ssp_luv_grid=luv_per_msun,
            ssp_lookback_max_myr=ssp_lookback_max_myr,
        )
        uv_convolution_method = "vectorized_final_time"
    else:
        # The outer UVLF Monte Carlo owns parallelism over Mh samples.
        # Keep the fallback per-mass UV convolution serial here to avoid nested process pools.
        _init_uv_worker(luv_per_msun)
        uv_luminosities = _compute_uv_chunk(
            (t_grid[0], mh_grid, sfr_grid, active_grid, ssp_age_grid_gyr, float(ssp_lookback_max_myr))
        )
        uv_convolution_method = "per_halo_fallback"
    t3 = time.perf_counter()

    metadata = {
        "n_tracks": n_halos,
        "steps_per_halo": steps_per_halo,
        "workers": max(1, workers),
        "ssp_file": str(Path(ssp_file).expanduser().resolve()),
        "enable_time_delay": enable_time_delay,
        "time_grid_mode": "uniform_in_t",
        "dt_gyr": float(dt_gyr),
        "burst_lookback_max_myr": float(burst_lookback_max_myr),
        "ssp_lookback_max_myr": float(ssp_lookback_max_myr),
        "sfr_model_parameters": {
            "epsilon_0": sfr_model_parameters.epsilon_0,
            "characteristic_mass": sfr_model_parameters.characteristic_mass,
            "beta_star": sfr_model_parameters.beta_star,
            "gamma_star": sfr_model_parameters.gamma_star,
        },
        "timing_seconds": {
            "mah_generation": t1 - t0,
            "sfr": t2 - t1,
            "uv_convolution": t3 - t2,
            "total_without_plotting": t3 - t0,
        },
        "uv_convolution_method": uv_convolution_method,
    }

    return HaloUVPipelineResult(
        histories=histories,
        sfr_tracks=sfr_tracks,
        uv_luminosities=np.asarray(uv_luminosities, dtype=float),
        redshift_grid=redshift_grid,
        floor_mass=floor_mass,
        active_grid=active_grid,
        metadata=metadata,
    )
