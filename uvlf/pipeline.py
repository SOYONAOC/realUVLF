from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mah import Cosmology, HaloHistoryResult, generate_halo_histories
from sfr import DEFAULT_SFR_MODEL_PARAMETERS, SFRModelParameters, compute_sfr_from_tracks
from ssp import compute_halo_uv_luminosity, load_uv1600_table


DEFAULT_SSP_FILE = "spectra-bin_byrne23/spectra-bin-imf135_300.BASEL.z001.a+00.dat"
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


def _init_uv_worker(ssp_luv_grid: np.ndarray) -> None:
    _UV_WORKER_STATE["ssp_luv_grid"] = np.asarray(ssp_luv_grid, dtype=float)


def _compute_uv_chunk(
    args: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    t_grid, mh_chunk, sfr_chunk, active_chunk, ssp_age_grid = args
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
) -> np.ndarray:
    if n_workers <= 1:
        _init_uv_worker(ssp_luv_grid)
        return _compute_uv_chunk((t_grid, mh_grid, sfr_grid, active_grid, ssp_age_grid))

    chunk_count = min(n_workers, mh_grid.shape[0])
    mh_chunks = np.array_split(mh_grid, chunk_count, axis=0)
    sfr_chunks = np.array_split(sfr_grid, chunk_count, axis=0)
    active_chunks = np.array_split(active_grid, chunk_count, axis=0)
    tasks = [
        (t_grid, mh_chunk, sfr_chunk, active_chunk, ssp_age_grid)
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
    sfr_model_parameters: SFRModelParameters = DEFAULT_SFR_MODEL_PARAMETERS,
) -> HaloUVPipelineResult:
    """Run the main mah -> sfr -> UV pipeline and return per-halo UV luminosities."""

    cosmology = Cosmology() if cosmology is None else cosmology
    workers = default_worker_count() if workers is None else int(workers)
    redshift_grid = np.linspace(z_start_max, z_final, n_grid, dtype=float)

    t0 = time.perf_counter()
    histories = generate_halo_histories(
        n_tracks=n_tracks,
        z_final=z_final,
        Mh_final=Mh_final,
        z_start_max=z_start_max,
        cosmology=cosmology,
        random_seed=random_seed,
        time_grid_mode="custom",
        custom_grid=redshift_grid,
        store_inactive_history=True,
        sampler=sampler,
    )
    t1 = time.perf_counter()

    sfr_tracks = compute_sfr_from_tracks(
        histories.tracks,
        enable_time_delay=enable_time_delay,
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

    # The outer UVLF Monte Carlo now owns parallelism over Mh samples.
    # Keep the per-mass UV convolution serial here to avoid nested process pools.
    _init_uv_worker(luv_per_msun)
    uv_luminosities = _compute_uv_chunk((t_grid[0], mh_grid, sfr_grid, active_grid, ssp_age_grid_gyr))
    # uv_luminosities = compute_uv_luminosities_parallel(
    #     t_grid=t_grid[0],
    #     mh_grid=mh_grid,
    #     sfr_grid=sfr_grid,
    #     active_grid=active_grid,
    #     ssp_age_grid=ssp_age_grid_gyr,
    #     ssp_luv_grid=luv_per_msun,
    #     n_workers=max(1, workers),
    # )
    t3 = time.perf_counter()

    metadata = {
        "n_tracks": n_halos,
        "steps_per_halo": steps_per_halo,
        "workers": max(1, workers),
        "ssp_file": str(Path(ssp_file).expanduser().resolve()),
        "enable_time_delay": enable_time_delay,
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
