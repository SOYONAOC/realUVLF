from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from massfunc import Mass_func

from sfr import DEFAULT_SFR_MODEL_PARAMETERS, SFRModelParameters
from .pipeline import DEFAULT_SSP_FILE, default_worker_count, run_halo_uv_pipeline


LOGM_MIN = 9.0
LOGM_MAX = 13.0
AB_ZEROPOINT_LNU = 51.60


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


def _write_progress(progress_path: Path, completed: int, total: int, elapsed_seconds: float) -> None:
    fraction = completed / total
    filled = int(round(30 * fraction))
    bar = "#" * filled + "-" * (30 - filled)
    rate = completed / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    remaining = total - completed
    eta_seconds = remaining / rate if rate > 0.0 else float("inf")
    eta_text = f"{eta_seconds:.1f}s" if np.isfinite(eta_seconds) else "inf"
    text = (
        f"[{bar}] {completed}/{total} "
        f"({fraction * 100.0:.2f}%) "
        f"elapsed={elapsed_seconds:.1f}s "
        f"eta={eta_text}\n"
    )
    progress_path.write_text(text, encoding="utf-8")


def _run_single_mass_sample(
    args: tuple[
        int,
        float,
        float,
        float,
        float,
        int,
        float,
        int,
        str,
        bool,
        str,
        int | None,
        SFRModelParameters,
    ],
) -> tuple[int, float, np.ndarray, float]:
    (
        mass_index,
        log_mass,
        mass,
        weight,
        z_obs,
        n_tracks,
        z_start_max,
        n_grid,
        sampler,
        enable_time_delay,
        ssp_file,
        random_seed,
        sfr_model_parameters,
    ) = args

    t0 = time.perf_counter()
    pipeline_result = run_halo_uv_pipeline(
        n_tracks=n_tracks,
        z_final=z_obs,
        Mh_final=float(mass),
        z_start_max=z_start_max,
        n_grid=n_grid,
        random_seed=random_seed,
        sampler=sampler,
        enable_time_delay=enable_time_delay,
        workers=1,
        ssp_file=ssp_file,
        sfr_model_parameters=sfr_model_parameters,
    )
    duration = time.perf_counter() - t0
    luminosity = np.asarray(pipeline_result.uv_luminosities, dtype=float)
    return mass_index, log_mass, luminosity, duration


def sample_uvlf_from_hmf(
    z_obs: float,
    N_mass: int = 3000,
    n_tracks: int = 1000,
    random_seed: int | None = 42,
    *,
    quantity: str = "Muv",
    bins: int | np.ndarray = 40,
    logM_min: float = LOGM_MIN,
    logM_max: float = LOGM_MAX,
    z_start_max: float = 50.0,
    n_grid: int = 240,
    sampler: str = "mcbride",
    enable_time_delay: bool = False,
    pipeline_workers: int | None = None,
    ssp_file: str = DEFAULT_SSP_FILE,
    progress_path: str | Path | None = None,
    sfr_model_parameters: SFRModelParameters = DEFAULT_SFR_MODEL_PARAMETERS,
) -> UVLFSamplingResult:
    """Sample a UVLF by Monte Carlo integration over the ST halo mass function."""

    if quantity not in {"Muv", "luminosity"}:
        raise ValueError("quantity must be either 'Muv' or 'luminosity'")
    if N_mass < 1 or n_tracks < 1:
        raise ValueError("N_mass and n_tracks must both be positive")
    if logM_max <= logM_min:
        raise ValueError("logM_max must be larger than logM_min")

    pipeline_workers = default_worker_count() if pipeline_workers is None else int(pipeline_workers)
    progress_file = None if progress_path is None else Path(progress_path).expanduser().resolve()
    if progress_file is not None:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        _write_progress(progress_file, completed=0, total=N_mass, elapsed_seconds=0.0)
    rng = np.random.default_rng(random_seed)
    hmf = Mass_func()
    hmf.sigma2_interpolation_set()
    hmf.dsig2dm_interpolation_set()

    t0 = time.perf_counter()
    logMh = rng.uniform(logM_min, logM_max, size=N_mass)
    Mh = np.power(10.0, logMh)
    dndm = np.asarray(hmf.dndmst(Mh, z_obs), dtype=float)
    dndlogM = Mh * np.log(10.0) * dndm
    mass_weight = (logM_max - logM_min) * dndlogM / N_mass

    total_samples = N_mass * n_tracks
    sample_logMh = np.empty(total_samples, dtype=float)
    sample_Mh = np.empty(total_samples, dtype=float)
    sample_mass_weight = np.empty(total_samples, dtype=float)
    sample_track_index = np.empty(total_samples, dtype=int)
    sample_luminosity = np.empty(total_samples, dtype=float)
    sample_sample_weight = np.empty(total_samples, dtype=float)
    sample_Muv = np.empty(total_samples, dtype=float)
    per_mass_pipeline_seconds = np.empty(N_mass, dtype=float)

    progress_stride = max(1, N_mass // 100)
    tasks = [
        (
            mass_index,
            float(log_mass),
            float(mass),
            float(weight),
            float(z_obs),
            int(n_tracks),
            float(z_start_max),
            int(n_grid),
            sampler,
            bool(enable_time_delay),
            ssp_file,
            None if random_seed is None else int(random_seed + mass_index),
            sfr_model_parameters,
        )
        for mass_index, (log_mass, mass, weight) in enumerate(zip(logMh, Mh, mass_weight, strict=True))
    ]

    if max(1, pipeline_workers) == 1:
        results_iter = (_run_single_mass_sample(task) for task in tasks)
        completed = 0
        for mass_index, log_mass, luminosity, duration in results_iter:
            if luminosity.size != n_tracks:
                raise RuntimeError("run_halo_uv_pipeline returned an unexpected number of luminosity samples")

            start = mass_index * n_tracks
            stop = start + n_tracks
            sample_logMh[start:stop] = log_mass
            sample_Mh[start:stop] = Mh[mass_index]
            sample_mass_weight[start:stop] = mass_weight[mass_index]
            sample_track_index[start:stop] = np.arange(n_tracks, dtype=int)
            sample_luminosity[start:stop] = luminosity
            sample_sample_weight[start:stop] = mass_weight[mass_index] / n_tracks
            sample_Muv[start:stop] = np.asarray(uv_luminosity_to_muv(luminosity), dtype=float)
            per_mass_pipeline_seconds[mass_index] = duration

            completed += 1
            if progress_file is not None and (completed == N_mass or completed % progress_stride == 0):
                _write_progress(progress_file, completed=completed, total=N_mass, elapsed_seconds=time.perf_counter() - t0)
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=max(1, pipeline_workers)) as executor:
            future_to_index = {executor.submit(_run_single_mass_sample, task): task[0] for task in tasks}
            for future in as_completed(future_to_index):
                mass_index, log_mass, luminosity, duration = future.result()
                if luminosity.size != n_tracks:
                    raise RuntimeError("run_halo_uv_pipeline returned an unexpected number of luminosity samples")

                start = mass_index * n_tracks
                stop = start + n_tracks
                sample_logMh[start:stop] = log_mass
                sample_Mh[start:stop] = Mh[mass_index]
                sample_mass_weight[start:stop] = mass_weight[mass_index]
                sample_track_index[start:stop] = np.arange(n_tracks, dtype=int)
                sample_luminosity[start:stop] = luminosity
                sample_sample_weight[start:stop] = mass_weight[mass_index] / n_tracks
                sample_Muv[start:stop] = np.asarray(uv_luminosity_to_muv(luminosity), dtype=float)
                per_mass_pipeline_seconds[mass_index] = duration

                completed += 1
                if progress_file is not None and (completed == N_mass or completed % progress_stride == 0):
                    _write_progress(progress_file, completed=completed, total=N_mass, elapsed_seconds=time.perf_counter() - t0)

    if quantity == "luminosity":
        histogram_values = sample_luminosity
    else:
        histogram_values = sample_Muv

    bin_edges = _resolve_bin_edges(histogram_values, quantity=quantity, bins=bins)
    valid_mask = np.isfinite(histogram_values) & np.isfinite(sample_sample_weight)
    if quantity == "luminosity":
        valid_mask &= histogram_values > 0.0

    weighted_counts, used_edges = np.histogram(
        histogram_values[valid_mask],
        bins=bin_edges,
        weights=sample_sample_weight[valid_mask],
    )
    bin_width = np.diff(used_edges)
    phi = weighted_counts / bin_width
    bin_centers = 0.5 * (used_edges[:-1] + used_edges[1:])
    total_seconds = time.perf_counter() - t0

    samples = {
        "logMh": sample_logMh,
        "Mh": sample_Mh,
        "mass_weight": sample_mass_weight,
        "track_index": sample_track_index,
        "luminosity": sample_luminosity,
        "Muv": sample_Muv,
        "sample_weight": sample_sample_weight,
    }
    uvlf = {
        "quantity": np.array([quantity]),
        "bin_edges": used_edges,
        "bin_centers": bin_centers,
        "bin_width": bin_width,
        "weighted_counts": weighted_counts,
        "phi": phi,
    }
    metadata = {
        "z_obs": z_obs,
        "N_mass": N_mass,
        "n_tracks": n_tracks,
        "random_seed": random_seed,
        "logM_min": logM_min,
        "logM_max": logM_max,
        "pipeline_workers": max(1, pipeline_workers),
        "quantity": quantity,
        "ssp_file": ssp_file,
        "enable_time_delay": enable_time_delay,
        "sfr_model_parameters": {
            "epsilon_0": sfr_model_parameters.epsilon_0,
            "characteristic_mass": sfr_model_parameters.characteristic_mass,
            "beta_star": sfr_model_parameters.beta_star,
            "gamma_star": sfr_model_parameters.gamma_star,
        },
        "sampling_seconds": total_seconds,
        "per_mass_pipeline_seconds": per_mass_pipeline_seconds,
        "progress_path": None if progress_file is None else str(progress_file),
    }
    return UVLFSamplingResult(samples=samples, uvlf=uvlf, metadata=metadata)
