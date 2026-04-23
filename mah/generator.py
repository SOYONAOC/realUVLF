from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from .models import Cosmology, HaloHistoryResult
from .physics import accretion_rate
from .sampling import sample_parameters


def _build_astropy_cosmology(cosmology: Cosmology) -> FlatLambdaCDM:
    return FlatLambdaCDM(H0=cosmology.h0_km_s_mpc, Om0=cosmology.omega_m, Ob0=cosmology.omega_b)


def _resolve_redshift_grid(
    z_final: float,
    z_start_max: float,
    time_grid_mode: str,
    dt: float | None,
    dz: float | None,
    cosmology: Cosmology,
    custom_grid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    astro = _build_astropy_cosmology(cosmology)

    if time_grid_mode == "uniform_in_z":
        if dz is None or dz <= 0.0:
            raise ValueError("uniform_in_z requires dz > 0")
        steps = int(np.ceil((z_start_max - z_final) / dz))
        redshift = z_start_max - np.arange(steps + 1, dtype=float) * dz
        redshift[-1] = z_final
        redshift = np.clip(redshift, z_final, z_start_max)
        time_gyr = astro.age(redshift).value
        return redshift, time_gyr

    if time_grid_mode == "uniform_in_t":
        if dt is None or dt <= 0.0:
            raise ValueError("uniform_in_t requires dt > 0")
        t_start = astro.age(z_start_max).value
        t_end = astro.age(z_final).value
        steps = int(np.ceil((t_end - t_start) / dt))
        time_gyr = t_start + np.arange(steps + 1, dtype=float) * dt
        time_gyr[-1] = t_end
        dense_redshift = np.linspace(z_start_max, z_final, 4096)
        dense_time = astro.age(dense_redshift).value
        redshift = np.interp(time_gyr, dense_time, dense_redshift)
        return redshift, time_gyr

    if time_grid_mode == "custom":
        if custom_grid is None:
            raise ValueError("custom time_grid_mode requires custom_grid")
        redshift = np.asarray(custom_grid, dtype=float)
        if redshift.ndim != 1 or redshift.size == 0:
            raise ValueError("custom_grid must be a non-empty 1D redshift grid")
        redshift = np.sort(redshift)[::-1]
        if redshift[0] > z_start_max or redshift[-1] < z_final:
            raise ValueError("custom_grid must stay within [z_final, z_start_max]")
        time_gyr = astro.age(redshift).value
        return redshift, time_gyr

    raise ValueError("time_grid_mode must be one of: uniform_in_z, uniform_in_t, custom")


def _resolve_mass_floor(
    mass_floor: float | np.ndarray | Callable[[np.ndarray], np.ndarray] | None,
    redshift: np.ndarray,
) -> np.ndarray:
    if mass_floor is None:
        try:
            import massfunc as mf
        except ImportError as exc:
            raise ImportError("M_min=None requires the optional dependency massfunc to be installed") from exc

        cosmo = mf.SFRD()
        return np.asarray(cosmo.M_vir(0.61, 1.0e4, redshift), dtype=float)

    if callable(mass_floor):
        values = np.asarray(mass_floor(redshift), dtype=float)
    else:
        values = np.asarray(mass_floor, dtype=float)

    if values.ndim == 0:
        return np.full(redshift.shape, float(values), dtype=float)
    if values.shape != redshift.shape:
        raise ValueError("M_min array must match the redshift grid shape")
    return values


def _flatten_tracks(
    redshift: np.ndarray,
    time_gyr: np.ndarray,
    dt_gyr: np.ndarray,
    mass: np.ndarray,
    mdot: np.ndarray,
    floor_mass: np.ndarray,
    store_inactive_history: bool,
) -> dict[str, np.ndarray]:
    columns: dict[str, list[np.ndarray]] = {
        "halo_id": [],
        "step": [],
        "z": [],
        "t_gyr": [],
        "dt_gyr": [],
        "Mh": [],
        "dMh_dt": [],
        "active_flag": [],
        "termination_flag": [],
    }

    n_steps = redshift.size
    completed_flag = np.full(n_steps, "active", dtype=object)
    completed_flag[-1] = "completed"

    for halo_id in range(mass.shape[0]):
        below_floor = mass[halo_id] < floor_mass
        active = ~below_floor
        termination = completed_flag.copy()

        if np.any(active):
            first_active = int(np.flatnonzero(active)[0])
            if first_active > 0:
                termination[:first_active] = "below_M_min"
            if store_inactive_history:
                slc = slice(0, n_steps)
            else:
                slc = slice(first_active, n_steps)
        else:
            active[:] = False
            termination[:] = "below_M_min"
            if store_inactive_history:
                slc = slice(0, n_steps)
            else:
                continue

        size = slc.stop - slc.start
        columns["halo_id"].append(np.full(size, halo_id, dtype=int))
        columns["step"].append(np.arange(size, dtype=int))
        columns["z"].append(redshift[slc].copy())
        columns["t_gyr"].append(time_gyr[slc].copy())
        columns["dt_gyr"].append(dt_gyr[slc].copy())
        columns["Mh"].append(mass[halo_id, slc].copy())
        columns["dMh_dt"].append(mdot[halo_id, slc].copy())
        columns["active_flag"].append(active[slc].copy())
        columns["termination_flag"].append(termination[slc].copy())

    return {name: np.concatenate(parts) for name, parts in columns.items()}


def generate_halo_histories(
    n_tracks: int,
    z_final: float,
    Mh_final: float,
    z_start_max: float = 50.0,
    M_min: float | np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
    cosmology: Cosmology | None = None,
    random_seed: int | None = None,
    time_grid_mode: str = "uniform_in_z",
    dt: float | None = None,
    dz: float | None = 0.1,
    custom_grid: np.ndarray | None = None,
    store_inactive_history: bool = True,
    sampler: str = "mcbride",
    pilot_samples: int = 50_000,
) -> HaloHistoryResult:
    if n_tracks <= 0:
        raise ValueError("n_tracks must be positive")
    if z_start_max <= z_final:
        raise ValueError("z_start_max must be greater than z_final")
    if Mh_final <= 0.0:
        raise ValueError("Mh_final must be positive")

    cosmology = Cosmology() if cosmology is None else cosmology
    redshift, time_gyr = _resolve_redshift_grid(
        z_final=z_final,
        z_start_max=z_start_max,
        time_grid_mode=time_grid_mode,
        dt=dt,
        dz=dz,
        cosmology=cosmology,
        custom_grid=custom_grid,
    )
    dt_gyr = np.diff(time_gyr, prepend=time_gyr[0])
    floor_mass = _resolve_mass_floor(M_min, redshift)

    rng = np.random.default_rng(random_seed)
    samples, gaussian_approximation = sample_parameters(
        mass_ref=Mh_final,
        size=n_tracks,
        sampler=sampler,
        rng=rng,
        pilot_samples=pilot_samples,
    )
    beta = samples[:, 0]
    gamma = samples[:, 1]

    mass, analytic_mdot = accretion_rate(
        redshift=redshift,
        redshift_final=z_final,
        mass_final=Mh_final,
        beta=beta,
        gamma=gamma,
        cosmology=cosmology,
    )
    mdot = analytic_mdot

    tracks = _flatten_tracks(
        redshift=redshift,
        time_gyr=time_gyr,
        dt_gyr=dt_gyr,
        mass=mass,
        mdot=mdot,
        floor_mass=floor_mass,
        store_inactive_history=store_inactive_history,
    )

    metadata: dict[str, Any] = {
        "n_tracks": n_tracks,
        "z_final": z_final,
        "Mh_final": Mh_final,
        "z_start_max": z_start_max,
        "time_grid_mode": time_grid_mode,
        "grid_size": redshift.size,
        "sampler": sampler,
        "store_inactive_history": store_inactive_history,
        "M_min_mode": "massfunc.SFRD().M_vir(mu=0.61, Tvir=1e4, z)" if M_min is None else "user_provided",
        "random_seed": random_seed,
        "cosmology": {
            "H0_km_s_Mpc": cosmology.h0_km_s_mpc,
            "Omega_m": cosmology.omega_m,
            "Omega_b": cosmology.omega_b,
            "Omega_lambda": cosmology.omega_lambda,
        },
        "sampling_summary": {
            "beta_mean": float(beta.mean()),
            "beta_std": float(beta.std(ddof=1)) if beta.size > 1 else 0.0,
            "gamma_mean": float(gamma.mean()),
            "gamma_std": float(gamma.std(ddof=1)) if gamma.size > 1 else 0.0,
            "gaussian_approximation": None
            if gaussian_approximation is None
            else {
                "mean": gaussian_approximation.mean.copy(),
                "covariance": gaussian_approximation.covariance.copy(),
            },
        },
    }
    if custom_grid is not None:
        metadata["custom_grid_mode"] = "custom redshift grid"

    return HaloHistoryResult(tracks=tracks, metadata=metadata)
