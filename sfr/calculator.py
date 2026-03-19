from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mah import CosmologySet
from mah.models import GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN, KM_PER_MPC, SECONDS_PER_GYR


PROTON_MASS_KG = 1.67262192369e-27
BOLTZMANN_CONSTANT_J_K = 1.380649e-23
YEARS_PER_GYR = 1.0e9
EPSILON_0 = 0.12
CHARACTERISTIC_MASS = 10.0**11.7
BETA_STAR = 0.66
GAMMA_STAR = 0.65
EXTENDED_BURST_KAPPA = 1.0


@dataclass(frozen=True)
class SFRModelParameters:
    epsilon_0: float = EPSILON_0
    characteristic_mass: float = CHARACTERISTIC_MASS
    beta_star: float = BETA_STAR
    gamma_star: float = GAMMA_STAR


DEFAULT_SFR_MODEL_PARAMETERS = SFRModelParameters()


def _resolve_sfr_model_parameters(model_parameters: SFRModelParameters | None) -> SFRModelParameters:
    params = DEFAULT_SFR_MODEL_PARAMETERS if model_parameters is None else model_parameters
    if not 0.0 <= float(params.epsilon_0) <= 1.0:
        raise ValueError("epsilon_0 must lie in [0, 1]")
    if float(params.characteristic_mass) <= 0.0:
        raise ValueError("characteristic_mass must be positive")
    if float(params.beta_star) < 0.0:
        raise ValueError("beta_star must be non-negative")
    if float(params.gamma_star) < 0.0:
        raise ValueError("gamma_star must be non-negative")
    return params


def _stellar_formation_efficiency(mass: np.ndarray, model_parameters: SFRModelParameters) -> np.ndarray:
    ratio = np.asarray(mass, dtype=float) / model_parameters.characteristic_mass
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        denominator = ratio ** (-model_parameters.beta_star) + ratio**model_parameters.gamma_star
    return 2.0 * model_parameters.epsilon_0 / denominator


def _extended_burst_kernel(delta_t_gyr: np.ndarray, td_gyr: float, kappa: float = EXTENDED_BURST_KAPPA) -> np.ndarray:
    delta_t_gyr = np.asarray(delta_t_gyr, dtype=float)
    kernel = np.zeros_like(delta_t_gyr, dtype=float)
    positive = (delta_t_gyr >= 0.0) & np.isfinite(delta_t_gyr) & np.isfinite(td_gyr) & (td_gyr > 0.0)
    if not np.any(positive):
        return kernel
    x = delta_t_gyr[positive]
    kernel[positive] = x / (kappa**2 * td_gyr**2) * np.exp(-x / (kappa * td_gyr))
    return kernel


def _compute_extended_burst_mdot(
    t_gyr: np.ndarray,
    mdot: np.ndarray,
    active: np.ndarray,
    boundaries: np.ndarray,
    kappa: float,
    td_burst: np.ndarray,
) -> np.ndarray:
    mdot_burst = np.zeros_like(mdot, dtype=float)
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        active_group = np.asarray(active[start:end], dtype=bool)
        if not np.any(active_group):
            continue
        local_first = int(np.argmax(active_group))
        first = start + local_first
        td_group = float(td_burst[first])
        if not np.isfinite(td_group) or td_group <= 0.0:
            continue

        t_group = np.asarray(t_gyr[first:end], dtype=float)
        mdot_group = np.asarray(mdot[first:end], dtype=float)
        active_slice = np.asarray(active[first:end], dtype=bool)

        for local_i in range(t_group.size):
            if not active_slice[local_i]:
                continue
            t_now = float(t_group[local_i])
            t_src = t_group[: local_i + 1]
            mdot_src = mdot_group[: local_i + 1]
            valid = np.isfinite(t_src) & np.isfinite(mdot_src)
            if np.count_nonzero(valid) < 2:
                continue
            delta_t = t_now - t_src[valid]
            kernel = _extended_burst_kernel(delta_t, td_group, kappa=kappa)
            mdot_burst[first + local_i] = np.trapezoid(kernel * mdot_src[valid], x=t_src[valid])
    return mdot_burst


def _tracks_are_grouped_and_sorted(halo_id: np.ndarray, time: np.ndarray) -> bool:
    if halo_id.size <= 1:
        return True
    if np.any(halo_id[1:] < halo_id[:-1]):
        return False
    same_halo = halo_id[1:] == halo_id[:-1]
    return bool(np.all(time[1:][same_halo] >= time[:-1][same_halo]))


def _prepare_track_columns(
    tracks: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    arrays = {name: np.asarray(values) for name, values in tracks.items()}
    halo_ids = np.asarray(arrays["halo_id"], dtype=int)
    time = np.asarray(arrays["t_gyr"], dtype=float)

    if _tracks_are_grouped_and_sorted(halo_ids, time):
        sorted_arrays = arrays
    else:
        order = np.lexsort((time, halo_ids))
        sorted_arrays = {name: values[order] for name, values in arrays.items()}

    sorted_halo_ids = np.asarray(sorted_arrays["halo_id"], dtype=int)
    unique_ids, start_indices = np.unique(sorted_halo_ids, return_index=True)
    boundaries = np.empty(start_indices.size + 1, dtype=int)
    boundaries[:-1] = start_indices
    boundaries[-1] = sorted_halo_ids.size
    group_ids = np.repeat(np.arange(unique_ids.size, dtype=int), np.diff(boundaries))
    return sorted_arrays, unique_ids, boundaries, group_ids


def _interpolate_grouped(
    x: np.ndarray,
    y: np.ndarray,
    x_query: np.ndarray,
    boundaries: np.ndarray,
    group_ids: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    result = np.full_like(y, np.nan, dtype=float)
    if not np.any(valid_mask):
        return result

    starts = boundaries[:-1][group_ids]
    ends = boundaries[1:][group_ids]
    counts = ends - starts

    single_point = counts == 1
    single_mask = valid_mask & single_point
    result[single_mask] = y[single_mask]

    multi_mask = valid_mask & ~single_point
    if not np.any(multi_mask):
        return result

    # Shift each halo to a disjoint x-range so one global searchsorted works.
    step = max(1.0, float(x.max() - x.min()) + 1.0)
    offsets = np.arange(boundaries.size - 1, dtype=float) * step
    adjusted_x = x + offsets[group_ids]

    query_groups = group_ids[multi_mask]
    query_x = x_query[multi_mask]
    adjusted_query = query_x + offsets[query_groups]

    right = np.searchsorted(adjusted_x, adjusted_query, side="left")
    right = np.clip(right, starts[multi_mask] + 1, ends[multi_mask] - 1)
    left = right - 1

    x0 = x[left]
    x1 = x[right]
    y0 = y[left]
    y1 = y[right]
    weight = np.zeros_like(query_x)
    denominator = x1 - x0
    nonzero = denominator > 0.0
    weight[nonzero] = (query_x[nonzero] - x0[nonzero]) / denominator[nonzero]
    result[multi_mask] = y0 + weight * (y1 - y0)
    return result


def compute_sfr_from_tracks(
    tracks: dict[str, np.ndarray],
    mu: float = 0.61,
    atomic_cooling_temperature: float = 1.0e4,
    enable_time_delay: bool = False,
    burst_kappa: float = EXTENDED_BURST_KAPPA,
    model_parameters: SFRModelParameters | None = None,
) -> dict[str, np.ndarray]:
    """Compute SFR in Msun/yr and related virial quantities from halo history tracks."""

    required = ("halo_id", "step", "z", "t_gyr", "Mh", "dMh_dt")
    missing = [name for name in required if name not in tracks]
    if missing:
        raise KeyError(f"tracks is missing required columns: {missing}")

    n_rows = int(np.asarray(tracks["halo_id"]).size)
    for name in required:
        if np.asarray(tracks[name]).size != n_rows:
            raise ValueError(f"tracks column '{name}' does not match halo_id length")

    model_parameters = _resolve_sfr_model_parameters(model_parameters)
    cosmo = CosmologySet()
    baryon_fraction = cosmo.omegab / cosmo.omegam
    sorted_tracks, _, boundaries, group_ids = _prepare_track_columns(tracks)
    z = np.asarray(sorted_tracks["z"], dtype=float)
    t_gyr = np.asarray(sorted_tracks["t_gyr"], dtype=float)
    mass = np.asarray(sorted_tracks["Mh"], dtype=float)
    mdot = np.asarray(sorted_tracks["dMh_dt"], dtype=float)

    hubble = cosmo.H0u * np.sqrt(cosmo.omegam * (1.0 + z) ** 3 + cosmo.omegalam)
    rho_crit = cosmo.rhocrit * (hubble / cosmo.H0u) ** 2
    omega_m_z = cosmo.omegam * (1.0 + z) ** 3 / (cosmo.omegam * (1.0 + z) ** 3 + cosmo.omegalam)
    delta = omega_m_z - 1.0
    delta_vir = 18.0 * np.pi**2 + 82.0 * delta - 39.0 * delta**2
    rho_vir = delta_vir * rho_crit

    # Use Msun and Mpc consistently so virial quantities remain numerically stable.
    r_vir = (3.0 * mass / (4.0 * np.pi * delta_vir * rho_crit)) ** (1.0 / 3.0)
    v_c = np.sqrt(GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN * mass / r_vir)
    t_vir = mu * PROTON_MASS_KG * (v_c * 1.0e3) ** 2 / (2.0 * BOLTZMANN_CONSTANT_J_K)
    tau_del = r_vir * KM_PER_MPC / v_c / SECONDS_PER_GYR
    td_burst = np.sqrt(3.0 * np.pi / (32.0 * GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN * rho_vir))
    td_burst = td_burst * KM_PER_MPC / SECONDS_PER_GYR

    starts = boundaries[:-1][group_ids]
    ends = boundaries[1:][group_ids]
    t_src = t_gyr - tau_del
    valid_source = (t_src >= t_gyr[starts]) & (t_src <= t_gyr[ends - 1])
    mh_src = _interpolate_grouped(t_gyr, mass, t_src, boundaries, group_ids, valid_source)
    mdot_src = _interpolate_grouped(t_gyr, mdot, t_src, boundaries, group_ids, valid_source)

    fstar_src = np.full_like(mass, np.nan, dtype=float)
    finite_mass = np.isfinite(mh_src)
    fstar_src[finite_mass] = _stellar_formation_efficiency(mh_src[finite_mass], model_parameters)

    fstar_now = np.full_like(mass, np.nan, dtype=float)
    finite_mass_now = np.isfinite(mass)
    fstar_now[finite_mass_now] = _stellar_formation_efficiency(mass[finite_mass_now], model_parameters)

    sfr = np.zeros_like(mass, dtype=float)
    active_now = np.isfinite(mass) & np.isfinite(mdot) & (t_vir >= atomic_cooling_temperature)
    if enable_time_delay:
        mdot_burst = _compute_extended_burst_mdot(
            t_gyr=t_gyr,
            mdot=mdot,
            active=active_now,
            boundaries=boundaries,
            kappa=float(burst_kappa),
            td_burst=td_burst,
        )
        active_burst = active_now & np.isfinite(fstar_now) & np.isfinite(mdot_burst)
        sfr[active_burst] = baryon_fraction * fstar_now[active_burst] * mdot_burst[active_burst] / YEARS_PER_GYR
    else:
        mdot_burst = np.full_like(mass, np.nan, dtype=float)
        active = np.isfinite(fstar_now) & np.isfinite(mdot) & (t_vir >= atomic_cooling_temperature)
        sfr[active] = baryon_fraction * fstar_now[active] * mdot[active] / YEARS_PER_GYR

    output = {name: values.copy() for name, values in sorted_tracks.items()}
    output["r_vir"] = r_vir
    output["V_c"] = v_c
    output["T_vir"] = t_vir
    output["tau_del"] = tau_del
    output["td_burst"] = td_burst
    output["t_src"] = t_src
    output["Mh_src"] = mh_src
    output["dMh_dt_src"] = mdot_src
    output["fstar_src"] = fstar_src
    output["fstar_now"] = fstar_now
    output["mdot_burst"] = mdot_burst
    output["SFR"] = sfr
    return output

__all__ = ["DEFAULT_SFR_MODEL_PARAMETERS", "SFRModelParameters", "compute_sfr_from_tracks"]
