from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constants import (
    BOLTZMANN_CONSTANT_J_K,
    GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN,
    KM_PER_MPC,
    PROTON_MASS_KG,
    SECONDS_PER_GYR,
    YEARS_PER_GYR,
)
from mah import CosmologySet


EPSILON_0 = 0.12
CHARACTERISTIC_MASS = 10.0**11.7
BETA_STAR = 0.66
GAMMA_STAR = 0.65
EXTENDED_BURST_KAPPA = 0.1
EXTENDED_BURST_LOOKBACK_MAX_MYR = 100.0
STREAMING_VELOCITY_RMS_RECOMBINATION_KMS = 30.0
STREAMING_VELOCITY_RECOMBINATION_REDSHIFT = 1089.0
H2_COOLING_A_KMS = 3.714
H2_COOLING_B_KMS = 4.015


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


def _validate_positive_timescale(name: str, value: float) -> float:
    timescale = float(value)
    if not np.isfinite(timescale) or timescale <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return timescale


def atomic_cooling_circular_velocity(
    temperature_k: float = 1.0e4,
    mu: float = 0.61,
) -> float:
    if float(temperature_k) <= 0.0:
        raise ValueError("temperature_k must be positive")
    if float(mu) <= 0.0:
        raise ValueError("mu must be positive")
    return float(np.sqrt(2.0 * BOLTZMANN_CONSTANT_J_K * float(temperature_k) / (float(mu) * PROTON_MASS_KG)) / 1.0e3)


def atomic_cooling_mass_threshold(
    redshift: float | np.ndarray,
    mu: float = 0.61,
    atomic_cooling_temperature: float = 1.0e4,
    cosmology: CosmologySet | None = None,
) -> np.ndarray:
    v_atom = atomic_cooling_circular_velocity(temperature_k=atomic_cooling_temperature, mu=mu)
    return virial_mass_from_circular_velocity(v_atom, redshift, cosmology=cosmology)


def _hubble_parameter(redshift: np.ndarray, cosmology: CosmologySet) -> np.ndarray:
    return cosmology.H0u * np.sqrt(cosmology.omegam * (1.0 + redshift) ** 3 + cosmology.omegalam)


def omega_matter_at_redshift(redshift: float | np.ndarray, cosmology: CosmologySet | None = None) -> np.ndarray:
    cosmo = CosmologySet() if cosmology is None else cosmology
    z = np.asarray(redshift, dtype=float)
    hubble_ratio_sq = (_hubble_parameter(z, cosmo) / cosmo.H0u) ** 2
    return cosmo.omegam * (1.0 + z) ** 3 / hubble_ratio_sq


def virial_overdensity(redshift: float | np.ndarray, cosmology: CosmologySet | None = None) -> np.ndarray:
    omega_m_z = omega_matter_at_redshift(redshift, cosmology=cosmology)
    delta = omega_m_z - 1.0
    return 18.0 * np.pi**2 + 82.0 * delta - 39.0 * delta**2


def critical_density(redshift: float | np.ndarray, cosmology: CosmologySet | None = None) -> np.ndarray:
    cosmo = CosmologySet() if cosmology is None else cosmology
    z = np.asarray(redshift, dtype=float)
    hubble = _hubble_parameter(z, cosmo)
    return cosmo.rhocrit * (hubble / cosmo.H0u) ** 2


def virial_mass_from_circular_velocity(
    circular_velocity_kms: float | np.ndarray,
    redshift: float | np.ndarray,
    cosmology: CosmologySet | None = None,
) -> np.ndarray:
    cosmo = CosmologySet() if cosmology is None else cosmology
    vc = np.asarray(circular_velocity_kms, dtype=float)
    rho_crit = critical_density(redshift, cosmology=cosmo)
    delta_vir = virial_overdensity(redshift, cosmology=cosmo)
    prefactor = (GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN ** 1.5) * np.sqrt(4.0 * np.pi * delta_vir * rho_crit / 3.0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return vc**3 / prefactor


def streaming_velocity_rms(
    redshift: float | np.ndarray,
    sigma_vbc_recombination_kms: float = STREAMING_VELOCITY_RMS_RECOMBINATION_KMS,
    z_recombination: float = STREAMING_VELOCITY_RECOMBINATION_REDSHIFT,
) -> np.ndarray:
    z = np.asarray(redshift, dtype=float)
    return float(sigma_vbc_recombination_kms) * (1.0 + z) / (1.0 + float(z_recombination))


def h2_cooling_circular_velocity(
    redshift: float | np.ndarray,
    v_bc_kms: float | np.ndarray | None = None,
    sigma_vbc_recombination_kms: float = STREAMING_VELOCITY_RMS_RECOMBINATION_KMS,
    z_recombination: float = STREAMING_VELOCITY_RECOMBINATION_REDSHIFT,
    a_kms: float = H2_COOLING_A_KMS,
    b_kms: float = H2_COOLING_B_KMS,
) -> np.ndarray:
    if v_bc_kms is None:
        v_bc = streaming_velocity_rms(
            redshift,
            sigma_vbc_recombination_kms=sigma_vbc_recombination_kms,
            z_recombination=z_recombination,
        )
    else:
        v_bc = np.asarray(v_bc_kms, dtype=float)
    return np.sqrt(float(a_kms) ** 2 + (float(b_kms) * v_bc) ** 2)


def minihalo_mass_floor(
    redshift: float | np.ndarray,
    v_bc_kms: float | np.ndarray | None = None,
    cosmology: CosmologySet | None = None,
    sigma_vbc_recombination_kms: float = STREAMING_VELOCITY_RMS_RECOMBINATION_KMS,
    z_recombination: float = STREAMING_VELOCITY_RECOMBINATION_REDSHIFT,
    a_kms: float = H2_COOLING_A_KMS,
    b_kms: float = H2_COOLING_B_KMS,
) -> np.ndarray:
    v_cool = h2_cooling_circular_velocity(
        redshift,
        v_bc_kms=v_bc_kms,
        sigma_vbc_recombination_kms=sigma_vbc_recombination_kms,
        z_recombination=z_recombination,
        a_kms=a_kms,
        b_kms=b_kms,
    )
    return virial_mass_from_circular_velocity(v_cool, redshift, cosmology=cosmology)


def _extended_burst_kernel(delta_t_gyr: np.ndarray, td_gyr: float, kappa: float = EXTENDED_BURST_KAPPA) -> np.ndarray:
    delta_t_gyr = np.asarray(delta_t_gyr, dtype=float)
    kernel = np.zeros_like(delta_t_gyr, dtype=float)
    positive = (delta_t_gyr >= 0.0) & np.isfinite(delta_t_gyr) & np.isfinite(td_gyr) & (td_gyr > 0.0)
    if not np.any(positive):
        return kernel
    x = delta_t_gyr[positive]
    kernel[positive] = x / (kappa**2 * td_gyr**2) * np.exp(-x / (kappa * td_gyr))
    return kernel


def _compute_extended_burst_convolution(
    t_gyr: np.ndarray,
    source_values: np.ndarray,
    active: np.ndarray,
    boundaries: np.ndarray,
    kappa: float,
    td_burst: np.ndarray,
    max_lookback_gyr: float,
) -> np.ndarray:
    convolved = np.zeros_like(source_values, dtype=float)
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
        source_group = np.asarray(source_values[first:end], dtype=float)
        active_slice = np.asarray(active[first:end], dtype=bool)

        for local_i in range(t_group.size):
            if not active_slice[local_i]:
                continue
            t_now = float(t_group[local_i])
            t_src = t_group[: local_i + 1]
            source_src = source_group[: local_i + 1]
            valid = np.isfinite(t_src) & np.isfinite(source_src)
            if np.count_nonzero(valid) < 2:
                continue
            delta_t = t_now - t_src[valid]
            valid_window = delta_t <= max_lookback_gyr
            if np.count_nonzero(valid_window) < 2:
                continue
            delta_t = delta_t[valid_window]
            t_valid = t_src[valid][valid_window]
            source_valid = source_src[valid][valid_window]
            kernel = _extended_burst_kernel(delta_t, td_group, kappa=kappa)
            convolved[first + local_i] = np.trapezoid(kernel * source_valid, x=t_valid)
    return convolved


def _reshape_grouped_regular_grid(
    values: np.ndarray,
    boundaries: np.ndarray,
) -> np.ndarray | None:
    counts = np.diff(boundaries)
    if counts.size == 0 or np.any(counts != counts[0]):
        return None
    return np.asarray(values, dtype=float).reshape(counts.size, counts[0])


def _compute_extended_burst_convolution_vectorized_regular_grid(
    t_grid: np.ndarray,
    source_grid: np.ndarray,
    active_grid: np.ndarray,
    td_burst_grid: np.ndarray,
    kappa: float,
    max_lookback_gyr: float,
) -> np.ndarray:
    n_halos, n_steps = source_grid.shape
    result = np.zeros_like(source_grid, dtype=float)
    valid_halos = np.any(active_grid, axis=1)
    if not np.any(valid_halos):
        return result

    first_active = np.argmax(active_grid, axis=1)
    halo_index = np.arange(n_halos, dtype=int)
    td_per_halo = td_burst_grid[halo_index, first_active]
    valid_td = valid_halos & np.isfinite(td_per_halo) & (td_per_halo > 0.0)
    if not np.any(valid_td):
        return result

    time_row = np.asarray(t_grid[0], dtype=float)
    for halo_id in np.flatnonzero(valid_td):
        start_index = int(first_active[halo_id])
        t_local = time_row[start_index:]
        source_local = np.asarray(source_grid[halo_id, start_index:], dtype=float)
        if t_local.size < 2:
            continue

        delta_t = t_local[:, None] - t_local[None, :]
        causal_mask = delta_t >= 0.0
        window_mask = delta_t <= max_lookback_gyr
        valid_mask = causal_mask & window_mask
        causal_delta_t = np.where(causal_mask, delta_t, 0.0)
        td_scale = float(td_per_halo[halo_id])
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            kernel = causal_delta_t / (kappa**2 * td_scale**2)
            kernel *= np.exp(-causal_delta_t / (kappa * td_scale))
        kernel *= valid_mask
        integrand = kernel * source_local[None, :]
        result[halo_id, start_index:] = np.trapezoid(integrand, x=t_local, axis=1)

    return result


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
    burst_lookback_max_myr: float = EXTENDED_BURST_LOOKBACK_MAX_MYR,
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
    max_burst_lookback_gyr = float(burst_lookback_max_myr) / 1.0e3
    if max_burst_lookback_gyr <= 0.0:
        raise ValueError("burst_lookback_max_myr must be positive")
    cosmo = CosmologySet()
    baryon_fraction = cosmo.omegab / cosmo.omegam
    sorted_tracks, _, boundaries, group_ids = _prepare_track_columns(tracks)
    z = np.asarray(sorted_tracks["z"], dtype=float)
    t_gyr = np.asarray(sorted_tracks["t_gyr"], dtype=float)
    mass = np.asarray(sorted_tracks["Mh"], dtype=float)
    mdot = np.asarray(sorted_tracks["dMh_dt"], dtype=float)

    rho_crit = critical_density(z, cosmology=cosmo)
    delta_vir = virial_overdensity(z, cosmology=cosmo)
    rho_vir = delta_vir * rho_crit

    # Use Msun and Mpc consistently so virial quantities remain numerically stable.
    r_vir = (3.0 * mass / (4.0 * np.pi * delta_vir * rho_crit)) ** (1.0 / 3.0)
    v_c = np.sqrt(GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN * mass / r_vir)
    t_vir = mu * PROTON_MASS_KG * (v_c * 1.0e3) ** 2 / (2.0 * BOLTZMANN_CONSTANT_J_K)
    sigma_vbc_rms = streaming_velocity_rms(z)
    v_cool_h2 = h2_cooling_circular_velocity(z, v_bc_kms=sigma_vbc_rms)
    m_cool_h2 = minihalo_mass_floor(z, v_bc_kms=sigma_vbc_rms, cosmology=cosmo)
    m_atom = atomic_cooling_mass_threshold(
        z,
        mu=mu,
        atomic_cooling_temperature=atomic_cooling_temperature,
        cosmology=cosmo,
    )
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
    sfr_pop2 = np.zeros_like(mass, dtype=float)
    pop2_active_now = np.isfinite(mass) & np.isfinite(mdot) & (t_vir >= atomic_cooling_temperature)
    if enable_time_delay:
        t_grid = _reshape_grouped_regular_grid(t_gyr, boundaries)
        mdot_grid = _reshape_grouped_regular_grid(mdot, boundaries)
        td_grid = _reshape_grouped_regular_grid(td_burst, boundaries)
        source_rate = np.zeros_like(mass, dtype=float)
        valid_pop2_source = pop2_active_now & np.isfinite(fstar_now)
        source_rate[valid_pop2_source] = fstar_now[valid_pop2_source] * mdot[valid_pop2_source]
        source_rate_grid = _reshape_grouped_regular_grid(source_rate, boundaries)
        active_grid = _reshape_grouped_regular_grid(pop2_active_now.astype(float), boundaries)
        if (
            t_grid is not None
            and mdot_grid is not None
            and td_grid is not None
            and source_rate_grid is not None
            and active_grid is not None
            and np.all(np.isfinite(t_grid))
            and np.allclose(t_grid, t_grid[0], rtol=0.0, atol=0.0)
        ):
            mdot_burst = _compute_extended_burst_convolution_vectorized_regular_grid(
                t_grid=t_grid,
                source_grid=mdot_grid,
                active_grid=active_grid.astype(bool),
                td_burst_grid=td_grid,
                kappa=float(burst_kappa),
                max_lookback_gyr=max_burst_lookback_gyr,
            ).reshape(-1)
            sfr_source_burst = _compute_extended_burst_convolution_vectorized_regular_grid(
                t_grid=t_grid,
                source_grid=source_rate_grid,
                active_grid=active_grid.astype(bool),
                td_burst_grid=td_grid,
                kappa=float(burst_kappa),
                max_lookback_gyr=max_burst_lookback_gyr,
            ).reshape(-1)
        else:
            mdot_burst = _compute_extended_burst_convolution(
                t_gyr=t_gyr,
                source_values=mdot,
                active=pop2_active_now,
                boundaries=boundaries,
                kappa=float(burst_kappa),
                td_burst=td_burst,
                max_lookback_gyr=max_burst_lookback_gyr,
            )
            sfr_source_burst = _compute_extended_burst_convolution(
                t_gyr=t_gyr,
                source_values=source_rate,
                active=pop2_active_now,
                boundaries=boundaries,
                kappa=float(burst_kappa),
                td_burst=td_burst,
                max_lookback_gyr=max_burst_lookback_gyr,
            )
        active_burst = pop2_active_now & np.isfinite(sfr_source_burst)
        sfr_pop2[active_burst] = baryon_fraction * sfr_source_burst[active_burst] / YEARS_PER_GYR
    else:
        mdot_burst = np.full_like(mass, np.nan, dtype=float)
        active = pop2_active_now & np.isfinite(fstar_now)
        sfr_pop2[active] = baryon_fraction * fstar_now[active] * mdot[active] / YEARS_PER_GYR

    sfr = np.maximum(sfr_pop2, 0.0)
    pop2_active_flag = pop2_active_now
    branch_active_flag = pop2_active_flag

    output = {name: values.copy() for name, values in sorted_tracks.items()}
    output["r_vir"] = r_vir
    output["V_c"] = v_c
    output["T_vir"] = t_vir
    output["sigma_vbc_rms"] = sigma_vbc_rms
    output["V_cool_H2"] = v_cool_h2
    output["M_cool_H2"] = m_cool_h2
    output["M_atom"] = m_atom
    output["tau_del"] = tau_del
    output["td_burst"] = td_burst
    output["t_src"] = t_src
    output["Mh_src"] = mh_src
    output["dMh_dt_src"] = mdot_src
    output["fstar_src"] = fstar_src
    output["fstar_now"] = fstar_now
    output["pop2_active_flag"] = pop2_active_flag
    output["branch_active_flag"] = branch_active_flag
    output["SFR_pop2"] = np.maximum(sfr_pop2, 0.0)
    output["mdot_burst"] = mdot_burst
    output["SFR_total"] = sfr
    output["SFR"] = sfr
    return output

__all__ = [
    "DEFAULT_SFR_MODEL_PARAMETERS",
    "EXTENDED_BURST_LOOKBACK_MAX_MYR",
    "EXTENDED_BURST_KAPPA",
    "H2_COOLING_A_KMS",
    "H2_COOLING_B_KMS",
    "SFRModelParameters",
    "STREAMING_VELOCITY_RECOMBINATION_REDSHIFT",
    "STREAMING_VELOCITY_RMS_RECOMBINATION_KMS",
    "atomic_cooling_circular_velocity",
    "atomic_cooling_mass_threshold",
    "compute_sfr_from_tracks",
    "critical_density",
    "h2_cooling_circular_velocity",
    "minihalo_mass_floor",
    "omega_matter_at_redshift",
    "streaming_velocity_rms",
    "virial_mass_from_circular_velocity",
    "virial_overdensity",
]
