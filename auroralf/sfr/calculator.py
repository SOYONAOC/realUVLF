"""Pop II star-formation calculation along halo assembly histories.

The efficiency structure ``SFR = f_b f_star(Mh) dMh/dt`` with a double-power-
law ``f_star`` is adapted from Sun & Furlanetto (2016), DOI:
10.1093/mnras/stw980, arXiv:1512.06219, and Mirocha et al. (2017), DOI:
10.1093/mnras/stw2412, arXiv:1607.00386.  The delayed-SFR kernel is adapted
from Yue, Ferrara & Xu (2016), DOI: 10.1093/mnras/stw2145,
arXiv:1604.01314.

The numerical values ``epsilon_0=0.12``, ``Mc=10**11.7 Msun``, ``beta=0.66``,
and ``gamma=0.65`` are AuroraLF's z=6 UVLF calibration.  They are not quoted
parameter values from the papers above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from auroralf.constants import (
    BOLTZMANN_CONSTANT_J_K,
    GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN,
    KM_PER_MPC,
    PROTON_MASS_KG,
    SECONDS_PER_GYR,
    YEARS_PER_GYR,
)
from auroralf.mah.models import (
    Cosmology,
)
from auroralf.mah.physics import compute_bryan_norman_virial_terms


EPSILON_0 = 0.12
CHARACTERISTIC_MASS = 10.0**11.7
BETA_STAR = 0.66
GAMMA_STAR = 0.65
EXTENDED_BURST_KAPPA = 0.1
EXTENDED_BURST_LOOKBACK_MAX_MYR = 100.0


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
    """Evaluate the literature-motivated double-power-law efficiency.

    Formula structure: Sun & Furlanetto (2016) / Mirocha et al. (2017).
    Parameter values: calibrated in this work against the z=6 UVLF.
    """
    ratio = np.asarray(mass, dtype=float) / model_parameters.characteristic_mass
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        denominator = ratio ** (-model_parameters.beta_star) + ratio**model_parameters.gamma_star
    return 2.0 * model_parameters.epsilon_0 / denominator


def _extended_burst_kernel(delta_t_gyr: np.ndarray, td_gyr: float, kappa: float = EXTENDED_BURST_KAPPA) -> np.ndarray:
    """Evaluate the delayed-SFR kernel adapted from Yue et al. (2016).

    ``kappa=0.1`` and AuroraLF's finite 100 Myr integration window are project
    settings rather than independently fitted parameters from that paper.
    """
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


def _compute_extended_burst_convolution_direct(
    t_grid, source_grid, active_grid, td_burst_grid, kappa, max_lookback_gyr,
):
    """Uniform-time convolution with the same causal trapezoidal endpoints.

    Explicit opt-in numerical backend. Uses direct positive convolution (not
    FFT); no kernel fitting, changed window or changed physical parameters.
    """
    time = np.asarray(t_grid[0], float)
    steps = np.diff(time)
    if steps.size == 0 or np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-11, atol=0):
        raise ValueError('direct convolution requires a uniform increasing time grid')
    dt = (time[-1]-time[0])/(len(time)-1)
    lags = np.arange(len(time))*dt
    # At an ambiguous floating-point cutoff, preserve the dense definition by
    # rejecting this backend rather than silently changing kernel membership.
    if np.any(np.abs(lags-max_lookback_gyr) < 64*np.finfo(float).eps*max(1., abs(time[-1]))):
        raise ValueError('lookback cutoff coincides with a floating-point grid boundary')
    result = np.zeros_like(source_grid, dtype=float)
    for i in np.flatnonzero(np.any(active_grid, axis=1)):
        start = int(np.flatnonzero(active_grid[i])[0])
        td = float(td_burst_grid[i, start])
        if not np.isfinite(td) or td <= 0 or len(time)-start < 2:
            continue
        lag = lags[:len(time)-start]
        kernel = lag/(kappa**2*td**2)*np.exp(-lag/(kappa*td))
        kernel[lag>max_lookback_gyr] = 0
        source = np.array(source_grid[i, start:], copy=True)
        source[0] *= .5
        # Last integration endpoint has kernel(0)=0; first has weight 1/2.
        result[i, start:] = dt*np.convolve(source, kernel, mode='full')[:len(source)]
    return result


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


def _validate_accretion_rate_columns(tracks: dict[str, np.ndarray], n_rows: int) -> None:
    if n_rows == 0:
        raise ValueError("tracks contains no halo history rows")

    rate_columns: dict[str, np.ndarray] = {}
    for name in ("dMh_dt_raw", "dMh_dt_sfr", "dMh_dt_clipped"):
        values = np.asarray(tracks[name])
        if values.ndim != 1:
            raise ValueError(f"tracks column '{name}' must be a 1D array")
        if values.size != n_rows:
            raise ValueError(f"tracks column '{name}' does not match halo_id length")
        rate_columns[name] = values

    clipped = rate_columns["dMh_dt_clipped"]
    if clipped.dtype != np.dtype(np.bool_):
        raise ValueError("tracks column 'dMh_dt_clipped' must have bool dtype")

    numeric_rates: dict[str, np.ndarray] = {}
    for name in ("dMh_dt_raw", "dMh_dt_sfr"):
        try:
            values = np.asarray(rate_columns[name], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"tracks column '{name}' must contain finite numeric values") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f"tracks column '{name}' must contain only finite values")
        numeric_rates[name] = values

    raw = numeric_rates["dMh_dt_raw"]
    effective = numeric_rates["dMh_dt_sfr"]
    if np.any(effective < 0.0):
        raise ValueError("dMh_dt_sfr must be non-negative")
    if not np.array_equal(effective, np.maximum(raw, 0.0)):
        raise ValueError("dMh_dt_sfr must equal maximum(dMh_dt_raw, 0)")
    if not np.array_equal(clipped, raw < 0.0):
        raise ValueError("dMh_dt_clipped must equal dMh_dt_raw < 0")


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
    *,
    cosmology: Cosmology,
    mu: float = 0.61,
    atomic_cooling_temperature: float = 1.0e4,
    enable_time_delay: bool = False,
    burst_kappa: float = EXTENDED_BURST_KAPPA,
    burst_lookback_max_myr: float = EXTENDED_BURST_LOOKBACK_MAX_MYR,
    model_parameters: SFRModelParameters | None = None,
    regular_convolution_backend: str = "dense",
) -> dict[str, np.ndarray]:
    """Compute SFR in Msun/yr and related virial quantities from halo tracks.

    The Pop II efficiency structure and calibrated/project-specific parameter
    boundary are documented in this module's citation provenance above.
    ``regular_convolution_backend='direct'`` explicitly selects the equivalent
    uniform-grid 1D convolution. Default ``dense`` remains unchanged.
    """

    if not isinstance(cosmology, Cosmology):
        raise TypeError("cosmology must be an instance of auroralf.mah.models.Cosmology")
    if regular_convolution_backend not in ('dense', 'direct'):
        raise ValueError('unknown regular_convolution_backend')

    required = (
        "halo_id",
        "step",
        "z",
        "t_gyr",
        "Mh",
        "dMh_dt_raw",
        "dMh_dt_sfr",
        "dMh_dt_clipped",
    )
    missing = [name for name in required if name not in tracks]
    if missing:
        raise KeyError(f"tracks is missing required columns: {missing}")

    n_rows = int(np.asarray(tracks["halo_id"]).size)
    _validate_accretion_rate_columns(tracks, n_rows)
    for name in required:
        if np.asarray(tracks[name]).size != n_rows:
            raise ValueError(f"tracks column '{name}' does not match halo_id length")

    model_parameters = _resolve_sfr_model_parameters(model_parameters)
    max_burst_lookback_gyr = float(burst_lookback_max_myr) / 1.0e3
    if max_burst_lookback_gyr <= 0.0:
        raise ValueError("burst_lookback_max_myr must be positive")
    baryon_fraction = cosmology.omega_b / cosmology.omega_m
    sorted_tracks, _, boundaries, group_ids = _prepare_track_columns(tracks)
    z = np.asarray(sorted_tracks["z"], dtype=float)
    t_gyr = np.asarray(sorted_tracks["t_gyr"], dtype=float)
    mass = np.asarray(sorted_tracks["Mh"], dtype=float)
    mdot = np.asarray(sorted_tracks["dMh_dt_sfr"], dtype=float)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        expansion_squared, _, delta_vir = compute_bryan_norman_virial_terms(
            z,
            cosmology=cosmology,
        )
        hubble = cosmology.h0_km_s_mpc * np.sqrt(expansion_squared)
        rho_crit = cosmology.rhocrit * (hubble / cosmology.h0_km_s_mpc) ** 2
        rho_vir = delta_vir * rho_crit

        # Use Msun and Mpc consistently so virial quantities remain numerically stable.
        r_vir = (3.0 * mass / (4.0 * np.pi * rho_vir)) ** (1.0 / 3.0)
        v_c = np.sqrt(GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN * mass / r_vir)
        t_vir = mu * PROTON_MASS_KG * (v_c * 1.0e3) ** 2 / (2.0 * BOLTZMANN_CONSTANT_J_K)
        tau_del = r_vir * KM_PER_MPC / v_c / SECONDS_PER_GYR
        td_burst = np.sqrt(
            3.0 * np.pi / (32.0 * GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN * rho_vir)
        )
        td_burst = td_burst * KM_PER_MPC / SECONDS_PER_GYR

    core_virial_quantities = {
        "r_vir": r_vir,
        "V_c": v_c,
        "T_vir": t_vir,
        "tau_del": tau_del,
        "td_burst": td_burst,
    }
    if any(not np.all(np.isfinite(values)) for values in core_virial_quantities.values()):
        raise RuntimeError("SFR core physical quantities must be finite")

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
        t_grid = _reshape_grouped_regular_grid(t_gyr, boundaries)
        mdot_grid = _reshape_grouped_regular_grid(mdot, boundaries)
        td_grid = _reshape_grouped_regular_grid(td_burst, boundaries)
        source_rate_grid = _reshape_grouped_regular_grid(fstar_now * mdot, boundaries)
        active_grid = _reshape_grouped_regular_grid(active_now.astype(float), boundaries)
        if (
            t_grid is not None
            and mdot_grid is not None
            and td_grid is not None
            and source_rate_grid is not None
            and active_grid is not None
            and np.all(np.isfinite(t_grid))
            and np.allclose(t_grid, t_grid[0], rtol=0.0, atol=0.0)
        ):
            convolver = (_compute_extended_burst_convolution_direct if regular_convolution_backend == 'direct'
                         else _compute_extended_burst_convolution_vectorized_regular_grid)
            mdot_burst = convolver(
                t_grid=t_grid,
                source_grid=mdot_grid,
                active_grid=active_grid.astype(bool),
                td_burst_grid=td_grid,
                kappa=float(burst_kappa),
                max_lookback_gyr=max_burst_lookback_gyr,
            ).reshape(-1)
            sfr_source_burst = convolver(
                t_grid=t_grid,
                source_grid=source_rate_grid,
                active_grid=active_grid.astype(bool),
                td_burst_grid=td_grid,
                kappa=float(burst_kappa),
                max_lookback_gyr=max_burst_lookback_gyr,
            ).reshape(-1)
        else:
            if regular_convolution_backend == 'direct':
                raise ValueError('direct convolution requires grouped shared time grids')
            mdot_burst = _compute_extended_burst_convolution(
                t_gyr=t_gyr,
                source_values=mdot,
                active=active_now,
                boundaries=boundaries,
                kappa=float(burst_kappa),
                td_burst=td_burst,
                max_lookback_gyr=max_burst_lookback_gyr,
            )
            sfr_source_burst = _compute_extended_burst_convolution(
                t_gyr=t_gyr,
                source_values=fstar_now * mdot,
                active=active_now,
                boundaries=boundaries,
                kappa=float(burst_kappa),
                td_burst=td_burst,
                max_lookback_gyr=max_burst_lookback_gyr,
            )
        active_burst = active_now & np.isfinite(sfr_source_burst)
        sfr[active_burst] = baryon_fraction * sfr_source_burst[active_burst] / YEARS_PER_GYR
    else:
        mdot_burst = np.full_like(mass, np.nan, dtype=float)
        active = np.isfinite(fstar_now) & np.isfinite(mdot) & (t_vir >= atomic_cooling_temperature)
        sfr[active] = baryon_fraction * fstar_now[active] * mdot[active] / YEARS_PER_GYR

    output = {name: values.copy() for name, values in sorted_tracks.items()}
    output.update(core_virial_quantities)
    output["t_src"] = t_src
    output["Mh_src"] = mh_src
    output["dMh_dt_sfr_src"] = mdot_src
    output["fstar_src"] = fstar_src
    output["fstar_now"] = fstar_now
    output["mdot_burst"] = mdot_burst
    output["SFR"] = sfr
    if not np.all(np.isfinite(sfr)):
        raise RuntimeError("SFR core physical quantities must be finite")
    if np.any(sfr < 0.0):
        raise RuntimeError("SFR calculation returned a negative value")
    return output

__all__ = [
    "DEFAULT_SFR_MODEL_PARAMETERS",
    "EXTENDED_BURST_LOOKBACK_MAX_MYR",
    "EXTENDED_BURST_KAPPA",
    "SFRModelParameters",
    "compute_sfr_from_tracks",
]
