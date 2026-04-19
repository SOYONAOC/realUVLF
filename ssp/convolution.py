from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .uv1600 import load_uv1600_table


DEFAULT_TIME_UNIT_IN_YEARS = 1.0e9
SSP_UV_LOOKBACK_MAX_MYR = 100.0


def _ensure_1d_float_array(name: str, values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _prepare_sorted_history(
    t_history: np.ndarray,
    mh_history: np.ndarray,
    sfr_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (t_history.size == mh_history.size == sfr_history.size):
        raise ValueError("t_history, mh_history, and sfr_history must have the same length")

    order = np.argsort(t_history, kind="stable")
    t_sorted = t_history[order]
    mh_sorted = mh_history[order]
    sfr_sorted = sfr_history[order]

    if np.any(np.diff(t_sorted) < 0.0):
        raise ValueError("t_history could not be sorted into ascending order")
    if np.any(np.diff(t_sorted) == 0.0):
        raise ValueError("t_history must be strictly increasing after sorting")

    return t_sorted, mh_sorted, sfr_sorted


def _find_mass_crossing_time(t_history: np.ndarray, mh_history: np.ndarray, m_min: float) -> float | None:
    above = mh_history >= m_min
    if not np.any(above):
        return None

    first_above = int(np.flatnonzero(above)[0])
    if first_above == 0:
        return None

    left_index = first_above - 1
    t_left = float(t_history[left_index])
    t_right = float(t_history[first_above])
    mh_left = float(mh_history[left_index])
    mh_right = float(mh_history[first_above])

    if mh_right == mh_left:
        return t_right

    weight = (m_min - mh_left) / (mh_right - mh_left)
    return t_left + weight * (t_right - t_left)


def _augment_with_boundaries(
    x: np.ndarray,
    y: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    interior_mask = (x > lower) & (x < upper)
    x_eval = x[interior_mask]

    if np.isclose(lower, x[0]) or np.any(np.isclose(x_eval, lower)):
        lower_point = np.array([], dtype=float)
    else:
        lower_point = np.array([lower], dtype=float)

    if np.isclose(upper, x[-1]) or np.any(np.isclose(x_eval, upper)):
        upper_point = np.array([], dtype=float)
    else:
        upper_point = np.array([upper], dtype=float)

    x_used = np.concatenate((lower_point, x_eval, upper_point))
    if np.isclose(lower, x[0]):
        x_used = np.concatenate((np.array([x[0]], dtype=float), x_used))
    elif not np.any(np.isclose(x_used, lower)):
        x_used = np.concatenate((np.array([lower], dtype=float), x_used))

    if np.isclose(upper, x[-1]):
        x_used = np.concatenate((x_used, np.array([x[-1]], dtype=float)))
    elif not np.any(np.isclose(x_used, upper)):
        x_used = np.concatenate((x_used, np.array([upper], dtype=float)))

    x_used = np.unique(np.sort(x_used))
    y_used = np.interp(x_used, x, y)
    return x_used, y_used


def interpolate_ssp_luminosity(
    age: float | np.ndarray,
    ssp_age_grid: np.ndarray | list[float] | tuple[float, ...],
    ssp_luv_grid: np.ndarray | list[float] | tuple[float, ...],
) -> float | np.ndarray:
    """Interpolate the SSP UV luminosity kernel on log-age, with old-age contributions truncated to zero."""

    age_grid = _ensure_1d_float_array("ssp_age_grid", ssp_age_grid)
    luv_grid = _ensure_1d_float_array("ssp_luv_grid", ssp_luv_grid)
    if age_grid.size != luv_grid.size:
        raise ValueError("ssp_age_grid and ssp_luv_grid must have the same length")
    if np.any(age_grid <= 0.0):
        raise ValueError("ssp_age_grid must contain strictly positive ages")

    order = np.argsort(age_grid, kind="stable")
    age_grid = age_grid[order]
    luv_grid = luv_grid[order]

    age_array = np.asarray(age, dtype=float)
    if np.any(age_array < 0.0):
        raise ValueError("age must be non-negative")

    result = np.zeros_like(age_array, dtype=float)

    below_mask = age_array < age_grid[0]
    in_range_mask = (age_array >= age_grid[0]) & (age_array <= age_grid[-1])
    result[below_mask] = luv_grid[0]

    if np.any(in_range_mask):
        log_age_grid = np.log10(age_grid)
        log_age = np.log10(age_array[in_range_mask])
        result[in_range_mask] = np.interp(log_age, log_age_grid, luv_grid)

    if np.ndim(age) == 0:
        return float(result)
    return result


def _resolve_uv_kernel_grid(
    uv_kernel: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    if isinstance(uv_kernel, (str, Path)):
        ages_myr, luminosity_per_msun = load_uv1600_table(uv_kernel)
        return ages_myr / 1.0e3, luminosity_per_msun

    if isinstance(uv_kernel, (tuple, list)) and len(uv_kernel) == 2:
        age_grid = _ensure_1d_float_array("uv_kernel age grid", uv_kernel[0])
        luv_grid = _ensure_1d_float_array("uv_kernel luminosity grid", uv_kernel[1])
        if age_grid.size != luv_grid.size:
            raise ValueError("uv_kernel age and luminosity grids must have the same length")
        return age_grid, luv_grid

    if not isinstance(uv_kernel, dict):
        return None

    if "callable" in uv_kernel and callable(uv_kernel["callable"]):
        return None

    age_grid_raw = None
    age_scale_to_gyr = 1.0
    for key in ("ages_gyr", "age_gyr"):
        if key in uv_kernel:
            age_grid_raw = uv_kernel[key]
            age_scale_to_gyr = 1.0
            break
    if age_grid_raw is None:
        for key in ("ages_myr", "age_myr"):
            if key in uv_kernel:
                age_grid_raw = uv_kernel[key]
                age_scale_to_gyr = 1.0e-3
                break
    if age_grid_raw is None:
        for key in ("age_grid", "ages", "age"):
            if key in uv_kernel:
                age_grid_raw = uv_kernel[key]
                age_scale_to_gyr = float(uv_kernel.get("age_unit_in_gyr", 1.0))
                break
    if age_grid_raw is None:
        return None

    luv_grid_raw = None
    for key in ("luminosity_per_msun", "luv_grid", "luv", "kernel"):
        if key in uv_kernel:
            luv_grid_raw = uv_kernel[key]
            break
    if luv_grid_raw is None:
        raise ValueError("uv_kernel dict must include a luminosity grid")

    age_grid = _ensure_1d_float_array("uv_kernel age grid", age_grid_raw) * age_scale_to_gyr
    luv_grid = _ensure_1d_float_array("uv_kernel luminosity grid", luv_grid_raw)
    if age_grid.size != luv_grid.size:
        raise ValueError("uv_kernel age and luminosity grids must have the same length")
    return age_grid, luv_grid


def evaluate_uv_luminosity_kernel(
    age: float | np.ndarray,
    uv_kernel: Any,
) -> float | np.ndarray:
    """Evaluate a UV kernel at the requested age in Gyr.

    The kernel can be provided as:
    - a callable ``kernel(age_gyr)``
    - a ``(age_grid, luminosity_grid)`` tuple/list in Gyr
    - a dict with ``ages_myr``/``ages_gyr`` and ``luminosity_per_msun`` or ``luv_grid``
    - a path to an SSP file understood by ``load_uv1600_table``
    """

    age_array = np.asarray(age, dtype=float)
    if np.any(age_array < 0.0):
        raise ValueError("age must be non-negative")

    if callable(uv_kernel):
        try:
            result = np.asarray(uv_kernel(age_array), dtype=float)
        except Exception:
            vectorized = np.vectorize(lambda single_age: float(uv_kernel(float(single_age))), otypes=[float])
            result = vectorized(age_array)
        if result.shape != age_array.shape:
            result = np.broadcast_to(result, age_array.shape)
    elif isinstance(uv_kernel, dict) and "callable" in uv_kernel and callable(uv_kernel["callable"]):
        return evaluate_uv_luminosity_kernel(age, uv_kernel["callable"])
    else:
        kernel_grid = _resolve_uv_kernel_grid(uv_kernel)
        if kernel_grid is None:
            raise TypeError(
                "uv_kernel must be a callable, a path, a two-array tuple/list, "
                "or a dict containing kernel arrays"
            )
        age_grid_gyr, luv_grid = kernel_grid
        result = np.asarray(
            interpolate_ssp_luminosity(age_array, ssp_age_grid=age_grid_gyr, ssp_luv_grid=luv_grid),
            dtype=float,
        )

    if np.ndim(age) == 0:
        return float(result)
    return np.asarray(result, dtype=float)


def compute_stellar_mass_formed_per_step(
    t_history: np.ndarray | list[float] | tuple[float, ...],
    sfr_history: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Return forward-Euler stellar mass packets from a time-ordered SFR history.

    ``t_history`` and ``sfr_history`` must use consistent time units so that
    ``sfr_history * dt`` yields stellar mass. The final packet is set to zero
    because there is no forward interval beyond the last stored time.
    """

    t_array = _ensure_1d_float_array("t_history", t_history)
    sfr_array = _ensure_1d_float_array("sfr_history", sfr_history)
    if t_array.size != sfr_array.size:
        raise ValueError("t_history and sfr_history must have the same length")
    if np.any(np.diff(t_array) <= 0.0):
        raise ValueError("t_history must be strictly increasing")

    dt = np.diff(t_array, append=t_array[-1])
    mass_packets = np.zeros_like(sfr_array, dtype=float)
    valid = np.isfinite(sfr_array) & (sfr_array > 0.0) & np.isfinite(dt)
    mass_packets[valid] = sfr_array[valid] * dt[valid]
    return mass_packets


def convolve_stellar_mass_history(
    t_history: np.ndarray | list[float] | tuple[float, ...],
    dMstar_history: np.ndarray | list[float] | tuple[float, ...],
    uv_kernel: Any,
    packet_times: np.ndarray | list[float] | tuple[float, ...] | None = None,
) -> np.ndarray:
    """Convolve stellar mass packets with a UV kernel and return the full UV history."""

    t_array = _ensure_1d_float_array("t_history", t_history)
    dMstar_array = _ensure_1d_float_array("dMstar_history", dMstar_history)
    if t_array.size != dMstar_array.size:
        raise ValueError("t_history and dMstar_history must have the same length")
    if np.any(np.diff(t_array) <= 0.0):
        raise ValueError("t_history must be strictly increasing")
    if packet_times is None:
        packet_time_array = t_array
    else:
        packet_time_array = _ensure_1d_float_array("packet_times", packet_times)
        if packet_time_array.size != t_array.size:
            raise ValueError("packet_times must have the same length as t_history")

    age_matrix = t_array[:, None] - packet_time_array[None, :]
    causal = age_matrix >= 0.0
    kernel_matrix = np.zeros_like(age_matrix, dtype=float)
    if np.any(causal):
        kernel_matrix[causal] = np.asarray(evaluate_uv_luminosity_kernel(age_matrix[causal], uv_kernel), dtype=float)
    return np.sum(kernel_matrix * dMstar_array[None, :], axis=1)


def compute_halo_uv_luminosity(
    t_obs: float,
    t_history: np.ndarray | list[float] | tuple[float, ...],
    mh_history: np.ndarray | list[float] | tuple[float, ...],
    sfr_history: np.ndarray | list[float] | tuple[float, ...],
    ssp_age_grid: np.ndarray | list[float] | tuple[float, ...],
    ssp_luv_grid: np.ndarray | list[float] | tuple[float, ...],
    M_min: float,
    t_z50: float,
    time_unit_in_years: float = DEFAULT_TIME_UNIT_IN_YEARS,
    ssp_lookback_max_myr: float = SSP_UV_LOOKBACK_MAX_MYR,
    return_details: bool = False,
) -> float | dict[str, Any]:
    """Convolve a halo SFR history with an SSP UV kernel and return the halo UV luminosity at t_obs."""

    t_history_array = _ensure_1d_float_array("t_history", t_history)
    mh_history_array = _ensure_1d_float_array("mh_history", mh_history)
    sfr_history_array = _ensure_1d_float_array("sfr_history", sfr_history)
    t_sorted, mh_sorted, sfr_sorted = _prepare_sorted_history(
        t_history=t_history_array,
        mh_history=mh_history_array,
        sfr_history=sfr_history_array,
    )

    t_obs = float(t_obs)
    t_z50 = float(t_z50)
    M_min = float(M_min)
    time_unit_in_years = float(time_unit_in_years)
    ssp_lookback_max_gyr = float(ssp_lookback_max_myr) / 1.0e3

    if time_unit_in_years <= 0.0:
        raise ValueError("time_unit_in_years must be positive")
    if ssp_lookback_max_gyr <= 0.0:
        raise ValueError("ssp_lookback_max_myr must be positive")
    if not (t_sorted[0] <= t_obs <= t_sorted[-1]):
        raise ValueError("t_obs must lie within the covered t_history range")

    t_cross = _find_mass_crossing_time(t_sorted, mh_sorted, M_min)
    ti = max(t_z50, t_cross) if t_cross is not None else t_z50
    ti = max(ti, t_sorted[0])
    ti = max(ti, t_obs - ssp_lookback_max_gyr)

    if ti >= t_obs:
        details = {
            "L_uv_halo": 0.0,
            "ti": ti,
            "mask_used": np.zeros_like(t_sorted, dtype=bool),
            "age_used": np.array([], dtype=float),
            "t_used": np.array([], dtype=float),
            "kernel_used": np.array([], dtype=float),
            "integrand_used": np.array([], dtype=float),
            "t_cross_Mmin": t_cross,
        }
        if return_details:
            return details
        return 0.0

    mask_used = (t_sorted >= ti) & (t_sorted <= t_obs)
    t_used, sfr_used = _augment_with_boundaries(t_sorted, sfr_sorted, lower=ti, upper=t_obs)
    age_used = np.maximum(t_obs - t_used, 0.0)
    kernel_used = np.asarray(
        interpolate_ssp_luminosity(age_used, ssp_age_grid=ssp_age_grid, ssp_luv_grid=ssp_luv_grid),
        dtype=float,
    )
    integrand_used = sfr_used * kernel_used
    t_used_years = t_used * time_unit_in_years
    l_uv_halo = float(np.trapezoid(integrand_used, x=t_used_years))

    details = {
        "L_uv_halo": l_uv_halo,
        "ti": ti,
        "ssp_lookback_max_myr": float(ssp_lookback_max_myr),
        "mask_used": mask_used,
        "age_used": age_used,
        "t_used": t_used,
        "kernel_used": kernel_used,
        "integrand_used": integrand_used,
        "t_cross_Mmin": t_cross,
    }
    if return_details:
        return details
    return l_uv_halo
