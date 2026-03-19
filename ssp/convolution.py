from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_TIME_UNIT_IN_YEARS = 1.0e9


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

    if time_unit_in_years <= 0.0:
        raise ValueError("time_unit_in_years must be positive")
    if not (t_sorted[0] <= t_obs <= t_sorted[-1]):
        raise ValueError("t_obs must lie within the covered t_history range")

    t_cross = _find_mass_crossing_time(t_sorted, mh_sorted, M_min)
    ti = max(t_z50, t_cross) if t_cross is not None else t_z50
    ti = max(ti, t_sorted[0])

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
