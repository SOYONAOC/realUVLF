from __future__ import annotations

import numpy as np


DEFAULT_C0 = 2.10
DEFAULT_C1 = 4.85
DEFAULT_M0 = -19.5


def _interp_log10_with_linear_extrapolation(
    x_query: np.ndarray,
    x: np.ndarray,
    y_log10: np.ndarray,
) -> np.ndarray:
    """Interpolate log10(y) and linearly extrapolate beyond both edges."""

    interpolated = np.interp(x_query, x, y_log10)

    left_mask = x_query < x[0]
    if np.any(left_mask):
        left_slope = (y_log10[1] - y_log10[0]) / (x[1] - x[0])
        interpolated[left_mask] = y_log10[0] + left_slope * (x_query[left_mask] - x[0])

    right_mask = x_query > x[-1]
    if np.any(right_mask):
        right_slope = (y_log10[-1] - y_log10[-2]) / (x[-1] - x[-2])
        interpolated[right_mask] = y_log10[-1] + right_slope * (x_query[right_mask] - x[-1])

    return interpolated


def uv_continuum_slope_beta(
    muv_obs: np.ndarray | float,
    z: float,
    *,
    m0: float = DEFAULT_M0,
) -> np.ndarray | float:
    """Return the UV continuum slope beta(M_UV^obs, z)."""

    muv_obs_array = np.asarray(muv_obs, dtype=float)
    beta0 = -0.09 * z - 1.49
    dbeta_dmuv = -0.007 * z - 0.09
    beta = beta0 + dbeta_dmuv * (muv_obs_array - m0)
    if np.ndim(muv_obs) == 0:
        return float(beta)
    return beta


def uv_dust_attenuation(
    muv_obs: np.ndarray | float,
    z: float,
    *,
    c0: float = DEFAULT_C0,
    c1: float = DEFAULT_C1,
    m0: float = DEFAULT_M0,
) -> np.ndarray | float:
    """Return A_UV(M_UV^obs, z) with the non-negative attenuation floor."""

    beta = np.asarray(uv_continuum_slope_beta(muv_obs, z, m0=m0), dtype=float)
    attenuation = np.maximum(c1 + c0 * beta, 0.0)
    if np.ndim(muv_obs) == 0:
        return float(attenuation)
    return attenuation


def intrinsic_muv_from_observed(
    muv_obs: np.ndarray | float,
    z: float,
    *,
    c0: float = DEFAULT_C0,
    c1: float = DEFAULT_C1,
    m0: float = DEFAULT_M0,
) -> np.ndarray | float:
    """Map observed UV magnitude to intrinsic UV magnitude."""

    muv_obs_array = np.asarray(muv_obs, dtype=float)
    intrinsic = muv_obs_array - np.asarray(uv_dust_attenuation(muv_obs_array, z, c0=c0, c1=c1, m0=m0), dtype=float)
    if np.ndim(muv_obs) == 0:
        return float(intrinsic)
    return intrinsic


def intrinsic_muv_jacobian(
    muv_obs: np.ndarray | float,
    z: float,
    *,
    c0: float = DEFAULT_C0,
    c1: float = DEFAULT_C1,
    m0: float = DEFAULT_M0,
) -> np.ndarray | float:
    """Return dM_UV / dM_UV^obs for the dust-corrected mapping."""

    muv_obs_array = np.asarray(muv_obs, dtype=float)
    jacobian = np.full_like(muv_obs_array, 1.09 + 0.007 * z, dtype=float)
    if np.ndim(muv_obs) == 0:
        return float(jacobian)
    return jacobian


def compute_dust_attenuated_uvlf(
    intrinsic_muv: np.ndarray,
    intrinsic_phi: np.ndarray,
    z: float,
    *,
    muv_obs: np.ndarray | None = None,
    c0: float = DEFAULT_C0,
    c1: float = DEFAULT_C1,
    m0: float = DEFAULT_M0,
    clip_to_bounds: bool = False,
    match_faint_end_after_intersection: bool = True,
    insert_transition_point: bool = False,
) -> dict[str, np.ndarray]:
    """
    Transform an intrinsic UVLF into an observed UVLF with dust attenuation.

    The mapping uses:
      M_UV = M_UV^obs - A_UV(M_UV^obs, z)
      phi_obs_raw(M_UV^obs) = phi_int(M_UV) * dM_UV / dM_UV^obs

    The current implementation follows the formula directly:
      phi_obs(M_UV^obs) = phi_int(M_UV^obs - A_UV) * dM_UV / dM_UV^obs

    `match_faint_end_after_intersection` and `insert_transition_point` are retained only
    for call-site compatibility.
    """

    intrinsic_muv_array = np.asarray(intrinsic_muv, dtype=float)
    intrinsic_phi_array = np.asarray(intrinsic_phi, dtype=float)
    if intrinsic_muv_array.ndim != 1 or intrinsic_phi_array.ndim != 1:
        raise ValueError("intrinsic_muv and intrinsic_phi must both be 1D arrays")
    if intrinsic_muv_array.size != intrinsic_phi_array.size:
        raise ValueError("intrinsic_muv and intrinsic_phi must have the same length")

    observed_grid = intrinsic_muv_array.copy() if muv_obs is None else np.asarray(muv_obs, dtype=float)
    if observed_grid.ndim != 1:
        raise ValueError("muv_obs must be a 1D array when provided")

    positive = np.isfinite(intrinsic_muv_array) & np.isfinite(intrinsic_phi_array) & (intrinsic_phi_array > 0.0)
    if np.count_nonzero(positive) < 2:
        raise RuntimeError("at least two positive intrinsic UVLF points are required for dust correction")

    model_x = intrinsic_muv_array[positive]
    model_y = intrinsic_phi_array[positive]
    order = np.argsort(model_x)
    model_x = model_x[order]
    model_y = model_y[order]
    log_model_y = np.log10(model_y)

    intrinsic_from_obs = np.asarray(
        intrinsic_muv_from_observed(observed_grid, z, c0=c0, c1=c1, m0=m0),
        dtype=float,
    )
    jacobian = np.asarray(intrinsic_muv_jacobian(observed_grid, z, c0=c0, c1=c1, m0=m0), dtype=float)
    attenuation = np.asarray(uv_dust_attenuation(observed_grid, z, c0=c0, c1=c1, m0=m0), dtype=float)
    nodust_interp = np.power(10.0, _interp_log10_with_linear_extrapolation(observed_grid, model_x, log_model_y))

    if clip_to_bounds:
        interp_log10 = np.interp(
            intrinsic_from_obs,
            model_x,
            log_model_y,
            left=log_model_y[0],
            right=log_model_y[-1],
        )
    else:
        interp_log10 = _interp_log10_with_linear_extrapolation(intrinsic_from_obs, model_x, log_model_y)

    interpolated_phi = np.power(10.0, interp_log10)
    observed_phi_raw = interpolated_phi * jacobian
    finite = np.isfinite(observed_phi_raw) & np.isfinite(nodust_interp)
    observed_phi = observed_phi_raw.copy()
    observed_phi[finite] = np.minimum(observed_phi_raw[finite], nodust_interp[finite])

    exceeded = finite & (observed_phi_raw > nodust_interp)
    transition_index = int(np.flatnonzero(exceeded)[0]) if np.any(exceeded) else -1

    return {
        "Muv_obs": observed_grid,
        "Muv_intrinsic": intrinsic_from_obs,
        "A_uv": attenuation,
        "dMuv_dMuv_obs": jacobian,
        "phi_nodust_obs": nodust_interp,
        "phi_intrinsic_interp": interpolated_phi,
        "phi_obs": observed_phi,
        "Muv_obs_eval": observed_grid,
        "Muv_intrinsic_eval": intrinsic_from_obs,
        "A_uv_eval": attenuation,
        "dMuv_dMuv_obs_eval": jacobian,
        "phi_nodust_obs_eval": nodust_interp,
        "phi_intrinsic_interp_eval": interpolated_phi,
        "phi_obs_eval": observed_phi_raw,
        "Muv_obs_plot": observed_grid,
        "Muv_intrinsic_plot": intrinsic_from_obs,
        "A_uv_plot": attenuation,
        "dMuv_dMuv_obs_plot": jacobian,
        "phi_nodust_obs_plot": nodust_interp,
        "phi_intrinsic_interp_plot": interpolated_phi,
        "phi_obs_plot": observed_phi,
        "transition_index": np.array([transition_index], dtype=int),
    }


__all__ = [
    "compute_dust_attenuated_uvlf",
    "intrinsic_muv_from_observed",
    "intrinsic_muv_jacobian",
    "uv_continuum_slope_beta",
    "uv_dust_attenuation",
]
