from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy import units as u


DEFAULT_WAVELENGTH_A = 1600.0
MODEL_NORMALIZATION_MSUN = 1.0e6


def _ssp_ages_myr(n_bins: int) -> np.ndarray:
    indices = np.arange(n_bins, dtype=float)
    log_age_yr = 6.0 + 0.1 * indices
    return 10.0 ** (log_age_yr - 6.0)


@lru_cache(maxsize=None)
def _load_uv1600_table_cached(file_path: str, wavelength_a: float) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(file_path)
    wavelength_grid = data[:, 0]
    idx = int(np.argmin(np.abs(wavelength_grid - wavelength_a)))
    l_lambda = data[idx, 1:]
    ages_myr = _ssp_ages_myr(l_lambda.size)

    lum_nu = (l_lambda * (u.L_sun / u.AA)).to(
        u.erg / u.s / u.Hz,
        equivalencies=u.spectral_density(wavelength_grid[idx] * u.AA),
    )
    luminosity_per_msun = lum_nu.value / MODEL_NORMALIZATION_MSUN
    return ages_myr, luminosity_per_msun


def load_uv1600_table(
    file_path: str | Path,
    wavelength_a: float = DEFAULT_WAVELENGTH_A,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the 1600 A SSP luminosity table once and return age and luminosity-per-Msun arrays."""

    resolved = str(Path(file_path).expanduser().resolve())
    ages_myr, luminosity_per_msun = _load_uv1600_table_cached(resolved, float(wavelength_a))
    return ages_myr.copy(), luminosity_per_msun.copy()


def interpolate_uv1600_luminosity_per_msun(
    time_myr: float | np.ndarray,
    file_path: str | Path,
    wavelength_a: float = DEFAULT_WAVELENGTH_A,
) -> float | np.ndarray:
    """Interpolate the 1600 A luminosity per solar mass at the requested SSP age in Myr."""

    ages_myr, luminosity_per_msun = load_uv1600_table(file_path=file_path, wavelength_a=wavelength_a)
    time_myr_array = np.asarray(time_myr, dtype=float)
    log_ages = np.log10(ages_myr)
    log_time = np.log10(np.clip(time_myr_array, ages_myr[0], ages_myr[-1]))
    interpolated = np.interp(log_time, log_ages, luminosity_per_msun)
    if np.ndim(time_myr) == 0:
        return float(interpolated)
    return interpolated
