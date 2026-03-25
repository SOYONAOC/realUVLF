from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy import units as u

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - exercised only when h5py is absent
    h5py = None


DEFAULT_WAVELENGTH_A = 1600.0
MODEL_NORMALIZATION_MSUN = 1.0e6


def _ssp_ages_myr(n_bins: int) -> np.ndarray:
    indices = np.arange(n_bins, dtype=float)
    log_age_yr = 6.0 + 0.1 * indices
    return 10.0 ** (log_age_yr - 6.0)


def _resolve_hdf5_metallicity_index(metallicity_zsun: float, metallicities_dex: np.ndarray) -> int:
    metallicities_zsun = np.power(10.0, metallicities_dex)
    matched = np.isclose(metallicities_zsun, float(metallicity_zsun), rtol=0.0, atol=1.0e-10)
    if np.any(matched):
        return int(np.flatnonzero(matched)[0])

    formatted = ", ".join(
        f"{zsun:g} Zsun (dex={dex:.6f})" for zsun, dex in zip(metallicities_zsun, metallicities_dex, strict=True)
    )
    raise ValueError(
        "metallicity must exactly match one of the discrete HDF5 options in Z/Zsun; "
        f"requested {metallicity_zsun:g}, available: {formatted}"
    )


def _load_uv1600_table_from_dat(file_path: str, wavelength_a: float) -> tuple[np.ndarray, np.ndarray]:
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


def _load_uv1600_table_from_npz(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(file_path)
    if "ages_myr" not in payload or "luminosity_per_msun" not in payload:
        raise ValueError("NPZ SSP files must contain 'ages_myr' and 'luminosity_per_msun' arrays")

    ages_myr = np.asarray(payload["ages_myr"], dtype=float)
    luminosity_per_msun = np.asarray(payload["luminosity_per_msun"], dtype=float)
    if ages_myr.ndim != 1 or luminosity_per_msun.ndim != 1:
        raise ValueError("NPZ SSP arrays must be 1D")
    if ages_myr.size != luminosity_per_msun.size:
        raise ValueError("ages_myr and luminosity_per_msun must have the same length")
    return ages_myr, luminosity_per_msun


def _load_uv1600_table_from_hdf5(
    file_path: str,
    wavelength_a: float,
    metallicity: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if h5py is None:
        raise ModuleNotFoundError("h5py is required to read HDF5 SSP files")
    if metallicity is None:
        raise ValueError("metallicity must be provided in Z/Zsun when loading an HDF5 SSP file")

    with h5py.File(file_path, "r") as handle:
        wavelength_grid = np.asarray(handle["/wavelengths"], dtype=float)
        wavelength_index = int(np.argmin(np.abs(wavelength_grid - wavelength_a)))
        metallicities_dex = np.asarray(handle["/metallicities"], dtype=float)
        metallicity_index = _resolve_hdf5_metallicity_index(metallicity, metallicities_dex)
        ages_myr = np.asarray(handle["/ages"], dtype=float) * 1.0e3
        l_nu = np.asarray(handle["/spectra"][metallicity_index, :, wavelength_index], dtype=float)

    lum_nu = (l_nu * (u.L_sun / u.Hz)).to(u.erg / u.s / u.Hz)
    # These HDF5 templates are already normalized per unit stellar mass.
    luminosity_per_msun = lum_nu.value
    return ages_myr, luminosity_per_msun


@lru_cache(maxsize=None)
def _load_uv1600_table_cached(
    file_path: str,
    wavelength_a: float,
    metallicity: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    suffix = Path(file_path).suffix.lower()
    if suffix in {".hdf5", ".h5"}:
        return _load_uv1600_table_from_hdf5(file_path=file_path, wavelength_a=wavelength_a, metallicity=metallicity)
    if suffix == ".npz":
        if metallicity is not None:
            raise ValueError("metallicity is only supported for HDF5 SSP files")
        return _load_uv1600_table_from_npz(file_path=file_path)
    if metallicity is not None:
        raise ValueError("metallicity is only supported for HDF5 SSP files")
    return _load_uv1600_table_from_dat(file_path=file_path, wavelength_a=wavelength_a)


def load_uv1600_table(
    file_path: str | Path,
    wavelength_a: float = DEFAULT_WAVELENGTH_A,
    metallicity: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a 1600 A SSP luminosity table and return age and luminosity-per-Msun arrays.

    For HDF5 SSP files, ``metallicity`` must be supplied in linear ``Z/Zsun`` and must
    exactly match one of the discrete metallicity bins stored in the file.
    """

    resolved = str(Path(file_path).expanduser().resolve())
    ages_myr, luminosity_per_msun = _load_uv1600_table_cached(
        resolved,
        float(wavelength_a),
        None if metallicity is None else float(metallicity),
    )
    return ages_myr.copy(), luminosity_per_msun.copy()


def interpolate_uv1600_luminosity_per_msun(
    time_myr: float | np.ndarray,
    file_path: str | Path,
    wavelength_a: float = DEFAULT_WAVELENGTH_A,
    metallicity: float | None = None,
) -> float | np.ndarray:
    """Interpolate the 1600 A luminosity per solar mass at the requested SSP age in Myr."""

    ages_myr, luminosity_per_msun = load_uv1600_table(
        file_path=file_path,
        wavelength_a=wavelength_a,
        metallicity=metallicity,
    )
    time_myr_array = np.asarray(time_myr, dtype=float)
    log_ages = np.log10(ages_myr)
    log_time = np.log10(np.clip(time_myr_array, ages_myr[0], ages_myr[-1]))
    interpolated = np.interp(log_time, log_ages, luminosity_per_msun)
    if np.ndim(time_myr) == 0:
        return float(interpolated)
    return interpolated
