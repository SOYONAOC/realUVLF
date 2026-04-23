from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .imf import IMFSpec, resolve_pop3_imf
from .schaerer import (
    get_schaerer_column_definition,
    load_schaerer_header_metadata,
    load_schaerer_table,
    parse_schaerer_model_name,
    resolve_schaerer_model_path,
)
from mah.constants import SPEED_OF_LIGHT_A_PER_S


DEFAULT_POP3_IMF_NAMES = ("Sal", "logA", "logE")
PAPER_REFERENCE = "Raiter, Schaerer & Fosbury (2010)"


def convert_lnu_to_llambda(
    wavelength_a: float | np.ndarray,
    luminosity_nu: float | np.ndarray,
) -> float | np.ndarray:
    """Convert ``L_nu`` in ``erg/s/Hz`` to ``L_lambda`` in ``erg/s/A``."""

    wavelength_array = np.asarray(wavelength_a, dtype=float)
    luminosity_array = np.asarray(luminosity_nu, dtype=float)
    if np.any(wavelength_array <= 0.0):
        raise ValueError("wavelength_a must be strictly positive")
    converted = luminosity_array * SPEED_OF_LIGHT_A_PER_S / np.square(wavelength_array)
    if np.ndim(wavelength_a) == 0 and np.ndim(luminosity_nu) == 0:
        return float(converted)
    return np.asarray(converted, dtype=float)


def convert_llambda_to_lnu(
    wavelength_a: float | np.ndarray,
    luminosity_lambda: float | np.ndarray,
) -> float | np.ndarray:
    """Convert ``L_lambda`` in ``erg/s/A`` to ``L_nu`` in ``erg/s/Hz``."""

    wavelength_array = np.asarray(wavelength_a, dtype=float)
    luminosity_array = np.asarray(luminosity_lambda, dtype=float)
    if np.any(wavelength_array <= 0.0):
        raise ValueError("wavelength_a must be strictly positive")
    converted = luminosity_array * np.square(wavelength_array) / SPEED_OF_LIGHT_A_PER_S
    if np.ndim(wavelength_a) == 0 and np.ndim(luminosity_lambda) == 0:
        return float(converted)
    return np.asarray(converted, dtype=float)


def measure_average_luminosity_lambda(
    wavelength_a: np.ndarray | list[float] | tuple[float, ...],
    luminosity_lambda: np.ndarray | list[float] | tuple[float, ...],
    *,
    center_a: float = 1500.0,
    half_width_a: float = 20.0,
) -> float:
    """Return the mean ``L_lambda`` inside a symmetric wavelength window."""

    wavelength = np.asarray(wavelength_a, dtype=float)
    luminosity = np.asarray(luminosity_lambda, dtype=float)
    if wavelength.ndim != 1 or luminosity.ndim != 1:
        raise ValueError("wavelength_a and luminosity_lambda must be 1D arrays")
    if wavelength.size != luminosity.size:
        raise ValueError("wavelength_a and luminosity_lambda must have the same length")
    if half_width_a <= 0.0:
        raise ValueError("half_width_a must be positive")

    mask = (
        np.isfinite(wavelength)
        & np.isfinite(luminosity)
        & (luminosity > 0.0)
        & (np.abs(wavelength - float(center_a)) <= float(half_width_a))
    )
    if not np.any(mask):
        raise ValueError(
            f"no valid spectrum samples found inside {center_a:g} +/- {half_width_a:g} A"
        )
    return float(np.mean(luminosity[mask]))


def fit_uv_power_law_beta(
    wavelength_a: np.ndarray | list[float] | tuple[float, ...],
    luminosity_lambda: np.ndarray | list[float] | tuple[float, ...],
    *,
    wavelength_min_a: float = 1300.0,
    wavelength_max_a: float = 1800.0,
) -> float:
    """Fit the UV slope ``beta`` defined by ``L_lambda ~ lambda**beta``."""

    wavelength = np.asarray(wavelength_a, dtype=float)
    luminosity = np.asarray(luminosity_lambda, dtype=float)
    if wavelength.ndim != 1 or luminosity.ndim != 1:
        raise ValueError("wavelength_a and luminosity_lambda must be 1D arrays")
    if wavelength.size != luminosity.size:
        raise ValueError("wavelength_a and luminosity_lambda must have the same length")
    if wavelength_min_a >= wavelength_max_a:
        raise ValueError("wavelength_min_a must be smaller than wavelength_max_a")

    mask = (
        np.isfinite(wavelength)
        & np.isfinite(luminosity)
        & (wavelength >= float(wavelength_min_a))
        & (wavelength <= float(wavelength_max_a))
        & (luminosity > 0.0)
    )
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            f"need at least two valid points inside [{wavelength_min_a:g}, {wavelength_max_a:g}] A to fit beta"
        )

    log_wavelength = np.log10(wavelength[mask])
    log_luminosity = np.log10(luminosity[mask])
    beta, _intercept = np.polyfit(log_wavelength, log_luminosity, deg=1)
    return float(beta)


def reconstruct_uv_power_law_lambda(
    wavelength_a: float | np.ndarray,
    *,
    anchor_wavelength_a: float,
    anchor_luminosity_lambda: float,
    beta: float,
) -> float | np.ndarray:
    """Reconstruct a UV continuum from one anchor luminosity and one slope ``beta``."""

    wavelength = np.asarray(wavelength_a, dtype=float)
    if np.any(wavelength <= 0.0):
        raise ValueError("wavelength_a must be strictly positive")
    if anchor_wavelength_a <= 0.0:
        raise ValueError("anchor_wavelength_a must be strictly positive")
    if anchor_luminosity_lambda <= 0.0:
        raise ValueError("anchor_luminosity_lambda must be strictly positive")

    reconstructed = float(anchor_luminosity_lambda) * np.power(
        wavelength / float(anchor_wavelength_a),
        float(beta),
    )
    if np.ndim(wavelength_a) == 0:
        return float(reconstructed)
    return np.asarray(reconstructed, dtype=float)


def _normalization_metadata_for_sfh(sfh: str, *, normalization_dependent: bool) -> tuple[str, str]:
    if not normalization_dependent:
        return "independent", ""
    sfh_token = str(sfh)
    if sfh_token.startswith("is"):
        return "per_msun_burst", "/Msun"
    if sfh_token.startswith("cs"):
        return "per_unit_sfr", "/(Msun/yr)"
    return "unknown", ""


def load_schaerer_uv_series(
    imf: str | IMFSpec,
    quantity: str = "L_15SB",
    *,
    sfh: str | None = None,
    metallicity: str | None = None,
    tracks: str | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load one UV-related quantity series from the local Schaerer/Raiter tables."""

    spec = resolve_pop3_imf(imf)
    column = get_schaerer_column_definition(quantity)
    model_path = resolve_schaerer_model_path(
        spec,
        quantity=column.quantity_name,
        sfh=sfh,
        metallicity=metallicity,
        tracks=tracks,
        data_dir=data_dir,
    )
    model_info = parse_schaerer_model_name(model_path)
    header_metadata = load_schaerer_header_metadata(model_path)
    table = load_schaerer_table(model_path)

    log_age_yr = np.asarray(table[:, 0], dtype=float)
    raw_values = np.asarray(table[:, column.column_index], dtype=float)
    values = np.power(10.0, raw_values) if column.is_log10 else raw_values.copy()
    ages_myr = np.power(10.0, log_age_yr - 6.0)

    normalization_kind, normalization_suffix = _normalization_metadata_for_sfh(
        model_info["sfh"],
        normalization_dependent=column.normalization_dependent,
    )
    unit = (
        f"{column.physical_unit}{normalization_suffix}"
        if normalization_suffix
        else column.physical_unit
    )
    raw_unit = f"log10({column.physical_unit})" if column.is_log10 else column.physical_unit

    return {
        "ages_myr": ages_myr,
        "log_age_yr": log_age_yr,
        "values": values,
        "raw_values": raw_values,
        "quantity_name": column.quantity_name,
        "physical_unit": column.physical_unit,
        "unit": unit,
        "raw_unit": raw_unit,
        "normalization_kind": normalization_kind,
        "wavelength_a": column.wavelength_a,
        "description": column.description,
        "source_file": str(model_path),
        "source_basename": model_info["basename"],
        "source_column": column.quantity_name,
        "source_column_number": column.source_column_number,
        "table_extension": column.file_extension,
        "is_log10_quantity": bool(column.is_log10),
        "imf_metadata": spec.to_metadata(),
        "model_metadata": model_info,
        "header_metadata": header_metadata,
        "paper": PAPER_REFERENCE,
    }


def build_pop3_uv_kernel(
    imf: str | IMFSpec,
    quantity: str = "L_15SB",
    *,
    output_unit: str = "erg/s/Hz/Msun",
    sfh: str | None = None,
    metallicity: str | None = None,
    tracks: str | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a burst-normalized UV kernel compatible with the existing SSP layer."""

    series = load_schaerer_uv_series(
        imf,
        quantity=quantity,
        sfh=sfh,
        metallicity=metallicity,
        tracks=tracks,
        data_dir=data_dir,
    )
    if series["normalization_kind"] != "per_msun_burst":
        raise ValueError("build_pop3_uv_kernel currently supports only instantaneous-burst ('is*') models")
    wavelength_a = series["wavelength_a"]
    if wavelength_a is None:
        raise ValueError(f"quantity '{series['quantity_name']}' does not define a wavelength and cannot form a UV kernel")
    if series["physical_unit"] != "erg/s/A":
        raise ValueError(
            f"quantity '{series['quantity_name']}' has physical unit '{series['physical_unit']}', "
            "not a UV luminosity per unit wavelength"
        )

    l_lambda_per_msun = np.asarray(series["values"], dtype=float)
    normalized_output_unit = str(output_unit).strip()
    if normalized_output_unit == "erg/s/A/Msun":
        luminosity_per_msun = l_lambda_per_msun
    elif normalized_output_unit == "erg/s/Hz/Msun":
        luminosity_per_msun = l_lambda_per_msun * (float(wavelength_a) ** 2) / SPEED_OF_LIGHT_A_PER_S
    else:
        raise ValueError("output_unit must be 'erg/s/Hz/Msun' or 'erg/s/A/Msun'")

    spec = resolve_pop3_imf(imf)
    return {
        "ages_myr": np.asarray(series["ages_myr"], dtype=float).copy(),
        "luminosity_per_msun": np.asarray(luminosity_per_msun, dtype=float).copy(),
        "output_unit": normalized_output_unit,
        "quantity_name": series["quantity_name"],
        "source_file": series["source_file"],
        "source_column": series["source_column"],
        "source_column_number": series["source_column_number"],
        "table_extension": series["table_extension"],
        "wavelength_a": float(wavelength_a),
        "bandpass_note": series["description"],
        "paper": series["paper"],
        "normalization_kind": series["normalization_kind"],
        "imf_name": spec.canonical_name,
        "imf_family": spec.family,
        "source_imf_token": spec.source_imf_token,
        "m_low_msun": float(spec.m_low_msun),
        "m_up_msun": float(spec.m_up_msun),
        "slope": None if spec.slope is None else float(spec.slope),
        "lognormal_mc_msun": None if spec.lognormal_mc_msun is None else float(spec.lognormal_mc_msun),
        "lognormal_sigma": None if spec.lognormal_sigma is None else float(spec.lognormal_sigma),
        "sfh": str(series["model_metadata"]["sfh"]),
        "metallicity": str(series["model_metadata"]["metallicity"]),
        "tracks": str(series["model_metadata"]["tracks"]),
    }


def build_pop3_uv_kernel_dict(
    imf_names: Iterable[str | IMFSpec] = DEFAULT_POP3_IMF_NAMES,
    *,
    quantity: str = "L_15SB",
    output_unit: str = "erg/s/Hz/Msun",
    sfh: str | None = None,
    metallicity: str | None = None,
    tracks: str | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a mapping from IMF names to UV-kernel dicts usable by the current pipeline."""

    kernels: dict[str, dict[str, Any]] = {}
    for item in imf_names:
        spec = resolve_pop3_imf(item)
        if spec.canonical_name in kernels:
            raise ValueError(f"duplicate IMF entry '{spec.canonical_name}'")
        kernels[spec.canonical_name] = build_pop3_uv_kernel(
            spec,
            quantity=quantity,
            output_unit=output_unit,
            sfh=sfh,
            metallicity=metallicity,
            tracks=tracks,
            data_dir=data_dir,
        )
    return kernels
