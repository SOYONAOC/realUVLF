"""Utilities for reading and interpolating SSP spectra."""

from .convolution import (
    SSP_UV_LOOKBACK_MAX_MYR,
    compute_halo_uv_luminosity,
    evaluate_uv_luminosity_kernel,
    interpolate_ssp_luminosity,
)
from .uv1600 import interpolate_uv1600_luminosity_per_msun, load_uv1600_table

__all__ = [
    "SSP_UV_LOOKBACK_MAX_MYR",
    "compute_halo_uv_luminosity",
    "evaluate_uv_luminosity_kernel",
    "interpolate_ssp_luminosity",
    "interpolate_uv1600_luminosity_per_msun",
    "load_uv1600_table",
]
