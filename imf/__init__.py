"""Helpers for mapping literature IMF choices onto UV kernel tables."""

from .models import IMFSpec, list_available_pop3_imfs, resolve_pop3_imf
from .schaerer import (
    SCHAERER_POP3_DATA_DIR,
    SchaererColumnDefinition,
    get_schaerer_column_definition,
    load_schaerer_header_metadata,
    load_schaerer_table,
    parse_schaerer_model_name,
    resolve_schaerer_model_path,
)
from .uv import build_pop3_uv_kernel, build_pop3_uv_kernel_dict, load_schaerer_uv_series
from .uv import (
    convert_llambda_to_lnu,
    convert_lnu_to_llambda,
    fit_uv_power_law_beta,
    measure_average_luminosity_lambda,
    reconstruct_uv_power_law_lambda,
)

__all__ = [
    "IMFSpec",
    "SCHAERER_POP3_DATA_DIR",
    "SchaererColumnDefinition",
    "build_pop3_uv_kernel",
    "build_pop3_uv_kernel_dict",
    "convert_llambda_to_lnu",
    "convert_lnu_to_llambda",
    "fit_uv_power_law_beta",
    "get_schaerer_column_definition",
    "list_available_pop3_imfs",
    "load_schaerer_header_metadata",
    "load_schaerer_table",
    "load_schaerer_uv_series",
    "measure_average_luminosity_lambda",
    "parse_schaerer_model_name",
    "reconstruct_uv_power_law_lambda",
    "resolve_pop3_imf",
    "resolve_schaerer_model_path",
]
