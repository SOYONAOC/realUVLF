from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import numpy as np

from .imf import IMFSpec, resolve_pop3_imf


SCHAERER_POP3_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "schaerer2010_pop3"


@dataclass(frozen=True)
class SchaererColumnDefinition:
    """Column metadata for the Raiter/Schaerer UV summary tables."""

    quantity_name: str
    file_extension: str
    column_index: int
    physical_unit: str
    is_log10: bool
    wavelength_a: float | None = None
    normalization_dependent: bool = False
    description: str = ""

    @property
    def source_column_number(self) -> int:
        return self.column_index + 1


_SCHAERER_COLUMNS: dict[str, SchaererColumnDefinition] = {
    "L_15SB": SchaererColumnDefinition(
        quantity_name="L_15SB",
        file_extension="24",
        column_index=2,
        physical_unit="erg/s/A",
        is_log10=True,
        wavelength_a=1500.0,
        normalization_dependent=True,
        description="Average UV luminosity at 1500 Ang over a +-20 Ang band.",
    ),
    "L_1500": SchaererColumnDefinition(
        quantity_name="L_1500",
        file_extension="24",
        column_index=3,
        physical_unit="erg/s/A",
        is_log10=True,
        wavelength_a=1500.0,
        normalization_dependent=True,
        description="Average UV luminosity at 1500 Ang over a +-150 Ang band.",
    ),
    "L_2800": SchaererColumnDefinition(
        quantity_name="L_2800",
        file_extension="24",
        column_index=4,
        physical_unit="erg/s/A",
        is_log10=True,
        wavelength_a=2800.0,
        normalization_dependent=True,
        description="Average UV luminosity at 2800 Ang over a +-280 Ang band.",
    ),
    "beta_1550": SchaererColumnDefinition(
        quantity_name="beta_1550",
        file_extension="24",
        column_index=5,
        physical_unit="dimensionless",
        is_log10=False,
        description="UV slope over 1300-1800 Ang including stellar and nebular continuum.",
    ),
    "L_1460": SchaererColumnDefinition(
        quantity_name="L_1460",
        file_extension="24",
        column_index=13,
        physical_unit="erg/s/A",
        is_log10=True,
        wavelength_a=1460.0,
        normalization_dependent=True,
        description="Average UV luminosity at 1460 Ang based on the local bandpass used in the model file.",
    ),
    "beta_1460": SchaererColumnDefinition(
        quantity_name="beta_1460",
        file_extension="24",
        column_index=14,
        physical_unit="dimensionless",
        is_log10=False,
        description="UV slope between the 1195 Ang and 1738 Ang windows defined in the model table.",
    ),
    "Q_H_over_L1500": SchaererColumnDefinition(
        quantity_name="Q_H_over_L1500",
        file_extension="25",
        column_index=2,
        physical_unit="A/erg",
        is_log10=True,
        description="Ionising photon flux divided by the 1500 Ang luminosity.",
    ),
}

_SCHAERER_COLUMN_ALIASES = {
    "l15sb": "L_15SB",
    "l_15sb": "L_15SB",
    "l1500": "L_1500",
    "l_1500": "L_1500",
    "luv1500": "L_1500",
    "l2800": "L_2800",
    "l_2800": "L_2800",
    "beta1550": "beta_1550",
    "beta_1550": "beta_1550",
    "l1460": "L_1460",
    "l_1460": "L_1460",
    "beta1460": "beta_1460",
    "beta_1460": "beta_1460",
    "q0_over_l1500": "Q_H_over_L1500",
    "qh_over_l1500": "Q_H_over_L1500",
    "q_h_over_l1500": "Q_H_over_L1500",
}

_INPUT_PARAMETERS_RE = re.compile(
    r"M_low=\s*(?P<m_low>[0-9.]+),\s*"
    r"M_up=\s*(?P<m_up>[0-9.]+),\s*"
    r"IMF type=\s*(?P<imf_type>\d+),\s*"
    r"exponent=\s*(?P<exponent>[-+0-9.]+),\s*"
    r"\(Mc,sigma\)=\(\s*(?P<mc>[0-9.]+)\s+(?P<sigma>[0-9.]+)\),\s*"
    r"total mass=\s*(?P<total_mass>[-+0-9.Ee]+),\s*"
    r"WR mass limit=\s*(?P<wr_mass_limit>[0-9.]+)"
)


def _normalize_quantity_name(quantity: str) -> str:
    requested = str(quantity).strip()
    if not requested:
        allowed = ", ".join(sorted(_SCHAERER_COLUMNS))
        raise ValueError(f"quantity must be one of: {allowed}")
    if requested in _SCHAERER_COLUMNS:
        return requested
    alias_match = _SCHAERER_COLUMN_ALIASES.get(requested.casefold())
    if alias_match is not None:
        return alias_match
    allowed = ", ".join(sorted(_SCHAERER_COLUMNS))
    raise ValueError(f"unknown Schaerer quantity '{requested}'; allowed values are: {allowed}")


def get_schaerer_column_definition(quantity: str) -> SchaererColumnDefinition:
    return _SCHAERER_COLUMNS[_normalize_quantity_name(quantity)]


def parse_schaerer_model_name(model_name: str | Path) -> dict[str, Any]:
    """Parse a model basename such as ``pop3_ge0_sal_500_001_is5.24``."""

    path = Path(model_name)
    file_name = path.name
    extension = path.suffix[1:] if path.suffix.startswith(".") else ""
    stem = path.stem if extension else file_name
    parts = stem.split("_")
    if len(parts) != 6:
        raise ValueError(
            "Schaerer model names must have the form "
            "'metallicity_tracks_imf_mup_mlow_sfh[.ext]'"
        )

    metallicity, tracks, imf_token, m_up_token, m_low_token, sfh = parts
    return {
        "file_name": file_name,
        "basename": stem,
        "extension": extension,
        "metallicity": metallicity,
        "tracks": tracks,
        "imf_token": imf_token,
        "m_up_msun": float(m_up_token),
        "m_low_msun": float(m_low_token),
        "sfh": sfh,
    }


def resolve_schaerer_model_path(
    imf: str | IMFSpec,
    *,
    quantity: str,
    sfh: str | None = None,
    metallicity: str | None = None,
    tracks: str | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    """Resolve the local file path for a given literature IMF and quantity."""

    spec = resolve_pop3_imf(imf)
    column = get_schaerer_column_definition(quantity)
    base_dir = SCHAERER_POP3_DATA_DIR if data_dir is None else Path(data_dir).expanduser().resolve()
    model_basename = spec.build_schaerer_model_basename(sfh=sfh, metallicity=metallicity, tracks=tracks)
    model_path = base_dir / f"{model_basename}.{column.file_extension}"
    if not model_path.exists():
        raise FileNotFoundError(f"Schaerer model file not found: {model_path}")
    return model_path


@lru_cache(maxsize=None)
def _load_schaerer_table_cached(model_path: str) -> np.ndarray:
    return np.loadtxt(model_path, dtype=float)


def load_schaerer_table(model_path: str | Path) -> np.ndarray:
    resolved = str(Path(model_path).expanduser().resolve())
    return _load_schaerer_table_cached(resolved).copy()


@lru_cache(maxsize=None)
def _read_schaerer_header_cached(model_path: str) -> tuple[str, ...]:
    lines: list[str] = []
    with open(model_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            lines.append(line.rstrip("\n"))
    return tuple(lines)


def load_schaerer_header_metadata(model_path: str | Path) -> dict[str, Any]:
    """Parse a small subset of the model header into structured metadata."""

    resolved = str(Path(model_path).expanduser().resolve())
    lines = _read_schaerer_header_cached(resolved)
    metadata: dict[str, Any] = {}
    for line in lines:
        if "Author:" in line:
            metadata["author"] = line.split("Author:", 1)[1].strip()
        elif "Model run:" in line:
            metadata["model_run"] = line.split("Model run:", 1)[1].strip()
        elif "Star-formation:" in line:
            metadata["star_formation"] = line.split("Star-formation:", 1)[1].strip()
        elif "Input parameters:" in line:
            content = line.split("Input parameters:", 1)[1].strip()
            metadata["input_parameters_line"] = content
            match = _INPUT_PARAMETERS_RE.search(content)
            if match is not None:
                metadata.update(
                    {
                        "m_low_msun": float(match.group("m_low")),
                        "m_up_msun": float(match.group("m_up")),
                        "imf_type": int(match.group("imf_type")),
                        "exponent": float(match.group("exponent")),
                        "lognormal_mc_msun": float(match.group("mc")),
                        "lognormal_sigma": float(match.group("sigma")),
                        "total_mass_msun": float(match.group("total_mass")),
                        "wr_mass_limit_msun": float(match.group("wr_mass_limit")),
                    }
                )
    metadata["header_lines"] = list(lines)
    return metadata
