from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IMFSpec:
    """Minimal IMF description for the Pop III literature grids used here."""

    canonical_name: str
    family: str
    source_imf_token: str
    m_low_msun: float
    m_up_msun: float
    slope: float | None = None
    lognormal_mc_msun: float | None = None
    lognormal_sigma: float | None = None
    metallicity: str = "pop3"
    tracks: str = "ge0"
    default_sfh: str = "is5"

    def __post_init__(self) -> None:
        if not self.canonical_name:
            raise ValueError("canonical_name must not be empty")
        if not self.family:
            raise ValueError("family must not be empty")
        if not self.source_imf_token:
            raise ValueError("source_imf_token must not be empty")
        if self.m_low_msun <= 0.0 or self.m_up_msun <= 0.0:
            raise ValueError("m_low_msun and m_up_msun must be positive")
        if self.m_up_msun < self.m_low_msun:
            raise ValueError("m_up_msun must be greater than or equal to m_low_msun")

    def build_schaerer_model_basename(
        self,
        *,
        sfh: str | None = None,
        metallicity: str | None = None,
        tracks: str | None = None,
    ) -> str:
        """Return the Raiter/Schaerer basename without the file extension."""

        sfh_token = self.default_sfh if sfh is None else str(sfh)
        metallicity_token = self.metallicity if metallicity is None else str(metallicity)
        tracks_token = self.tracks if tracks is None else str(tracks)
        m_up_token = int(round(float(self.m_up_msun)))
        m_low_token = int(round(float(self.m_low_msun)))
        return (
            f"{metallicity_token}_{tracks_token}_{self.source_imf_token}_"
            f"{m_up_token:03d}_{m_low_token:03d}_{sfh_token}"
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "family": self.family,
            "source_imf_token": self.source_imf_token,
            "m_low_msun": float(self.m_low_msun),
            "m_up_msun": float(self.m_up_msun),
            "slope": None if self.slope is None else float(self.slope),
            "lognormal_mc_msun": None if self.lognormal_mc_msun is None else float(self.lognormal_mc_msun),
            "lognormal_sigma": None if self.lognormal_sigma is None else float(self.lognormal_sigma),
            "metallicity": self.metallicity,
            "tracks": self.tracks,
            "default_sfh": self.default_sfh,
        }


_POP3_IMF_LIBRARY: dict[str, IMFSpec] = {
    "Sal": IMFSpec(
        canonical_name="Sal",
        family="salpeter",
        source_imf_token="sal",
        m_low_msun=1.0,
        m_up_msun=500.0,
        slope=2.35,
    ),
    "logA": IMFSpec(
        canonical_name="logA",
        family="lognormal",
        source_imf_token="logA",
        m_low_msun=1.0,
        m_up_msun=500.0,
        lognormal_mc_msun=10.0,
        lognormal_sigma=1.0,
    ),
    "logB": IMFSpec(
        canonical_name="logB",
        family="lognormal",
        source_imf_token="logB",
        m_low_msun=1.0,
        m_up_msun=500.0,
        lognormal_mc_msun=15.0,
        lognormal_sigma=0.3,
    ),
    "logE": IMFSpec(
        canonical_name="logE",
        family="lognormal",
        source_imf_token="logE",
        m_low_msun=1.0,
        m_up_msun=500.0,
        lognormal_mc_msun=60.0,
        lognormal_sigma=1.0,
    ),
    "l05": IMFSpec(
        canonical_name="l05",
        family="larson",
        source_imf_token="l05",
        m_low_msun=1.0,
        m_up_msun=100.0,
        slope=2.35,
    ),
    "sca": IMFSpec(
        canonical_name="sca",
        family="scalo",
        source_imf_token="sca",
        m_low_msun=1.0,
        m_up_msun=100.0,
        slope=2.70,
    ),
}

_POP3_IMF_ALIASES = {
    "sal": "Sal",
    "salpeter": "Sal",
    "sal500": "Sal",
    "sal_500": "Sal",
    "loga": "logA",
    "loga500": "logA",
    "loga_500": "logA",
    "logb": "logB",
    "logb500": "logB",
    "logb_500": "logB",
    "loge": "logE",
    "loge500": "logE",
    "loge_500": "logE",
    "larson": "l05",
    "larson5": "l05",
    "scalo": "sca",
    "scalo86": "sca",
}


def list_available_pop3_imfs() -> tuple[str, ...]:
    return tuple(_POP3_IMF_LIBRARY)


def resolve_pop3_imf(imf: str | IMFSpec) -> IMFSpec:
    """Return the canonical Pop III IMF spec used by the local literature grid."""

    if isinstance(imf, IMFSpec):
        return imf

    requested = str(imf).strip()
    if not requested:
        allowed = ", ".join(list_available_pop3_imfs())
        raise ValueError(f"imf must be one of: {allowed}")

    canonical = _POP3_IMF_LIBRARY.get(requested)
    if canonical is not None:
        return canonical

    alias_match = _POP3_IMF_ALIASES.get(requested.casefold())
    if alias_match is not None:
        return _POP3_IMF_LIBRARY[alias_match]

    allowed = ", ".join(list_available_pop3_imfs())
    raise ValueError(f"unknown Pop III IMF '{requested}'; allowed values are: {allowed}")
