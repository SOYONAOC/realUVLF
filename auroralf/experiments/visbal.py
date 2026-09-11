"""Configuration and numerical helpers for the opt-in VH15 duty comparison."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from auroralf.constants import AB_ZEROPOINT_LNU

from .artifacts import read_toml, require, resolve_path


@dataclass(frozen=True)
class VisbalDutyConfig:
    experiment: str
    run: str
    source_root: Path
    z: float
    fstar: float
    duties: tuple[float, ...]
    csfr_age_myr: float
    full_run: str
    low_run: str
    output: Path
    samples_root: Path
    baseline: Path
    popiii_ssp: Path
    observations: Path

    @classmethod
    def load(cls, path: str | Path) -> "VisbalDutyConfig":
        path, data = read_toml(path)
        for key in (
            "source_root",
            "output",
            "samples_root",
            "baseline",
            "popiii_ssp",
            "observations",
        ):
            data[key] = resolve_path(path.parent, data[key])
        data["duties"] = tuple(data["duties"])
        config = cls(**data)
        require(np.isfinite(config.z) and config.z >= 0, "invalid redshift")
        require(np.isfinite(config.fstar) and 0 < config.fstar <= 1, "invalid fstar")
        require(np.isfinite(config.csfr_age_myr) and config.csfr_age_myr > 0, "invalid CSFR age")
        require(
            bool(config.duties) and len(set(config.duties)) == len(config.duties),
            "empty or duplicate duties",
        )
        require(all(np.isfinite(d) and 0 < d <= 1 for d in config.duties), "invalid duties")
        for name in (config.full_run, config.low_run):
            require(Path(name).name == name and name not in (".", ".."), "invalid run name")
        return config

    def as_metadata(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


@dataclass(frozen=True)
class VisbalCombineConfig:
    prior: Path
    full_run: Path
    runs: tuple[Path, ...]
    source_root: Path
    output: Path
    mass_min_ratio: float
    mass_max_ratio: float
    n_mass: int
    n_tracks: int
    n_grid: int

    @classmethod
    def load(cls, path: str | Path) -> "VisbalCombineConfig":
        path, data = read_toml(path)
        for key in ("prior", "full_run", "source_root", "output"):
            data[key] = resolve_path(path.parent, data[key])
        data["runs"] = tuple(resolve_path(path.parent, p) for p in data["runs"])
        config = cls(**data)
        require(
            len(config.runs) >= 2 and len(set(config.runs)) == len(config.runs),
            "need at least two distinct batch paths for batch SE",
        )
        require(
            np.isfinite(config.mass_min_ratio)
            and np.isfinite(config.mass_max_ratio)
            and 0 < config.mass_min_ratio < config.mass_max_ratio,
            "invalid mass window",
        )
        for key in ("n_mass", "n_tracks", "n_grid"):
            require(
                type(getattr(config, key)) is int and getattr(config, key) >= 2, f"invalid {key}"
            )
        return config


def active_weights(weight, eligible, duty):
    weight, eligible = np.asarray(weight, dtype=float), np.asarray(eligible)
    require(
        weight.shape == eligible.shape and eligible.dtype == np.bool_,
        "inconsistent occupancy arrays",
    )
    require(np.all(np.isfinite(weight)) and np.all(weight >= 0), "invalid halo weights")
    require(np.isfinite(duty) and 0 < duty <= 1, "duty must be in (0,1]")
    active = weight * eligible * duty
    inactive = weight - active
    np.testing.assert_allclose(active + inactive, weight)
    return active, inactive


def uv_coefficient(ages, luminosity, duration):
    """Integrate a linear-in-log-age SSP; hold its first value to age zero.

    Ages and duration are in Myr; the result converts SFR in Msun/yr to UV light.
    """
    ages, luminosity = np.asarray(ages, dtype=float), np.asarray(luminosity, dtype=float)
    require(
        ages.ndim == 1 and ages.size >= 2 and luminosity.shape == ages.shape, "invalid SSP shape"
    )
    require(
        np.all(np.isfinite(ages)) and np.all(ages > 0) and np.all(np.diff(ages) > 0),
        "invalid SSP ages",
    )
    require(np.all(np.isfinite(luminosity)) and np.all(luminosity >= 0), "invalid SSP luminosities")
    require(
        np.isfinite(duration) and ages[0] <= duration <= ages[-1], "duration outside SSP coverage"
    )
    x = np.r_[ages[ages < duration], duration]
    y = np.interp(np.log(x), np.log(ages), luminosity)
    slope = np.diff(y) / np.diff(np.log(x))
    integral = y[0] * x[0] + np.sum(
        y[:-1] * np.diff(x) + slope * (x[1:] * np.log(x[1:] / x[:-1]) - np.diff(x))
    )
    return float(integral * 1e6)


def mag(lum):
    with np.errstate(divide="ignore"):
        return AB_ZEROPOINT_LNU - 2.5 * np.log10(lum)


def combine_window(high, batches):
    batches = np.asarray(batches)
    require(batches.ndim == 2 and len(batches) >= 2, "need independent batches")
    require(np.asarray(high).shape == batches.shape[1:], "inconsistent histogram shapes")
    require(np.all(np.isfinite(batches)) and np.all(np.isfinite(high)), "nonfinite histogram")
    return high + batches.mean(axis=0), batches.std(axis=0, ddof=1) / np.sqrt(len(batches))
