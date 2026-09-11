"""Conditional Case-B post-processing of saved random-q bursts.

This experiment uses the matching Raiter et al. (2010) instantaneous-burst
tables, independently checked against their equation (3). It does not import
the archived SFH model or supply a gas/photoionization model.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from auroralf.ssp import interpolate_ssp_luminosity, load_popiii_uv_luminosity_table

from .artifacts import require


@dataclass(frozen=True)
class HeIIKernel:
    age_myr: np.ndarray
    line_per_msun: np.ndarray
    q2_per_msun: np.ndarray
    pure_popiii_ew_angstrom: np.ndarray
    uv_per_msun: np.ndarray
    caseb_max_relative_error: float


def load_kernel(uv_path: Path, line_path: Path) -> HeIIKernel:
    """Require a matching, unit-mass Pop III logE .25/.22 model set."""
    require(uv_path.stem == line_path.stem, "UV and line SSP model sets differ")
    require(uv_path.name == "pop3_ge0_logE_500_001_is5.25", "unvalidated SSP model")
    require(line_path.suffix == ".22", "expected recombination-line table .22")
    headers = [p.read_text().splitlines()[:16] for p in (uv_path, line_path)]
    for token in (
        "total mass= 1.0E+00",
        "instantaneous burst at age=0",
        "Z=0.",
        "No binaries included",
        "Fraction N_Lyc photons used= 1.000",
    ):
        require(all(any(token in row for row in h) for h in headers), f"SSP header: {token}")
    require(any("HeII_1640" in row for row in headers[1]), "missing He II columns")
    uv, lines = (np.loadtxt(p) for p in (uv_path, line_path))
    require(lines.ndim == 2 and lines.shape[1] == 30, "unexpected line table shape")
    require(np.isfinite(lines).all(), "nonfinite SSP table")
    require(np.array_equal(uv[:, 0], lines[:, 0]), "UV and line printed age grids differ")
    # Use exactly the UV model's documented is5 reconstruction: 0.01,1,...,1000
    # Myr. Rounded log ages repeat at late times; do not sort/drop these rows.
    age, luv = load_popiii_uv_luminosity_table(str(uv_path))
    require(len(age) == len(lines), "SSP age dimensions")
    q2 = np.where(lines[:, 3] <= -99, 0.0, 10.0 ** lines[:, 3])
    line = lines[:, 4] * lines[:, 14]  # L(Hbeta) times I(1640)/I(Hbeta)
    require(np.all(line >= 0) and np.all(lines[:, 13] >= 0), "negative line/EW")
    mask = q2 > 1e35
    error = float(np.max(np.abs(line[mask] / (5.67e-12 * q2[mask]) - 1)))
    require(error < 0.01, "line table disagrees with Raiter+2010 Case-B conversion")
    return HeIIKernel(age, line, q2, lines[:, 13], luv, error)


def evaluate_kernel(age_myr, kernel: HeIIKernel, method="linear_log_age"):
    """Baseline matches UV interpolation; alternatives diagnose table resolution.

    Below 0.01 Myr the first tabulated value is held constant explicitly.
    Out-of-domain older ages raise instead of inventing a zero tail.
    """
    age = np.asarray(age_myr, float)
    require(np.isfinite(age).all() and np.all(age >= 0), "invalid burst ages")
    require(np.all(age <= kernel.age_myr[-1]), "burst older than He II SSP table")
    if method == "linear_log_age":
        return interpolate_ssp_luminosity(age, kernel.age_myr, kernel.line_per_msun)
    if method == "linear_age":
        return np.interp(age, kernel.age_myr, kernel.line_per_msun)
    if method == "log_log_age":
        require(np.all(kernel.line_per_msun > 0), "log interpolation needs positive kernel")
        return 10.0 ** np.interp(
            np.log10(np.maximum(age, kernel.age_myr[0])),
            np.log10(kernel.age_myr),
            np.log10(kernel.line_per_msun),
        )
    raise ValueError(f"unknown interpolation: {method}")


def cluster_sum_and_se(per_mass):
    """Mean of independent run estimates; preserve each run's global HMF weights."""
    estimates = [np.asarray(x, float) for x in per_mass]
    require(bool(estimates) and all(x.shape[0] >= 2 for x in estimates), "missing mass clusters")
    mean = np.mean([x.sum(axis=0) for x in estimates], axis=0)
    se = np.sqrt(np.sum([len(x) * np.var(x, axis=0, ddof=1) for x in estimates], axis=0)) / len(
        estimates
    )
    return mean, se
