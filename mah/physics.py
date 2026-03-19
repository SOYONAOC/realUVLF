from __future__ import annotations

import numpy as np

from .models import Cosmology


def mass_history(
    redshift: float | np.ndarray,
    redshift_final: float,
    mass_final: float,
    beta: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    redshift = np.asarray(redshift, dtype=float)
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    if redshift.ndim == 0:
        ratio = (1.0 + redshift) / (1.0 + redshift_final)
        return mass_final * ratio**beta * np.exp(-gamma * (redshift - redshift_final))

    ratio = (1.0 + redshift[None, :]) / (1.0 + redshift_final)
    delta_z = redshift[None, :] - redshift_final
    return mass_final * ratio**beta[:, None] * np.exp(-gamma[:, None] * delta_z)


def accretion_rate(
    redshift: np.ndarray,
    redshift_final: float,
    mass_final: float,
    beta: np.ndarray,
    gamma: np.ndarray,
    cosmology: Cosmology,
) -> tuple[np.ndarray, np.ndarray]:
    mass = mass_history(redshift, redshift_final, mass_final, beta, gamma)
    one_plus_z = 1.0 + redshift[None, :]
    mdot = -cosmology.hubble(redshift)[None, :] * one_plus_z * mass * (beta[:, None] / one_plus_z - gamma[:, None])
    return mass, mdot
