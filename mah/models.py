from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SECONDS_PER_GYR = 3.15576e16
KM_PER_MPC = 3.085677581491367e19
GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN = 4.30091e-9
POWER_LAW_FRACTION = 0.0466

# Match the project defaults used elsewhere in the repository.
PLANCK18_H = 0.674
PLANCK18_H0_KM_S_MPC = 100.0 * PLANCK18_H
PLANCK18_OMEGA_M = 0.315
PLANCK18_OMEGA_B = 0.04897
PLANCK18_OMEGA_LAMBDA = 1.0 - PLANCK18_OMEGA_M
PLANCK18_H0_GYR = PLANCK18_H0_KM_S_MPC * SECONDS_PER_GYR / KM_PER_MPC


@dataclass(frozen=True)
class Cosmology:
    h0: float = PLANCK18_H0_GYR
    omega_m: float = PLANCK18_OMEGA_M
    omega_b: float = PLANCK18_OMEGA_B
    omega_lambda: float = PLANCK18_OMEGA_LAMBDA

    def hubble(self, redshift: float | np.ndarray) -> float | np.ndarray:
        redshift = np.asarray(redshift, dtype=float)
        return self.h0 * np.sqrt(self.omega_m * (1.0 + redshift) ** 3 + self.omega_lambda)

    @property
    def h0_km_s_mpc(self) -> float:
        return self.h0 * KM_PER_MPC / SECONDS_PER_GYR

    @property
    def omegam(self) -> float:
        return self.omega_m

    @property
    def omegab(self) -> float:
        return self.omega_b

    @property
    def omegalam(self) -> float:
        return self.omega_lambda

    @property
    def H0u(self) -> float:
        return self.h0_km_s_mpc

    @property
    def rhocrit(self) -> float:
        return 3.0 * self.H0u**2 / (8.0 * np.pi * GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN)


class CosmologySet(Cosmology):
    """Compatibility wrapper for code paths expecting the legacy cosmology API."""


@dataclass(frozen=True)
class GaussianApproximation:
    mean: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class HaloHistoryResult:
    tracks: dict[str, np.ndarray]
    metadata: dict[str, Any]
