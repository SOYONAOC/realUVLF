from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from constants import (
    GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN,
    KM_PER_MPC,
    PLANCK18_H0_GYR,
    PLANCK18_OMEGA_B,
    PLANCK18_OMEGA_LAMBDA,
    PLANCK18_OMEGA_M,
    SECONDS_PER_GYR,
)


POWER_LAW_FRACTION = 0.0466


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
