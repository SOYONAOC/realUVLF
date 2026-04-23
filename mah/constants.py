"""Planck 2018 cosmological and physical constants.

All cosmological values from Planck 2018 baseline.
Unit conversions use astropy.units.
"""
from __future__ import annotations

import astropy.constants as astro_const
import astropy.units as u

# ---------------------------------------------------------------------------
# Physical constants (via astropy)
# ---------------------------------------------------------------------------
SPEED_OF_LIGHT_A_PER_S: float = astro_const.c.to(u.AA / u.s).value

GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN: float = astro_const.G.to(
    u.Mpc * u.km**2 / u.Msun / u.s**2
).value

PROTON_MASS_KG: float = astro_const.m_p.to(u.kg).value
BOLTZMANN_CONSTANT_J_K: float = astro_const.k_B.to(u.J / u.K).value

# ---------------------------------------------------------------------------
# Unit conversion factors (via astropy)
# ---------------------------------------------------------------------------
SECONDS_PER_GYR: float = (1.0 * u.Gyr).to(u.s).value
KM_PER_MPC: float = (1.0 * u.Mpc).to(u.km).value
YEARS_PER_GYR: float = (1.0 * u.Gyr).to(u.yr).value

# ---------------------------------------------------------------------------
# Planck 2018 cosmological parameters
# ---------------------------------------------------------------------------
PLANCK18_H: float = 0.674
PLANCK18_H0_KM_S_MPC: float = 100.0 * PLANCK18_H
PLANCK18_OMEGA_M: float = 0.315
PLANCK18_OMEGA_B: float = 0.04897
PLANCK18_OMEGA_LAMBDA: float = 1.0 - PLANCK18_OMEGA_M
PLANCK18_H0_GYR: float = PLANCK18_H0_KM_S_MPC * SECONDS_PER_GYR / KM_PER_MPC

# ---------------------------------------------------------------------------
# AB magnitude zeropoint: M_UV = -2.5 * log10(L_nu [erg/s/Hz]) + 51.60
# ---------------------------------------------------------------------------
AB_ZEROPOINT_LNU: float = 51.60
