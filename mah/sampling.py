from __future__ import annotations

import numpy as np

from .models import GaussianApproximation, POWER_LAW_FRACTION


def mass_ratio_m12(mass_ref: float) -> float:
    return mass_ref / 1.0e12


def sample_mcbride_power_law_component(mass_ref: float, size: int, rng: np.random.Generator) -> np.ndarray:
    if size == 0:
        return np.empty((0, 2), dtype=float)

    mass_scale = mass_ratio_m12(mass_ref)
    accepted: list[np.ndarray] = []
    count = 0

    while count < size:
        batch = max(4_096, 6 * (size - count))
        beta = rng.uniform(-10.0, 0.0, size=batch)
        x_value = 7.443 * np.exp(0.6335 * beta + 0.2626 * mass_scale**0.1992) - 2.852 * mass_scale**(-0.05412)
        keep = rng.random(batch) < np.exp(-(x_value**2))
        if np.any(keep):
            accepted.append(beta[keep])
            count += int(np.count_nonzero(keep))

    beta_samples = np.concatenate(accepted)[:size]
    return np.column_stack((beta_samples, np.zeros(size, dtype=float)))


def appendix_a_t1(beta: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(1.174 * beta))


def appendix_a_t2(beta: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.7671 * beta - 0.1269))


def appendix_a_yprime_to_gamma(beta: np.ndarray, yprime: np.ndarray, mass_ref: float) -> np.ndarray:
    mass_scale = mass_ratio_m12(mass_ref)
    t1 = appendix_a_t1(beta)
    offset = (28.85 + 0.4537 * beta) * (1.0 - t1) + (28.38 + 0.7624 * beta) * t1
    return yprime + offset - 29.21 * mass_scale**(-0.001933)


def appendix_a_joint_weight(beta: np.ndarray, yprime: np.ndarray, mass_ref: float) -> np.ndarray:
    mass_scale = mass_ratio_m12(mass_ref)
    t1 = appendix_a_t1(beta)
    t2 = appendix_a_t2(beta)

    x_value = (-1.722 - 0.1568 * beta + 0.007592 * beta**2) * (1.0 - t2)
    x_value += (1.242 + 0.3138 * beta - 0.01399 * beta**2) * t2

    y_value = 13.39 * (1.0 - 1.224 * np.tanh(1.043 * yprime)) * (1.0 - 0.08018 * beta)
    exponent = -((x_value * mass_scale**(-0.05569)) ** 2) - ((y_value * mass_scale**(-0.05697)) ** 2)

    gamma = appendix_a_yprime_to_gamma(beta, yprime, mass_ref)
    valid = (beta > -8.0) & (beta < 12.0) & (yprime > 0.0) & (yprime < 3.0) & (gamma > 1.0e-3)

    weight = np.zeros_like(beta, dtype=float)
    weight[valid] = np.exp(exponent[valid])
    return weight


def sample_mcbride_joint_component(mass_ref: float, size: int, rng: np.random.Generator) -> np.ndarray:
    if size == 0:
        return np.empty((0, 2), dtype=float)

    beta_chunks: list[np.ndarray] = []
    gamma_chunks: list[np.ndarray] = []
    count = 0

    while count < size:
        batch = max(32_768, 10 * (size - count))
        beta = rng.uniform(-8.0, 12.0, size=batch)
        yprime = rng.uniform(0.0, 3.0, size=batch)
        keep = rng.random(batch) < appendix_a_joint_weight(beta, yprime, mass_ref)
        if np.any(keep):
            beta_kept = beta[keep]
            gamma_kept = appendix_a_yprime_to_gamma(beta_kept, yprime[keep], mass_ref)
            beta_chunks.append(beta_kept)
            gamma_chunks.append(gamma_kept)
            count += int(np.count_nonzero(keep))

    return np.column_stack((np.concatenate(beta_chunks)[:size], np.concatenate(gamma_chunks)[:size]))


def sample_mcbride_appendix_a(mass_ref: float, size: int, rng: np.random.Generator) -> np.ndarray:
    n_power_law = rng.binomial(size, POWER_LAW_FRACTION)
    samples = np.vstack(
        (
            sample_mcbride_power_law_component(mass_ref, n_power_law, rng),
            sample_mcbride_joint_component(mass_ref, size - n_power_law, rng),
        )
    )
    rng.shuffle(samples, axis=0)
    return samples


def estimate_gaussian_approximation(
    mass_ref: float,
    rng: np.random.Generator,
    pilot_samples: int,
) -> GaussianApproximation:
    pilot = sample_mcbride_appendix_a(mass_ref, pilot_samples, rng)
    covariance = np.cov(pilot, rowvar=False) + np.eye(2) * 1.0e-10
    return GaussianApproximation(mean=pilot.mean(axis=0), covariance=covariance)


def sample_parameters(
    mass_ref: float,
    size: int,
    sampler: str,
    rng: np.random.Generator,
    pilot_samples: int,
) -> tuple[np.ndarray, GaussianApproximation | None]:
    if sampler == "mcbride":
        return sample_mcbride_appendix_a(mass_ref, size, rng), None

    approximation = estimate_gaussian_approximation(mass_ref, rng, pilot_samples)
    draws = rng.multivariate_normal(approximation.mean, approximation.covariance, size=size)
    return draws, approximation
