"""Meaningful selection and normalization checks for observational post-processing."""

import numpy as np
import pytest

from scripts.analysis.compare_random_q_heii_observations import conditional_stats


def test_conditional_cdf_preserves_zero_unknown_and_mass_clusters():
    # Unknown cannot become a non-detection; zero PopIII line must remain in denominator.
    result = conditional_stats(
        np.array([0.0, 2.0, np.nan, 4.0]),
        np.array([1.0, 2.0, 3.0, 1.0]),
        np.array([0, 0, 1, 2]),
        2.0,
    )
    assert result["fraction_at_or_below_reference_flux"] == 0.75
    assert result["zero_fraction_known"] == 0.25
    assert result["unknown_weight_fraction"] == pytest.approx(3 / 7)
    assert result["mass_clusters_known"] == 2
    assert result["effective_mass_clusters"] == pytest.approx(16 / 10)
    # Uniform batch averaging changes abundance but cannot change conditional statistics.
    scaled = conditional_stats(
        np.array([0.0, 2.0, np.nan, 4.0]),
        np.array([1.0, 2.0, 3.0, 1.0]) / 4,
        np.array([0, 0, 1, 2]),
        2.0,
    )
    assert scaled["flux_q16_q50_q84"] == result["flux_q16_q50_q84"]
    assert scaled["density_selected_mpc3"] == result["density_selected_mpc3"] / 4
    with pytest.raises(ValueError, match="invalid known"):
        conditional_stats(np.array([np.nan]), np.ones(1), np.zeros(1, dtype=int), 2.0)
