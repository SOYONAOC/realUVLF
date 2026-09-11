from pathlib import Path

import numpy as np
import pytest

from auroralf.experiments.heii import cluster_sum_and_se, evaluate_kernel, load_kernel

SSP = Path("external_data/ssp_spectra/schaerer2010_pop3/pop3_ge0_logE_500_001_is5.25")


def test_real_matching_line_table_and_age_evolution():
    kernel = load_kernel(SSP, SSP.with_suffix(".22"))
    assert kernel.caseb_max_relative_error < 0.01
    np.testing.assert_allclose(kernel.line_per_msun[:3], [6.3954e35, 3.77268e35, 1.27738e34])
    np.testing.assert_array_equal(kernel.age_myr[:4], [0.01, 1, 2, 3])
    # Strong line fades before the UV peak; this is a real SSP prediction.
    assert kernel.line_per_msun[2] / kernel.line_per_msun[0] < 0.021
    assert kernel.uv_per_msun[2] > kernel.uv_per_msun[0]
    for method in ("linear_log_age", "linear_age", "log_log_age"):
        np.testing.assert_allclose(
            evaluate_kernel(kernel.age_myr[:4], kernel, method),
            kernel.line_per_msun[:4],
            rtol=1e-13,
        )
    with pytest.raises(ValueError, match="older"):
        evaluate_kernel([1001], kernel)
    with pytest.raises(ValueError, match="invalid"):
        evaluate_kernel([np.nan], kernel)


def test_different_imf_cannot_be_silently_used():
    with pytest.raises(ValueError, match="model sets"):
        load_kernel(SSP, SSP.with_name("pop3_ge0_sal_500_001_is5.22"))


def test_cluster_estimator_preserves_batch_normalization():
    # Each vector represents already-weighted halo-mass clusters in one run.
    mean, se = cluster_sum_and_se([np.array([1.0, 3.0]), np.array([2.0, 4.0])])
    assert mean == 5
    assert se == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError, match="clusters"):
        cluster_sum_and_se([])


def test_flux_counts_exclude_unknown_events_and_keep_threshold_equality():
    from scripts.analysis.analyze_random_q_heii import cumulative_rows

    # Arithmetic fixture: one unknown event, one zero emitter, exact cut values.
    flux = np.array([[np.nan, 0, 1, 2], [0, 2, 3, 4]])
    weights = np.array([0.1, 0.2])
    np.testing.assert_allclose(
        cumulative_rows(flux, weights, [1, 2, 4]), [[0.2, 0.1, 0], [0.6, 0.6, 0.2]]
    )
    selection = np.array([[True, False, True, False], [False, True, False, True]])
    np.testing.assert_allclose(
        cumulative_rows(flux, weights, [1, 2, 4], selection), [[0.1, 0, 0], [0.4, 0.4, 0.2]]
    )
