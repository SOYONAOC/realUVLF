from dataclasses import replace
import numpy as np
import pytest
from scripts.experiments.random_q_burst import draw_logq, first_crossing, burst_light, Config, initialize_worker, one_mass


def test_distribution_and_independent_stream():
    q = draw_logq(123, 0, 200000, .5, 1.5)
    assert abs(q.mean()-.5) < .015
    assert abs(q.std()-1.5) < .015
    assert abs(np.mean((q >= -1) & (q <= 2))-.682689) < .005
    assert np.any(q < -1) and np.any(q > 2)
    np.testing.assert_array_equal(q, draw_logq(123, 0, 200000, .5, 1.5))
    assert not np.array_equal(q, draw_logq(123, 1, 200000, .5, 1.5))


def test_first_passage_not_repeated_or_snapshot():
    t = np.tile([.1, .2, .3, .4], (3, 1))
    m = np.array([[1, 100, 1, 100], [100, 1, 1, 100], [1, 2, 3, 4.]])
    status, tb, mb = first_crossing(t, m, np.ones_like(m), np.ones(3))
    np.testing.assert_array_equal(status, [1, 2, 0])
    np.testing.assert_allclose(tb[0], .15)
    np.testing.assert_allclose(mb[0], 10)
    assert np.isnan(tb[1:]).all() and np.isnan(mb[1:]).all()


def test_moving_threshold_and_no_future_light():
    t = np.array([[.1, .101], [.1, .101]])
    m = np.array([[10., 100.], [10., 100.]])
    c = np.array([[10., 20.], [10., 20.]])
    x = np.array([np.log10(2), 3.])
    result = burst_light(t, m, c, x, np.array([.01, 10.]), np.array([1e20, 1e20]), .16, 100.)
    f = np.log10(2)/np.log10(5)
    assert result['burst_time_gyr'][0] == pytest.approx(.1+.001*f)
    assert result['popiii_per_efficiency'][0] == pytest.approx(.16*10*10**f*1e20)
    assert result['popiii_per_efficiency'][1] == 0


def test_real_inputs_popii_pipeline_equivalence():
    from pathlib import Path
    from auroralf.seeding import derive_pipeline_random_seeds
    from auroralf.uvlf.pipeline import run_halo_uv_pipeline
    from auroralf.mah import Cosmology
    root = Path(__file__).resolve().parents[1]
    cfg = replace(Config.load(root/'configs/experiments/random_q_R018.toml'),
                  n_mass=2, n_tracks=4, track_chunk=4, n_grid=64, workers=1)
    initialize_worker(cfg)
    _, a = one_mass((0, 1e9))
    _, b = one_mass((0, 1e9))
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    block_seed = int(np.random.SeedSequence([cfg.seed, 0x424C4F43, 0]).generate_state(1, dtype=np.uint64)[0])
    seeds = derive_pipeline_random_seeds(block_seed, redshift=cfg.z, mass_index=0)
    reference = run_halo_uv_pipeline(n_tracks=4, z_final=cfg.z, Mh_final=1e9,
        cosmology=Cosmology(), random_seeds=seeds, z_start_max=cfg.z_start,
        n_grid=64, enable_popiii=False, enable_time_delay=True, ssp_file=cfg.popii_ssp)
    np.testing.assert_allclose(a['popii'], reference.uv_luminosities, rtol=1e-12)
    # All dark-matter histories below the PopII atomic floor remain usable for q.
    _, low = one_mass((1, 1e5))
    assert np.all(low['popii'] == 0)


def test_invalid_history_rejected():
    with pytest.raises(ValueError):
        first_crossing(np.array([[1., 0.]]), np.ones((1, 2)), np.ones((1, 2)), np.zeros(1))


def test_direct_convolution_matches_dense():
    from auroralf.sfr.calculator import _compute_extended_burst_convolution_direct as direct
    from auroralf.sfr.calculator import _compute_extended_burst_convolution_vectorized_regular_grid as dense
    rng = np.random.default_rng(31)
    t = np.tile(np.linspace(.046, .275, 960), (12,1))
    source = 10**rng.uniform(-4, 8, t.shape)
    active = np.arange(960)[None,:] >= np.arange(12)[:,None]*80
    td = np.full_like(t, .035)
    kwargs=dict(t_grid=t,source_grid=source,active_grid=active,td_burst_grid=td,kappa=.1,max_lookback_gyr=.1)
    np.testing.assert_allclose(direct(**kwargs),dense(**kwargs),rtol=2e-12,atol=1e-12)
    kwargs['t_grid']=t**2
    with pytest.raises(ValueError):
        direct(**kwargs)


def test_real_direct_matches_dense_observables():
    from pathlib import Path
    cfg = replace(Config.load(Path(__file__).resolve().parents[1]/'configs/experiments/random_q_R018.toml'),
                  n_mass=2,n_tracks=8,track_chunk=8,n_grid=960,workers=1)
    for mass in (1e5,1e8,1e10,1e12):
        initialize_worker(cfg)
        _, dense = one_mass((0,mass))
        initialize_worker(replace(cfg,sfr_convolution='direct'))
        _, direct = one_mass((0,mass))
        for key in dense:
            np.testing.assert_allclose(direct[key],dense[key],rtol=2e-12,atol=0,equal_nan=True)
