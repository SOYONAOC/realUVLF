"""User-prescribed random first crossing, UV-only; NOT a causal Pop III model.

No enrichment/feedback/pristine survival closure is supplied by this experiment.
Pop II is unchanged. Random q is independent of assembly, once per track.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import tomllib
import numpy as np


@dataclass(frozen=True)
class Config:
    run_id: str
    z: float
    n_mass: int
    n_tracks: int
    track_chunk: int
    n_grid: int
    workers: int
    seed: int
    z_start: float
    logmass_min: float
    logmass_max: float
    q_log10_mean: float
    q_log10_sigma: float
    efficiencies: list[float]
    popii_ssp: str
    popiii_ssp: str
    lookback_myr: float
    sfr_convolution: str = 'dense'

    @classmethod
    def load(cls, path):
        import re
        path = Path(path).resolve(strict=True)
        cfg = cls(**tomllib.loads(path.read_text()))
        if not re.fullmatch(r'AUR-EX-\d{4}-R\d{3}', cfg.run_id):
            raise ValueError('invalid run id')
        for key in ('n_mass', 'n_tracks', 'track_chunk', 'n_grid', 'workers', 'seed'):
            if type(getattr(cfg, key)) is not int or getattr(cfg, key) < 1:
                raise ValueError(key)
        if cfg.n_mass < 2 or cfg.n_grid < 2 or cfg.n_tracks % cfg.track_chunk:
            raise ValueError('invalid dimensions / nondivisible track chunks')
        if not all(np.isfinite(getattr(cfg, k)) for k in
                   ('z', 'z_start', 'logmass_min', 'logmass_max', 'q_log10_mean', 'q_log10_sigma', 'lookback_myr')):
            raise ValueError('nonfinite parameters')
        if not (0 <= cfg.z < cfg.z_start and cfg.logmass_min < cfg.logmass_max
                and cfg.q_log10_sigma > 0 and cfg.lookback_myr > 0):
            raise ValueError('invalid physical parameters')
        if not cfg.efficiencies or any(not np.isfinite(e) or not 0 < e <= 1 for e in cfg.efficiencies):
            raise ValueError('invalid efficiencies')
        if cfg.efficiencies != sorted(set(cfg.efficiencies)):
            raise ValueError('efficiencies must be unique and sorted')
        if cfg.sfr_convolution not in ('dense','direct'):
            raise ValueError('sfr_convolution')
        data = asdict(cfg)
        for key in ('popii_ssp', 'popiii_ssp'):
            data[key] = str((path.parent / data[key]).resolve(strict=True))
        return cls(**data)


def draw_logq(seed, mass_index, n_tracks, mean, sigma):
    # Independent component stream; do not consume any MAH or HMF random numbers.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x51425552, mass_index]))
    return rng.normal(mean, sigma, n_tracks)


def first_crossing(time, mass, cooling, logq):
    """First passage in log(M/Mcool), linearly interpolated between time nodes.

    status: 0 not crossed; 1 resolved crossing; 2 left censored. NaN burst
    quantities for statuses 0/2 are intentional missing event coordinates.
    Below-PopII-floor masses remain valid dark matter history, not active SFR.
    """
    t, m, c = map(lambda x: np.asarray(x, float), (time, mass, cooling))
    q = np.asarray(logq, float)
    if t.ndim != 2 or not (t.shape == m.shape == c.shape) or q.shape != (len(t),):
        raise ValueError('inconsistent shapes')
    if (not all(np.all(np.isfinite(x)) for x in (t, m, c, q)) or
            np.any(np.diff(t, axis=1) <= 0) or np.any(m <= 0) or np.any(c <= 0)):
        raise ValueError('invalid histories')
    r = np.log10(m / c)
    above = r >= q[:, None]
    any_above = above.any(axis=1)
    j = above.argmax(axis=1)
    status = np.where(above[:, 0], 2, np.where(any_above, 1, 0)).astype(np.int8)
    tb = np.full(len(t), np.nan)
    mb = tb.copy()
    rows = np.flatnonzero(status == 1)
    hi = j[rows]; lo = hi - 1
    f = (q[rows] - r[rows, lo]) / (r[rows, hi] - r[rows, lo])
    tb[rows] = t[rows, lo] + f * (t[rows, hi] - t[rows, lo])
    # Same interpolation of log M and log Mcool gives exact threshold equality.
    mb[rows] = 10**(np.log10(m[rows, lo]) + f*np.log10(m[rows, hi]/m[rows, lo]))
    return status, tb, mb


def burst_light(time, mass, cooling, logq, ages, kernel, fb, lookback):
    from auroralf.ssp import interpolate_ssp_luminosity
    status, tb, mb = first_crossing(time, mass, cooling, logq)
    age = (time[:, -1] - tb)*1000  # Gyr -> Myr
    light = np.zeros(len(time))
    mask = (status == 1) & (age <= lookback)
    light[mask] = fb*mb[mask]*interpolate_ssp_luminosity(age[mask], ages, kernel)
    return dict(status=status, burst_time_gyr=tb, burst_halo_mass_msun=mb,
                age_myr=age, popiii_per_efficiency=light)


def initialize_worker(cfg):
    global STATE
    from astropy.cosmology import FlatLambdaCDM
    from auroralf.mah import Cosmology
    from auroralf.ssp import load_uv1600_table, load_popiii_uv_luminosity_table
    cosmo = Cosmology()
    astro = FlatLambdaCDM(H0=cosmo.h0_km_s_mpc, Om0=cosmo.omega_m, Ob0=cosmo.omega_b)
    dt = (astro.age(cfg.z).value - astro.age(cfg.z_start).value)/(cfg.n_grid-1)
    if dt*(cfg.n_grid-1)*1000 <= cfg.lookback_myr:
        raise ValueError('left-censored bursts not outside declared UV lookback')
    a2, k2 = load_uv1600_table(cfg.popii_ssp)
    a3, k3 = load_popiii_uv_luminosity_table(cfg.popiii_ssp)
    if cfg.lookback_myr > min(a2[-1], a3[-1]):
        raise ValueError('SSP does not cover lookback')
    STATE = cfg, cosmo, dt, a2, k2, a3, k3


def one_mass(task):
    from auroralf.mah import generate_halo_histories
    from auroralf.cooling import compute_atomic_cooling_mass_msun
    from auroralf.seeding import derive_pipeline_random_seeds
    from auroralf.sfr import compute_sfr_from_tracks
    from auroralf.ssp import compute_final_ssp_observable_from_sfr_grid
    cfg, cosmo, dt, a2, k2, a3, k3 = STATE
    index, final_mass = task
    logq = draw_logq(cfg.seed, index, cfg.n_tracks, cfg.q_log10_mean, cfg.q_log10_sigma)
    pieces = []
    for start in range(0, cfg.n_tracks, cfg.track_chunk):
        block_seed = int(np.random.SeedSequence([cfg.seed, 0x424C4F43, start]).generate_state(1, dtype=np.uint64)[0])
        seeds = derive_pipeline_random_seeds(block_seed, redshift=cfg.z, mass_index=index)
        histories = generate_halo_histories(n_tracks=cfg.track_chunk, z_final=cfg.z,
            Mh_final=final_mass, z_start_max=cfg.z_start, cosmology=cosmo,
            random_seed=seeds.mah, time_grid_mode='uniform_in_t', dt=dt,
            store_inactive_history=True, sampler='mcbride')
        tracks = compute_sfr_from_tracks(histories.tracks, cosmology=cosmo, enable_time_delay=True,
                                        regular_convolution_backend=cfg.sfr_convolution)
        shape = (cfg.track_chunk, -1)
        time = tracks['t_gyr'].reshape(shape)
        mass = tracks['Mh'].reshape(shape)
        cooling = compute_atomic_cooling_mass_msun(tracks['z'].reshape(shape), cosmology=cosmo)
        p2 = compute_final_ssp_observable_from_sfr_grid(t_grid_gyr=time,
            sfr_grid=tracks['SFR'].reshape(shape), active_grid=tracks['active_flag'].reshape(shape),
            ssp_age_myr=a2, ssp_observable_per_msun=k2, lookback_max_myr=cfg.lookback_myr)
        event = burst_light(time, mass, cooling, logq[start:start+cfg.track_chunk],
                            a3, k3, cosmo.omega_b/cosmo.omega_m, cfg.lookback_myr)
        fixed = burst_light(time, mass, cooling, np.zeros(cfg.track_chunk),
                            a3, k3, cosmo.omega_b/cosmo.omega_m, cfg.lookback_myr)
        event.update(popii=p2, fixed_q1_per_efficiency=fixed['popiii_per_efficiency'])
        pieces.append(event)
    result = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    result['logq'] = logq
    return index, result
