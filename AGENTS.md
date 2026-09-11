# AGENTS.md

## Scope

These instructions apply to the whole AuroraLF repository.

Work from the repository root unless a task explicitly says otherwise. Treat
this repository as a research codebase: preserve scientific meaning, data
provenance, and reproducibility ahead of cosmetic cleanup.

## Agent-Owned Workflow

Agents are responsible for the full technical workflow: reading code, planning,
editing, running jobs, checking outputs, debugging failures, and summarizing the
result. Do not ask the user to inspect code, logs, or implementation details as
the primary review path.

The user's review surface is slides. When a result needs human inspection,
prepare or update the relevant slide deck under `slides/` and keep the chat
summary focused on conclusions, assumptions, and remaining decisions rather than
code-level detail.

## Project Map

- `auroralf/mah/`: halo assembly history generation, including McBride-style
  Monte Carlo histories and TNG/THESAN cache-backed MAH backends.
- `auroralf/sfr/`: star-formation model utilities.
- `auroralf/chemistry/`: MZR prior and gas-regulator metallicity diagnostics
  along fixed MAH/SFR histories.
- `auroralf/ssp/`: SSP loading and UV convolution utilities.
- `auroralf/uvlf/`: UV luminosity function sampling, HMF weighting, dust
  correction, and Pop II IMF mode logic.
- `tests/`: focused regression tests.
- `scripts/run/`: production or batch workflow entry points.
- `scripts/submit/`: SLURM submission wrappers.
- `scripts/data/`: external simulation data selection, download, and compact
  cache construction utilities for TNG/THESAN MAH workflows.
- `scripts/plot/`: plotting and visual comparison scripts.
- `scripts/analysis/`: post-processing and result comparison scripts.
- `scripts/experiments/`: one-off diagnostics and exploratory scripts; do not
  treat these as stable public APIs.

## Environment

Use the project virtual environment explicitly:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests
```

For focused checks, run only the affected tests, for example:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_imf_modes.py tests/test_hmf_sampling.py
```

Run project scripts from the repository root so relative data paths and package
imports resolve consistently:

```bash
PYTHONPATH=. .venv/bin/python scripts/experiments/<script>.py
```

Do not silently switch to the system Python when `.venv/bin/python` is expected.
If a dependency is missing, fix the project environment deliberately with `uv`
or report the setup problem.

The tracked `pyproject.toml` and `uv.lock` define the supported Python and exact
dependency versions. Restore the environment with `uv sync --frozen --all-groups`;
preserve the scientific dependency pins. Run `.venv/bin/ruff check` and
`.venv/bin/ruff format --check` for the experiment files listed in the Ruff
configuration, alongside the relevant regression tests.

## Data And Outputs

- `external_data/`: external source data, observational constraints, SSP
  spectra, empirical model releases, raw TNG/THESAN downloads, and literature
  source packages. Preserve source-data provenance and avoid editing these files
  unless the task is explicitly about data ingestion or correction.
- `data_save/`: reusable intermediate products, long-running `.npz` outputs,
  compact TNG/THESAN MAH caches, TNG merger-event caches, and summary tables.
- `outputs/`: logs, progress files, quick-look plots, and one-off diagnostics.
- `temp_data/`: scratch caches and temporary `.npz` products.
- `slides/`: Beamer sources, slide PDFs, and slide-dependent assets.
- `nbody/`: N-body experiment notes and launch documentation.

When adding new generated outputs, choose `data_save/` for reusable products and
`outputs/` for temporary diagnostics. Keep formal slide material under `slides/`.
For slide figures, prefer vector `.pdf` assets under `slides/assets/`; keep
preview rasters and draft figures in `outputs/` unless the user requests a
tracked raster asset.

Large external libraries, raw source packages, and simulation downloads may be
ignored by git, including `external_data/ssp_spectra/`,
`external_data/empirical_models/universemachine_dr1/tarballs/`,
`external_data/tng/`, `external_data/thesan/`, and literature source tarballs. A
fresh clone may not contain these files. If required data or compact caches are
missing, report the exact missing path instead of creating fake data, changing
default paths, or silently skipping the calculation.

## Scientific Constraints

- Fail fast. Do not hide missing data, failed imports, invalid parameters, or
  failed jobs with broad fallbacks, placeholder arrays, synthetic data, cached
  plots, or silent default values.
- `main` is the unified production branch. Do not maintain a separate old
  `topheavyIMF` path in parallel; use switches in the unified model for
  historical comparisons.
- The default MAH backend is `mcbride`. Use `mah_backend="tng"` or
  `mah_backend="thesan"` only with explicit real compact cache paths; do not
  fabricate cache files or resample around a missing simulation input.
- Preserve units and conversions:
  - halo mass: `Msun`
  - SFR: `Msun/yr`
  - UV luminosity: `erg/s/Hz`
  - HMF: `Mpc^-3 Msun^-1`
  - metallicity gates and chemistry summaries: linear `Z/Zsun`
  - SSP tables commonly provide ages in `Myr`; convert to `Gyr` before
    convolving with MAH/SFR histories stored in `Gyr`.
- The current production HMF path is `hmf` Reed07. Deprecated model names such
  as `massfunc_st` and `hmf_watson13_fof` intentionally raise errors; do not
  restore them unless the user explicitly asks for a historical comparison.
- Pop II top-heavy IMF variants are source-time gated. Do not replace the SSP
  globally when the intended behavior is `z10_mild_topheavy` or
  `mah_burst_mild_topheavy`.
- Current IMF modes are `canonical`, `z10_mild_topheavy`, and
  `mah_burst_mild_topheavy`. The default transition parameters are
  `source_redshift_gate_enabled=False`,
  `growth_time_threshold_myr=50.0`, and
  `metallicity_topheavy_max_zsun=0.05`. The old `z_topheavy_min=10.0`
  threshold is retained only for explicit historical comparisons.
- Do not re-enable the source-time `z>=10` top-heavy gate by default. Use
  `source_redshift_gate_enabled=True` or
  `--enable-source-redshift-topheavy-gate` only when reproducing old z-gated
  runs.
- `IMFTransitionParameters.metallicity_topheavy_max_zsun=None` disables the
  birth-metallicity gate and recovers the historical top-heavy behavior. The
  production CLI equivalent is `--disable-metallicity-topheavy-gate`.
- Non-canonical IMF modes with a non-`None` metallicity gate require
  an explicit regulator or MZR birth-metallicity source; keep this as an
  explicit error rather than silently dropping the gate.
- `topheavy_ssp_metallicity` selects the HDF5 SSP template metallicity. It is
  not the gas birth-metallicity gate, even though the default value is also
  `0.05 Zsun`.
- Regulator metallicity evolution is diagnostic along fixed MAH/SFR tracks and
  must not feed back into the SFR model. `birth_metallicity_zsun_grid` is the
  source-time metallicity used for IMF gating; `gas_metallicity_zsun_grid` is
  the regulator gas metallicity.
- Burst scatter and MZR/regulator metallicity are archived production features.
  Production requires `enable_archived_burst_scatter=False`,
  `burst_scatter_dex=0`, `enable_archived_metallicity=False`, and
  `metallicity_source="none"`. Historical configs must explicitly opt in to
  each archived feature; do not let old TOML files silently reactivate them.
- The SFR delay model is controlled by `enable_time_delay`. Low-level Python
  APIs keep `False` as their backward-compatible default; the production UVLF
  script defaults to delay enabled and uses `--disable-time-delay` only for
  historical no-delay comparisons.
- `burst_scatter_dex > 0` applies a correlated lognormal multiplier to the SFR
  after the delay-SFR calculation and before metallicity/UV convolution. The
  default correlation timescale is `20 Myr`.
- The default burst scatter is mass-conserving per halo:
  `SFR_burst(t) = SFR_0(t) B(t) integral SFR_0 dt / integral SFR_0(t) B(t) dt`.
  This is recorded as `burst_scatter_mass_conserving=True`. Do not switch to
  non-conserving scatter unless the user explicitly asks for that comparison.
- UV convolution should use valid active history segments only. Do not adjust
  scientific parameters, mass cuts, redshift cuts, or units just to make a test
  or run finish.
- HDF5 top-heavy SSP loading requires an exact linear `Z/Zsun` bin, for example
  `0.05`. If required SSP data or `h5py` are missing, surface the exact missing
  dependency/path.
- The dust-corrected UVLF currently applies the physical clipping
  `phi_obs = min(phi_obs_raw, phi_nodust_obs)`.

## Production Runs

Large UVLF comparisons are compute jobs. Submit them through the SLURM wrapper
instead of running the production target directly on a login node:

```bash
PYTHONPATH=. .venv/bin/python scripts/submit/submit_uvlf_v2.py \
  --config configs/uvlf/production.toml --dry-run
```

The target script `scripts/run/run_uvlf_v2.py` is the unified production entry
point for canonical, top-heavy, metallicity-gated, burst-scatter, TNG, and
THESAN UVLF runs. It requires a SLURM allocation; all scientific behavior is
selected in the typed TOML configuration. The production config enables the
time-delay model. The old `run_uvlf_compare_imf_no_delay_all_z.py`,
`submit_uvlf_imf_compare.py`, and
`run_uvlf_mass_function_compare_full.py` entry points are intentionally disabled
migration shims.

When reproducing non-canonical IMF modes with a metallicity gate, set
`enable_archived_metallicity = true` and choose the regulator or MZR backend
under `[star_formation]`. This path is archival and must not enter production.

Use `scripts/analysis/sweep_regulator_metallicity.py` for regulator parameter
scans and `scripts/analysis/plot_regulator_mzr_validation.py` /
`scripts/analysis/plot_regulator_metallicity_redshift_evolution.py` for MZR and
redshift-evolution checks. Plotting scripts under `scripts/plot/` and
`scripts/analysis/` expect real observational/source files under
`external_data/`; do not synthesize missing observations.

Use `scripts/data/` for TNG/THESAN source selection, API/download planning, and
compact MAH cache construction. TNG API workflows require a real `TNG_API_KEY`;
THESAN workflows require the real source files under `external_data/thesan/`.
Generated compact caches intended for reuse belong under `data_save/`, while
diagnostic summaries and quick-look plots belong under `outputs/`.

## Verification

Run the full focused suite after model changes:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests
```

For narrower edits, use the relevant tests:

- IMF and source-time gate behavior: `tests/test_imf_modes.py`
- regulator/MZR metallicity and pipeline metadata: `tests/test_chemistry.py`
- mass-conserving SFR burst scatter and production CLI defaults:
  `tests/test_burst_scatter.py`
- HMF validation and Reed07 unit handling: `tests/test_hmf_sampling.py`
- metallicity history summaries: `tests/test_metallicity_history_summary.py`
- MZR calibration helpers: `tests/test_mzr_constraints.py`
- TNG MAH backend and run-script CLI exposure: `tests/test_tng_mah_backend.py`
- THESAN MAH backend and run-script CLI exposure:
  `tests/test_thesan_mah_backend.py`
- TNG selection and merger-event cache builders:
  `tests/test_tng_selection_script.py` and
  `tests/test_tng_merger_event_cache.py`
- THESAN download planning: `tests/test_thesan_download_plan.py`

## Editing Rules

- Check `git status --short` before modifying files. This repository may have
  large moves, generated assets, or conflict markers in progress; do not revert
  unrelated user changes.
- Keep changes narrowly scoped to the requested behavior.
- Do not add temporary Python scripts at the repository root. Root-level `*.py`
  files are ignored here; place scripts under `scripts/run/`, `scripts/data/`,
  `scripts/plot/`, `scripts/analysis/`, or `scripts/experiments/` according to
  their purpose.
- Update README/API documentation when changing public functions, returned
  fields, accepted modes, units, or output paths.
- Prefer structured data loading and explicit validation over ad hoc parsing.
- Add or update focused tests for behavioral changes. If tests cannot be run,
  state the exact reason and the command that should be run later.
