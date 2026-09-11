# AuroraLF

## v2 typed API

AuroraLF v2 exposes one strict configuration boundary and one in-memory run
function:

```python
from pathlib import Path

from auroralf import UVLFRunConfig, UVLFRunResult, run_uvlf

config = UVLFRunConfig.from_toml(Path("configs/uvlf/production.toml"))
result: UVLFRunResult = run_uvlf(config)
canonical_z6 = result.for_redshift(6.0).for_mode("canonical")
```

The Pop II IMF gate, burst scatter, and metallicity models are archived
features. Production uses only the canonical IMF, zero burst scatter, and
`metallicity_source = "none"`. Their implementations remain available only
through explicit archive opt-ins for exact historical reproduction.

Shared reference values come from Astropy in `auroralf/constants.py`.
Production uses Astropy `Planck18`: `H0=67.66 km/s/Mpc`, `Omega_m=0.30966`,
`Omega_b=0.04897`, `sigma8=0.8102`, and `n_s=0.9665`. AuroraLF's analytic
equations neglect radiation, so their flat background uses
`Omega_Lambda=1-Omega_m=0.69034`. TNG/THESAN cache workflows retain the exact
simulation value `Omega_m=0.3089` and validate it against cache provenance.

TOML paths are resolved relative to the TOML file, not the current working
directory. `run_uvlf` validates the SSP files used by enabled modes and the
active MAH-backend cache, then delegates to the shared-batch streaming core and returns
`UVLFRunResult`; it does not write `config.output.artifact_path`. All configured
IMF modes reuse each halo's MAH, SFR, and chemistry preparation. Public runs
use serial execution when `sampling.workers = 1` and a spawn-based process pool
otherwise. The active worker count is capped by the number of final halo masses,
and the parallel scheduler keeps at most twice that many running or completed
mass tasks in memory while consuming results in deterministic mass order.
Every v2 TOML file must set a positive `sampling.mass_batch_size`; this bounds
the number of final halo masses prepared in each streaming chunk. Each
redshift uses one Reed07 interpolation grid over the full configured mass
range, so changing the chunk size changes neither seeded masses nor HMF
weights and histogram results.

## Reproducible environment

The supported interpreter and exact dependency versions are declared in
`pyproject.toml` and frozen in `uv.lock`. Create or synchronize a project-local
environment without re-resolving versions:

```bash
uv sync --frozen --all-groups
PYTHONPATH=. .venv/bin/python -m pytest tests
```

Runtime, analysis, test, and slide-build dependencies are separate groups. The
lock currently targets Python `>=3.13,<3.14`; SciPy is intentionally pinned to
`1.17.1` because the tested HMF/CAMB path is incompatible with SciPy `1.18`.

## Project layout

Core code:

- `auroralf/constants.py`: Astropy-derived unit conversions, physical constants, production Planck18 cosmology, and TNG/THESAN simulation cosmology
- `auroralf/model_options.py`: dependency-light IMF/HMF mode names and validation shared by config, results, I/O, and implementations
- `auroralf/driver.py`: single composition root that resolves every configured module switch, SSP kernel, and simulation-cache path before execution
- `auroralf/run_plan.py`: immutable module-switch snapshot and validated worker execution contract
- `auroralf/samples.py`: immutable halo-sample records shared by UVLF computation and persistence
- `auroralf/uvlf/`: UVLF stages, HMF weighting, dust mapping, scheduler messages, and accumulation; archived IMF-gate symbols are not re-exported from the package API
- `auroralf/io/`: artifact schema, HDF5 layout/codec, owned-file transactions, shards, and resume/merge workflows
- `auroralf/mah/`: Monte Carlo halo assembly history generation
- `auroralf/sfr/`: star-formation model utilities
- `auroralf/chemistry/`: MZR prior 和 gas-regulator 金属丰度诊断
- `auroralf/ssp/`: SSP UV convolution utilities
- `auroralf/experiments/`: opt-in VH15 and random-q calculations, configuration,
  artifact verification, and release snapshots; independent of production defaults
- `tests/`: focused regression tests

Workflow code:

- `scripts/run/`: production or batch workflow entry points
- `scripts/submit/`: SLURM submission wrappers
- `scripts/plot/`: plotting and visual comparison scripts
- `scripts/analysis/`: post-processing and result comparison scripts
- `scripts/experiments/`: one-off experiment launchers

Data and generated files:

- `external_data/`: external source data, including observations, SSP spectra, empirical model releases, and literature source packages
- `data_save/`: reusable intermediate products and summary tables; ignored by git
- `outputs/`: logs, progress files, one-off plots, and diagnostics; ignored by git
- `temp_data/`: scratch caches and temporary `.npz` products; ignored by git
- `slides/`: Beamer sources, slide PDFs, and slide assets; ignored by git in this branch
- `archive/`: archived legacy code or notes kept for reference
- `nbody/`: N-body experiment notes and launch documentation

Keep external source data under `external_data/`, with large local libraries ignored by git. Use `data_save/` for reusable computed products and `outputs/` for diagnostics.

## VH15 and random-q experiments

VH15 duty occupation and random first-crossing bursts remain separate scientific
experiments. Their numerical helpers live in `auroralf.experiments`; scripts
handle CLI, plotting, and scheduling. The old `scripts.experiments.random_q_burst`
imports remain compatible. Run from the repository root with `PYTHONPATH=.`.

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/visbal_duty_uvlf.py --config configs/experiments/visbal_duty.toml
PYTHONPATH=. .venv/bin/python scripts/analysis/combine_visbal_batches.py --config configs/experiments/visbal_combine.toml
```

All paths in these TOML files resolve relative to the file. The combination
configuration explicitly lists complete independent run directories, sampling
dimensions and the mass window. `--wide` selects `visbal_combine_wide.toml` and
cannot be combined with `--config`. At least two independent batches are needed
for this script's batch standard error; random-q analysis still accepts one
batch and reports its cluster error. Existing outputs are never overwritten.
Historical manifests and their recorded input hashes remain unchanged.

Random-q submission uses the same snapshot and SHA-256 verification helpers on
local and remote sites. SSP inputs are taken from the selected science TOML,
and must reside inside the repository for portable frozen releases. The remote
host, user, release directory and environment are declared in
`configs/compute/random_q_cp6.toml`; override with `--site-config` on
`scripts/submit/submit_random_q.py`. The cp6 placement policy, explicit local
site choices, and absence of separate queued preflights are preserved.

Numerical opt-in: `compute_sfr_from_tracks` accepts
`regular_convolution_backend="direct"` for shared uniform time grids, using the
original causal kernel and trapezoidal endpoint weights. The default is
`"dense"`. Nonuniform grids and ambiguous lookback/grid coincidences explicitly
error. Scientific parameters and physical windows are unchanged.

The conditional He II follow-up reads existing random-q event masses and ages:

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/analyze_random_q_heii.py --config configs/experiments/heii_random_q.toml
```

It requires completed, checksum-verified run products and matching logE UV/line
SSP tables. TOML paths are configuration-relative. Outputs are line luminosity
functions, integrated flux statistics and source hashes under the declared
`output`, with vector plots under `figures`. This Case-B Pop III contribution
does not predict total-galaxy equivalent widths or instrument detectability;
see [assumptions, results and age-resolution tests](docs/heii-random-q-study.md).

Compare UV-conditioned predictions with GHZ2 and GS-z14-1 using the verified
parent analysis and source-attributed local observational inputs:

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/compare_random_q_heii_observations.py
```

This writes binned population quantiles and target-window statistics under
`data_save/heii_observations_20260910/`, plus a vector observation overlay in the
He II slide assets. Population redshifts remain 12.5 and 14.5; only flux distances
are evaluated at the observed redshifts. See the [comparison record](docs/heii-observation-comparison.md)
for this approximation, measurement conventions and remaining likelihood work.

The [Pop III prediction roadmap](docs/popiii-predictions-roadmap.md) records
the planned He II, PISN, and 21 cm MAP predictions from the shared epsilon=0.03
random-q model, including current status, dependencies, and observational checks.

Experiment code has pinned Ruff checks; the include list in `pyproject.toml`
defines the current scope. After `uv sync --frozen --all-groups`, run:

```bash
.venv/bin/ruff check
.venv/bin/ruff format --check
PYTHONPATH=. .venv/bin/python -m pytest tests
```

## `auroralf.mah.generate_halo_histories()`

导入：

```python
from auroralf.mah import Cosmology, generate_halo_histories
```

输入：

- `n_tracks`
  要生成的 Monte Carlo 轨道条数
- `z_final`
  终止红移
- `Mh_final`
  在 `z_final` 处的 halo mass
- `z_start_max`
  回溯的最高红移，默认 `50.0`
- `M_min`
  最低质量阈值；默认 `None` 时使用
  `auroralf.cooling.compute_atomic_cooling_mass_msun()` 在项目内求解
  `mu=0.61, Tvir=1e4 K` 的 Barkana--Loeb virial-temperature 关系；
  也可以传标量、与红移网格同长度的数组，或 `M_min(z)` 形式的可调用对象
- `cosmology`
  keyword-only 必填的 `auroralf.mah.Cosmology`；生成 MAH、SFR、HMF 与 UVLF
  时必须复用同一对象，library 内部不会构造隐藏默认值
- `random_seed`
  MAH 低层抽样种子
- `time_grid_mode`
  支持：
  - `"uniform_in_z"`
  - `"uniform_in_t"`
  - `"custom"`
- `dt`
  当 `time_grid_mode="uniform_in_t"` 时使用的时间步长，单位 `Gyr`
- `dz`
  当 `time_grid_mode="uniform_in_z"` 时使用的红移步长
- `custom_grid`
  当 `time_grid_mode="custom"` 时使用的自定义红移网格；本轮实现固定按 redshift grid 解释
- `store_inactive_history`
  是否保留低于 `M_min` 之后的历史点
- `sampler`
  `beta, gamma` 的抽样方式，只支持 `"mcbride"` 和 `"gaussian_approximation"`
- `pilot_samples`
  当 `sampler="gaussian_approximation"` 时使用的 pilot sample 数目

输出：

- `HaloHistoryResult`

## `HaloHistoryResult`

字段：

- `tracks`
  扁平表格风格的 `dict[str, np.ndarray]`
- `metadata`
  输入参数回显、宇宙学、采样方式、采样摘要等信息

## `tracks` 字段

- `halo_id`
  轨道编号
- `step`
  该轨道内部的时间步编号
- `z`
  红移，按轨道内部单调下降
- `t_gyr`
  宇宙时间，单位 `Gyr`，按轨道内部单调升序
- `dt_gyr`
  相邻时间步间隔，单位 `Gyr`
- `Mh`
  halo mass
- `dMh_dt_raw`
  原始 halo mass 导数，单位 `Msun/Gyr`；允许为负，用于 MAH/SMAR 诊断
- `dMh_dt_sfr`
  成星路径使用的有效吸积率，单位 `Msun/Gyr`，严格等于
  `maximum(dMh_dt_raw, 0)`
- `dMh_dt_clipped`
  负吸积裁剪标记，严格等于 `dMh_dt_raw < 0`

负吸积统计 metadata 的 denominator 按后端定义：McBride 的
`negative_dmhdt_total_count` 是返回轨道中 finite analytic-rate 行数；TNG/THESAN
则是两端都 resolved 的相邻 snapshot transition 数，不包含每条轨道首步或
unresolved boundary。
- `active_flag`
  是否仍处于有效区间；当 `Mh < M_min` 后为 `False`
- `termination_flag`
  终止状态标记；当前实现使用：
  - `"active"`
  - `"below_M_min"`
  - `"completed"`

## `auroralf.sfr.compute_sfr_from_tracks()`

导入：

```python
from auroralf.sfr import compute_sfr_from_tracks
```

输入：

- `tracks`
  `HaloHistoryResult.tracks` 风格的扁平 `dict[str, np.ndarray]`；至少需要：
  - `halo_id`
  - `step`
  - `z`
  - `t_gyr`
  - `Mh`
  - `dMh_dt_raw`
  - `dMh_dt_sfr`
  - `dMh_dt_clipped`
- `cosmology`
  必填的 keyword-only `auroralf.mah.Cosmology` 实例；调用方应将生成 MAH
  时使用的同一对象继续传入 SFR，确保 Hubble 参数、临界密度与重子比例一致
- `mu`
  平均分子量，默认 `0.61`
- `atomic_cooling_temperature`
  原子冷却阈值，默认 `1e4 K`
- `enable_time_delay`
  是否启用基于 dynamical time 的 extended-burst 延迟核；默认 `False`

输出：

- `dict[str, np.ndarray]`
  保留输入列，并新增：
  - `r_vir`
  - `V_c`
  - `T_vir`
  - `tau_del`
  - `td_burst`
  - `t_src`
  - `Mh_src`
  - `dMh_dt_sfr_src`
  - `fstar_src`
  - `fstar_now`
  - `mdot_burst`
  - `SFR`

说明：

- `SFR` 单位为 `Msun/yr`
- 当 `enable_time_delay=False` 时，直接用当前时刻的 `Mh` 和 `dMh_dt_sfr`
- 当 `enable_time_delay=True` 时，使用
  `g(t-t') \propto (t-t') \exp[-(t-t')/(\kappa t_d)]`
  的 extended-burst 核对
  `fstar(Mh(t')) * dMh_dt_sfr(t')`
  做时间卷积
- `tau_del/t_src/Mh_src/dMh_dt_sfr_src` 仍保留，作为 source-time 诊断量
- `mdot_burst` 仍保留，表示只对 `dMh/dt` 做 kernel 卷积后的诊断量；真正进入 delay-SFR 的是
  `kernel * fstar(Mh) * dMh_dt_sfr` 的积分
- 若 `T_vir < 1e4 K`，则 `SFR = 0`

最小调用：

```python
from auroralf.mah import Cosmology, generate_halo_histories
from auroralf.sfr import compute_sfr_from_tracks

cosmology = Cosmology()
result = generate_halo_histories(
    n_tracks=100,
    z_final=6.0,
    Mh_final=1e11,
    cosmology=cosmology,
)
sfr_tracks = compute_sfr_from_tracks(
    result.tracks,
    cosmology=cosmology,
    enable_time_delay=True,
)
```

## `auroralf.chemistry.compute_regulator_metallicity()`

导入：

```python
from auroralf.chemistry import RegulatorMetallicityParameters, compute_regulator_metallicity
```

该 backend 实现沿固定 MAH/SFR 历史的 algebraic gas-regulator 金属闭合。SFR 不由金属反馈重算，而是沿用
`auroralf.sfr.compute_sfr_from_tracks()` 给出的
`SFR = f_star(Mh,z) fb dMh/dt / 1e9`。

核心量：

- `Mstar(t) = (1 - R) integral SFR(t') dt'`
- `Mgas(Mh,z) = fres(Mh,z) fb Mh`
- `Zgas = Z0 + y / [1 + Mgas/Mstar + lambda_Z/(1 - R)]`

`RegulatorMetallicityParameters` 常用字段：

- `gas_fraction_norm`、`gas_fraction_mass_slope`、`gas_fraction_redshift_slope`
  控制 `fres(Mh,z) = Mgas / (fb Mh)`；这是 halo baryons 中进入可稀释金属的冷气体库比例，不是星系内部 gas fraction
  `Mgas / (Mgas + Mstar)`。字段名保留 `gas_fraction_*` 是为了兼容当前 API/CLI
- `metal_yield`
  每形成单位 stellar mass 的 metal yield
- `returned_fraction`
  即时返回比例 `R`
- `inflow_metallicity_zsun`
  inflow/pre-enrichment 项 `Z0`，单位 `Z/Zsun`
- `metal_loading_norm`、`metal_loading_mass_slope`、`metal_loading_redshift_slope`
  有效金属损失项 `lambda_Z(Mh,z)`；只进入金属丰度分母，不改变 SFR
- `metallicity_scatter_dex`
  可选 lognormal metallicity scatter，单位 dex

输出：

- `RegulatorMetallicityResult`
  包含 `stellar_mass_msun_grid`、`gas_mass_grid`、`gas_fraction_grid`、`metal_loading_grid`、
  `gas_metallicity_zsun_grid`、`birth_metallicity_zsun_grid` 和 `metal_mass_grid`

说明：

- 这是诊断性金属闭合，不包含显式 metal production/mixing 时间延迟，也不把 `lambda_Z` 反馈到 SFR
- 当前实现保留把本步 regulator `Zgas` 作为 source-time `Zbirth` 提供给 archived IMF gate 的诊断路径；production 不启用该 gate
- 若要和观测常用的星系气体分数比较，应使用
  `fgas_gal = fres / (fres + Mstar / (fb Mh))`，不要把 `fres=0.02`
  误读成 2% galaxy gas fraction
- 参数扫描脚本见 `scripts/analysis/sweep_regulator_metallicity.py`
- 红移演化诊断脚本见 `scripts/analysis/plot_regulator_metallicity_redshift_evolution.py`

## `auroralf.ssp.load_uv1600_table()`

导入：

```python
from auroralf.ssp import load_uv1600_table
```

输入：

- `file_path`
  SSP 光谱文件路径，例如 `external_data/ssp_spectra/bpass_byrne23_imf135_300/BASEL/spectra-bin-imf135_300.BASEL.z001.a+00.dat`
  或 `external_data/ssp_spectra/bpass_v2_2_1/imf100_300/SSP_Spectra_BPASSv2.2.1_bin-imf100_300.hdf5`
- `wavelength_a`
  目标波长，单位 `Angstrom`，默认 `1600.0`
- `metallicity`
  仅对 `HDF5` SSP 文件生效，单位是线性 `Z/Zsun`；必须精确匹配文件中的离散金属丰度选项，例如 `0.05`、`0.1`、`0.2`

输出：

- `ages_myr`
  SSP 年龄数组，单位 `Myr`
- `luminosity_per_msun`
  对应波长下的单位恒星质量光度，单位 `erg/s/Hz/Msun`

说明：

- 内部带缓存；同一个文件和波长组合只会实际读取一次
- 对 `.dat` 文件保持现有读取行为不变
- 对 `.hdf5` 文件会直接使用文件内年龄网格，并按 `metallicity=Z/Zsun` 精确选择金属丰度 bin
- 当前这批 BPASS `HDF5` 模板中的 `spectra` 已经是按单位恒星质量归一化的 `Lnu/Msun`
- 读取 `.hdf5` 文件需要 `h5py`

## `auroralf.ssp.interpolate_uv1600_luminosity_per_msun()`

导入：

```python
from auroralf.ssp import interpolate_uv1600_luminosity_per_msun
```

输入：

- `time_myr`
  需要查询的 SSP 年龄，单位 `Myr`；支持标量或 `numpy.ndarray`
- `file_path`
  SSP 光谱文件路径
- `wavelength_a`
  目标波长，单位 `Angstrom`，默认 `1600.0`
- `metallicity`
  仅对 `HDF5` SSP 文件生效，单位是线性 `Z/Zsun`；必须精确匹配文件中的离散金属丰度选项

输出：

- 插值后的单位恒星质量光度，单位 `erg/s/Hz/Msun`
  输入是标量时返回 `float`，输入是数组时返回 `numpy.ndarray`

说明：

- 采用对 `log10(age)` 的一维线性插值
- 超出表格年龄范围时会夹到边界值

最小调用：

```python
from auroralf.ssp import interpolate_uv1600_luminosity_per_msun

lum_1600 = interpolate_uv1600_luminosity_per_msun(
    time_myr=10.0,
    file_path="external_data/ssp_spectra/bpass_byrne23_imf135_300/BASEL/spectra-bin-imf135_300.BASEL.z001.a+00.dat",
)
```

HDF5 示例：

```python
from auroralf.ssp import load_uv1600_table

ages_myr, luv_per_msun = load_uv1600_table(
    file_path="external_data/ssp_spectra/bpass_v2_2_1/imf100_300/SSP_Spectra_BPASSv2.2.1_bin-imf100_300.hdf5",
    metallicity=0.05,
)
```

## `auroralf.ssp.compute_halo_uv_luminosity()`

导入：

```python
from auroralf.ssp import compute_halo_uv_luminosity
```

输入：

- `t_obs`
  观测时刻；需与 `t_history`、`ssp_age_grid`、`t_z50` 使用同一时间单位
- `t_history`
  halo 历史时间数组；函数内部会兼容非升序输入
- `mh_history`
  与 `t_history` 对应的 halo mass 历史
- `sfr_history`
  与 `t_history` 对应的恒星形成率历史，单位 `Msun/yr`
- `ssp_age_grid`
  SSP 年龄网格；需与 `t_history` 使用同一时间单位
- `ssp_luv_grid`
  SSP UV 光度核，单位 `erg/s/Hz/Msun`
- `M_min`
  最小 halo 质量阈值
- `t_z50`
  `z=50` 对应的宇宙时间
- `time_unit_in_years`
  时间单位换算到 `yr` 的系数；若时间数组使用 `Gyr`，默认 `1e9`
- `return_details`
  是否额外返回卷积起点和实际积分网格等调试信息

输出：

- 默认返回 `L_uv_halo`
  观测时刻 halo 的总 UV 光度，单位 `erg/s/Hz`
- 当 `return_details=True` 时返回 `dict`
  包含：
  - `L_uv_halo`
  - `ti`
  - `mask_used`
  - `age_used`
  - `t_used`
  - `kernel_used`
  - `integrand_used`
  - `t_cross_Mmin`

说明：

- 卷积公式为 `L_uv = ∫ SFR(t') * L_uv^SSP(t_obs - t') dt'`
- 卷积下限使用 `ti = max(t_z50, t_cross_Mmin)`
- `t_cross_Mmin` 在 `Mh(t)` 穿过 `M_min` 时用线性插值求出
- 若 `ti` 早于 `t_history` 的首个采样点，实际积分会从首个可用历史点开始
- `dt` 的年单位换算已显式通过 `time_unit_in_years` 处理
- SSP 核采用与现有 `auroralf.ssp` 一致的 `log10(age)` 插值风格
- 当年龄小于 SSP 最小年龄时取最小年龄值；大于最大年龄时返回 `0`
- 若 `load_uv1600_table()` 返回的是 `Myr` 年龄网格，而 `auroralf/mah` 和 `auroralf/sfr` 历史是 `Gyr`，请先做 `ssp_age_grid_gyr = ages_myr / 1e3`

最小调用：

```python
from auroralf.mah import Cosmology, generate_halo_histories
from auroralf.sfr import compute_sfr_from_tracks
from auroralf.ssp import compute_halo_uv_luminosity, load_uv1600_table

cosmology = Cosmology()
histories = generate_halo_histories(
    n_tracks=1,
    z_final=6.0,
    Mh_final=1e11,
    cosmology=cosmology,
)
sfr_tracks = compute_sfr_from_tracks(histories.tracks, cosmology=cosmology)

ages_myr, luv_per_msun = load_uv1600_table(
    "external_data/ssp_spectra/bpass_byrne23_imf135_300/BASEL/spectra-bin-imf135_300.BASEL.z001.a+00.dat"
)
ssp_age_grid_gyr = ages_myr / 1e3

halo_mask = sfr_tracks["halo_id"] == 0
L_uv = compute_halo_uv_luminosity(
    t_obs=float(sfr_tracks["t_gyr"][halo_mask][-1]),
    t_history=sfr_tracks["t_gyr"][halo_mask],
    mh_history=sfr_tracks["Mh"][halo_mask],
    sfr_history=sfr_tracks["SFR"][halo_mask],
    ssp_age_grid=ssp_age_grid_gyr,
    ssp_luv_grid=luv_per_msun,
    M_min=1e8,
    t_z50=float(sfr_tracks["t_gyr"][halo_mask][0]),
)
```

## `auroralf.uvlf.run_halo_uv_pipeline()`

导入：

```python
from auroralf.uvlf import run_halo_uv_pipeline
```

输入：

- `n_tracks`
  要生成并卷积的 halo 条数
- `z_final`
  观测红移
- `Mh_final`
  在 `z_final` 处的最终 halo mass
- `z_start_max`
  回溯的最高红移，默认 `50.0`
- `n_grid`
  redshift grid 点数，默认 `240`
- `ssp_file`
  canonical Pop II SSP 光谱文件路径；默认使用 `external_data/ssp_spectra/bpass_byrne23_imf135_300/BASEL/spectra-bin-imf135_300.BASEL.z001.a+00.dat`
- `topheavy_ssp_file`
  mild top-heavy Pop II SSP 光谱文件路径；默认使用 `external_data/ssp_spectra/bpass_v2_2_1/imf100_300/SSP_Spectra_BPASSv2.2.1_bin-imf100_300.hdf5`
- `topheavy_ssp_metallicity`
  读取 HDF5 top-heavy SSP 时使用的金属丰度，单位为 `Z/Zsun`；默认 `0.05`
- `imf_mode`
  Pop II IMF 模式，支持：
  - `"canonical"`：所有源时刻都使用 canonical Pop II SSP
  - `"z10_mild_topheavy"`：archived historical mode
  - `"mah_burst_mild_topheavy"`：archived historical mode；满足 growth-time 与 birth-metallicity 条件时使用 mild top-heavy SSP
- `imf_transition_parameters`
  archived `auroralf.uvlf.imf.IMFTransitionParameters`，默认
  `source_redshift_gate_enabled=False`、`growth_time_threshold_myr=50.0`、
  `metallicity_topheavy_max_zsun=0.05`。`z_topheavy_min=10.0` 只在显式启用
  历史 source-redshift gate 时使用；若 `metallicity_topheavy_max_zsun` 设为
  `None`，则关闭金属丰度 gate
- `cosmology`
  keyword-only 必填的 `auroralf.mah.Cosmology`；同一对象会传给 MAH、SFR、
  Pop III/chemistry 与 UV convolution 路径
- `random_seeds`
  keyword-only 必填的 `auroralf.seeding.PipelineRandomSeeds`；其中 `mah`、
  `metallicity`、`burst` 分别且只传给对应随机过程，不允许 `None` 或隐式 entropy fallback
- `sampler`
  `auroralf.mah` 参数抽样方式，只支持 `"mcbride"` 和
  `"gaussian_approximation"`，默认 `"mcbride"`
- `mah_backend`
  MAH 来源，默认 `"mcbride"`。可选 `"tng"` 或 `"thesan"` 时只替代 halo assembly
  history；SFR、IMF gate、SSP、UV 卷积和 UVLF normalization 仍使用 AuroraLF 当前流程
- `tng_mah_cache_path`
  `mah_backend="tng"` 时必填，指向 TNG compact MAH cache 文件或目录
- `tng_mass_bin_width_dex` / `tng_min_candidates` / `tng_smoothing_myr`
  TNG cache 中按 `log10(Mh_final)` 近邻抽取 MAH shape 的 bin 半宽、最少候选数和平滑时间尺度
- `thesan_mah_cache_path`
  `mah_backend="thesan"` 时必填，指向 THESAN-dark-1 compact MAH cache 文件或目录
- `thesan_mass_bin_width_dex` / `thesan_min_candidates` / `thesan_smoothing_myr`
  THESAN cache 中按 `log10(Mh_final)` 近邻抽取 MAH shape 的 bin 半宽、最少候选数和平滑时间尺度

TNG/THESAN cache 必须包含 finite、物理有效的 `hubble`、`omega_m`、`omega_b`
attributes，并与传入 `cosmology` 严格匹配；不匹配会显式报错，不能把不同宇宙学的
时间网格、质量换算与 HMF 权重混在同一次运行中。
- `enable_time_delay`
  是否在 `auroralf.sfr` 计算中启用基于 dynamical time 的 extended-burst 延迟核，默认 `False`
- `workers`
  保留的接口参数；当前实现中 `run_halo_uv_pipeline()` 内部 UV 卷积按串行执行
- `mzr_metallicity_parameters`
  可选 `auroralf.chemistry.MZRBirthMetallicityParameters`；提供时由累计 surviving stellar mass 和经验 MZR
  直接给出 `Z_birth(t)`；该低层接口仅保留用于历史复现与诊断
- `regulator_metallicity_parameters`
  可选 `auroralf.chemistry.RegulatorMetallicityParameters`；提供时由累计 `Mstar`、halo baryon gas reservoir
  和有效 `lambda_Z` 给出 `Z_birth(t)` 与 `Z_gas(t)`；该低层接口仅保留用于历史复现与诊断
- `burst_scatter_dex`
  archived 低层参数；对源时刻 SFR 施加 lognormal burst scatter 的标准差，单位 dex
- `burst_scatter_timescale_myr`
  burst scatter 在同一 halo 内保持相关的时间尺度，单位 `Myr`；默认 `20.0`
- `burst_scatter_preserve_mean`
  是否对每条 halo 的 burst 后 SFR 逐条归一化，使 `integral SFR dt` 与 no-burst 历史一致；默认 `True`。
  关闭后仅使用原始 lognormal multiplier，不保证单条 halo 的形成恒星质量守恒

输出：

- `HaloUVPipelineResult`

## `HaloUVPipelineResult`

字段：

- `histories`
  `auroralf.mah.generate_halo_histories()` 返回的原始 `HaloHistoryResult`
- `sfr_tracks`
  `auroralf.sfr.compute_sfr_from_tracks()` 输出的扁平表格
- `uv_luminosities`
  每个 halo 在 `z_final` 的总 UV 光度，单位 `erg/s/Hz`
- `uv_luminosities_canonical`
  canonical Pop II SSP 对总 UV 光度的分量
- `uv_luminosities_topheavy`
  mild top-heavy Pop II SSP 对总 UV 光度的分量
- `redshift_grid`
  这次计算使用的 redshift grid
- `floor_mass`
  从有效历史点反推出的有效 `M_min(z)` 下限，可直接用于画图
- `active_grid`
  每个 halo 每个时间步是否仍处于有效区间
- `imf_topheavy_source_grid`
  每个 halo 每个源时刻是否实际使用 mild top-heavy SSP kernel；若启用金属 gate，该字段已经过
  `birth_metallicity_zsun_grid <= metallicity_topheavy_max_zsun` 筛选
- `gas_metallicity_zsun_grid`
  若启用 regulator backend，返回每个 halo 每个时间步的 gas metallicity，单位 `Z/Zsun`
- `birth_metallicity_zsun_grid`
  若启用 MZR 或 regulator backend，返回每个 halo 每个时间步的 birth metallicity，单位 `Z/Zsun`
- `metal_mass_grid`
  若启用金属演化，返回诊断 metal mass，单位 `Msun`
- `gas_mass_grid`
  若启用金属演化，返回诊断 gas reservoir mass，单位 `Msun`
- `metadata`
  包含 `n_tracks`、`steps_per_halo`、`workers`、`canonical_ssp_file`、`topheavy_ssp_file`、`imf_mode`、`topheavy_source_fraction`、`topheavy_candidate_source_fraction`、`metallicity_source`、`mzr_metallicity_enabled`、`regulator_metallicity_enabled`、`final_gas_metallicity_zsun_median`、`birth_metallicity_zsun_starforming_median`、`burst_scatter_dex`、`burst_scatter_mass_conserving`、`burst_sfr_multiplier_median`、`enable_time_delay` 和各阶段耗时

说明：

- 这个函数封装了完整主流程：`auroralf.mah -> auroralf.sfr -> auroralf.ssp UV convolution`
- `auroralf.mah` 部分使用默认 `M_min` 时，atomic-cooling threshold 会显式使用
  同一个 `cosmology` 的 `h` 与 `omega_m`
- UV 卷积只对 `active_flag=True` 的有效历史段进行
- production 只使用 canonical Pop II SSP；archived top-heavy 路径按 `imf_mode` 在源时刻选择 SSP kernel
- archived mild top-heavy 路径默认还要求本步成星前 `Z_birth <= 0.05 Zsun`
- `mzr_metallicity_parameters` 和 `regulator_metallicity_parameters` 二选一；
  它们都只向 IMF selector 提供 source-time `Z_birth`，regulator 额外输出 `Z_gas`、`Mgas` 和 `metal_mass` 诊断
- 可选 burst scatter 使用
  `SFR_burst(t)=SFR_smooth(t) 10^Delta`，其中
  `Delta ~ Normal(-0.5 ln(10) sigma_burst^2, sigma_burst)`；该均值位移保证
  `E[10^Delta]=1`，因此 scatter 默认不整体抬高平均 SFR
- `load_uv1600_table()` 读出的 SSP 年龄网格会自动从 `Myr` 转成 `Gyr` 后再参与卷积

最小调用：

```python
from auroralf.mah import Cosmology
from auroralf.uvlf import run_halo_uv_pipeline

cosmology = Cosmology()
result = run_halo_uv_pipeline(
    n_tracks=10000,
    z_final=6.0,
    Mh_final=1e12,
    cosmology=cosmology,
    workers=32,
)

print(result.uv_luminosities.shape)
print(result.metadata["timing_seconds"])
```

## `auroralf.uvlf.sample_uvlf_from_hmf()`

导入：

```python
from auroralf.uvlf import sample_uvlf_from_hmf
```

输入：

- `z_obs`
  观测红移
- `cosmology`
  keyword-only 必填的 `auroralf.mah.Cosmology`；HMF 权重和每个内层 pipeline
  worker 都使用该对象
- `N_mass`
  外层 Monte Carlo 抽取的 halo 终质量个数，默认 `3000`
- `n_tracks`
  每个质量点内层生成的 luminosity realization 个数，默认 `1000`
- `base_seed`
  keyword-only 必填的非负 Python `int`，范围为 `0 <= base_seed <= 2**64-1`。HMF mass draw 和每个质量点的 MAH、
  metallicity、burst seed 由稳定 `SeedSequence` 派生；key 含 redshift 和 mass index，
  不含 IMF mode 或遍历顺序
- `quantity`
  统计对象，支持 `"Muv"` 和 `"luminosity"`；默认 `"Muv"`
- `bins`
  histogram 的 bin 数或 bin edges
- `logM_min`
  外层均匀抽样的最低 `log10 Mh`，默认 `9`
- `logM_max`
  外层均匀抽样的最高 `log10 Mh`，默认 `13`
- `z_start_max`
  内层 `auroralf.mah` 回溯的最高红移，默认 `50.0`
- `n_grid`
  内层 `auroralf/mah` 和 `auroralf/sfr` 使用的 redshift grid 点数，默认 `240`
- `sampler`
  `auroralf.mah` 参数抽样方式，只支持 `"mcbride"` 和
  `"gaussian_approximation"`，默认 `"mcbride"`
- `mah_backend`
  MAH 来源，默认 `"mcbride"`；`"tng"` 和 `"thesan"` 分别从对应 compact MAH cache
  有放回抽样条件分布 `P(MAH | Mh, z)`，不改变外层 Reed07 HMF 权重
- `tng_mah_cache_path` / `tng_mass_bin_width_dex` / `tng_min_candidates` / `tng_smoothing_myr`
  透传给 `run_halo_uv_pipeline()` 的 TNG MAH backend 参数
- `thesan_mah_cache_path` / `thesan_mass_bin_width_dex` / `thesan_min_candidates` / `thesan_smoothing_myr`
  透传给 `run_halo_uv_pipeline()` 的 THESAN MAH backend 参数
- `enable_time_delay`
  是否在 `auroralf.sfr` 中启用时间延迟，默认 `False`
- `pipeline_workers`
  外层 `N_mass` 质量点采样使用的并行 worker 数
- `ssp_file`
  canonical Pop II SSP 文件路径；默认使用 `external_data/ssp_spectra/bpass_byrne23_imf135_300/BASEL/spectra-bin-imf135_300.BASEL.z001.a+00.dat`
- `topheavy_ssp_file`
  mild top-heavy Pop II SSP 文件路径；仅非 canonical IMF 模式实际读取
- `topheavy_ssp_metallicity`
  HDF5 mild top-heavy SSP 的金属丰度，单位为 `Z/Zsun`；默认 `0.05`
- `enable_popiii` / `popiii_sfr_parameters` / `popiii_ssp_file`
  Pop III 开关、`auroralf.sfr.PopIIISFRParameters` 和 SSP 路径。
  `popiii_sfr_parameters.lw_background_j21` 是 HMF Pop III 质量下限、stellar-channel
  分类、内层 pipeline 和 metadata 的唯一 LW background 来源；接口不再接受独立的
  `lw_background_j21` 参数
- `imf_mode`
  同 `run_halo_uv_pipeline()`，默认 `"canonical"`
- `imf_transition_parameters`
  mild top-heavy IMF 的源时刻触发参数；默认包含 `metallicity_topheavy_max_zsun=0.05`
- `progress_path`
  可选进度文件路径；若提供，会把外层 `N_mass` 循环进度持续写入该 txt 文件
- `mass_function_model`
  外层 halo mass function 权重模型；当前生产接口只支持 `"hmf_reed07"`，使用 `hmf` 包中的 Reed07 fitting function。旧的 `"massfunc_st"` 和 Watson13 分支已禁用。
- `mzr_metallicity_parameters`
  可选 `auroralf.chemistry.MZRBirthMetallicityParameters`；会透传给每个质量点的 `run_halo_uv_pipeline()`
- `regulator_metallicity_parameters`
  可选 `auroralf.chemistry.RegulatorMetallicityParameters`；会透传给每个质量点的 `run_halo_uv_pipeline()`，
  与 MZR backend 二选一
- `burst_scatter_dex`
  透传给每个质量点的 SFR burst scatter 标准差，单位 dex；默认 `0.0`
- `burst_scatter_timescale_myr`
  burst scatter 时间相关尺度，单位 `Myr`；默认 `20.0`
- `burst_scatter_preserve_mean`
  是否对每条 halo 的 burst 后 SFR 逐条归一化，使 `integral SFR dt` 与 no-burst 历史一致；默认 `True`

输出：

- `UVLFSamplingResult`

## `UVLFSamplingResult`

字段：

- `samples`
  样本表，包含：
  - `logMh`
  - `Mh`
  - `mass_weight`
  - `track_index`
  - `luminosity`
  - `topheavy_light_fraction`
  - `stellar_channel`
    基于 Pop III H2/LW 下限和 atomic-cooling 上限的通道标记：
    低于 Pop III 下限为 `below_popiii_min`，介于 Pop III 下限和 atomic-cooling
    threshold 之间为 `popiii`，等于或高于 atomic-cooling threshold 为 `popii`
  - `atomic_cooling_mass_msun`
  - `popiii_minimum_mass_msun`
  - `Muv`
  - `sample_weight`
- `auroralf.uvlf`
  UVLF histogram 结果，包含：
  - `quantity`
  - `bin_edges`
  - `bin_centers`
  - `bin_width`
  - `raw_counts`
  - `weighted_counts`
  - `weight_squared_counts`
  - `weighted_count_sigma`
  - `effective_counts`
  - `phi`
  - `phi_sigma`
- `metadata`
  运行参数、耗时信息、top-heavy 诊断和可选金属演化诊断；启用金属演化时包含 `final_gas_metallicity_zsun_median_by_mass` 和 `birth_metallicity_zsun_starforming_median_by_mass`

说明：

- 外层在 `log10 Mh in [9, 13]` 上均匀抽样
- HMF 采样层会记录 Pop III minihalo 分流下限和 atomic-cooling 上限：
  `popiii_minimum_mass_msun = 3.3e7 * (1+z_obs)**(-1.5) *
  (1 + 2.0 * popiii_sfr_parameters.lw_background_j21**0.6)`。
  默认 `popiii_sfr_parameters.lw_background_j21=0.0` 时，该式退化到无 LW
  feedback 的 H2 cooling floor。
  `atomic_cooling_mass_msun` 由项目内的 Barkana--Loeb virial-temperature
  反解和 Bryan--Norman collapse-overdensity fit 直接计算，不调用外部
  `massfunc` package。
  `samples["stellar_channel"]` 和 `metadata["stellar_channel_by_mass"]`
  把 `Mh < M_PopIII,min` 标记为 `below_popiii_min`，
  `M_PopIII,min <= Mh < M_atomic` 标记为 `popiii`，
  `Mh >= M_atomic` 标记为 `popii`；
  当前主干仍只实现 Pop II UV luminosity pipeline，后续 Pop III 物理通道应接这个 mask
- 外层权重默认使用 `hmf` 包的 Reed07 halo mass function：
  - `dn/dlogM = M ln(10) dn/dM`
- `hmf` 的质量单位从 `Msun/h` 转成项目内部使用的 `Msun`，`dn/dM` 从 `h^4 Mpc^-3 Msun^-1` 转成 `Mpc^-3 Msun^-1`
- 若传入旧的 `mass_function_model="massfunc_st"` 或 `"hmf_watson13_fof"`，接口会显式报错，避免误用历史分支
- 每个质量点的总权重会平均分配给其 `n_tracks` 个 luminosity realization
- 内层条件采样器直接复用 `auroralf.uvlf.run_halo_uv_pipeline()`
- 当前并行层级放在外层 `N_mass` 循环；`run_halo_uv_pipeline()` 内部 UV 卷积保持串行，避免嵌套进程池
- 若设置 `progress_path`，外层 `N_mass` 进度条会实时写入文本文件
- 非 canonical IMF 模式是 archived feature；v2 配置必须显式设置
  `enable_archived_imf_gate = true` 才能用于历史复现，并仍需提供 regulator 或 MZR
  birth-metallicity backend。production 不使用这一入口
- burst scatter 是 archived feature；v2 配置必须同时设置
  `enable_archived_burst_scatter = true` 和 `burst_scatter_dex > 0`
- MZR/regulator 是 archived feature；v2 配置必须设置
  `enable_archived_metallicity = true` 后才能选择非 `none` backend
- 历史 burst 复现中，金属演化和 UV 卷积使用同一条 burst 后的 SFR 历史；
  相同 `base_seed`、redshift 和 mass index 在所有 IMF mode 中共享同一组 paired realization
- 默认 `burst_scatter_preserve_mean=True` 时，每条 halo 的 burst 历史会做 mass-conserving 归一化：
  `SFR_burst(t) = SFR_0(t) B(t) integral SFR_0 dt / integral SFR_0(t) B(t) dt`

最小调用：

```python
from auroralf.mah import Cosmology
from auroralf.uvlf import sample_uvlf_from_hmf

cosmology = Cosmology()
result = sample_uvlf_from_hmf(
    z_obs=6.0,
    cosmology=cosmology,
    base_seed=42,
    N_mass=3000,
    n_tracks=1000,
    pipeline_workers=32,
)

print(result.samples["Muv"].shape)
print(result.uvlf["phi"])
```

## v2 production UVLF workflow

`configs/uvlf/production.toml` is the single production configuration. It
contains the cosmology, MAH backend, time-delay model, canonical Pop II
population, HMF sampling, worker count, and
absolute output target after TOML-relative path resolution. Unknown keys,
invalid units/ranges, missing SSP/cache files, and archived-feature use without
explicit opt-in fail before sampling starts.

The strict v2.2 configuration archives the source-time IMF gate, burst scatter,
and metallicity models:
`imf_modes = ["canonical"]` and `enable_archived_imf_gate = false` are the
production IMF settings; `burst_scatter_dex = 0` and
`metallicity_source = "none"` are the production SFR settings. Archived modes
are rejected unless their corresponding opt-in is explicitly enabled for
historical reproduction.

Render and inspect the exact scheduler command before submission:

```bash
PYTHONPATH=. .venv/bin/python scripts/submit/submit_uvlf_v2.py \
  --config configs/uvlf/production.toml \
  --mem 64G --time 12:00:00 --dry-run
```

Submit the same command without `--dry-run`. `sampling.workers` must not exceed
the allocated CPUs. The runner refuses to execute outside a SLURM allocation;
large production calculations must not run on a login node.

The output is one versioned HDF5 v2 artifact plus a completion marker. Results
are first published as atomic `(redshift, IMF mode)` shards, then strictly
validated and atomically merged. Provenance records the canonical config hash,
Git revision and dirty state, seed namespace, checksummed scientific inputs,
and a checksummed immutable SLURM execution record containing resources,
command, job ID, logs, exit code, and final validation.

Existing outputs are never silently replaced:

- default policy: fail if final or shard artifacts already exist;
- `--resume`: reuse only exact schema/config/provenance-compatible shards;
- `--overwrite`: deliberately replace the requested run's artifact set;
- an HDF5 file without its marker, or a marker without its HDF5 file, is an
  explicit incomplete-artifact error.

The old `run_uvlf_compare_imf_no_delay_all_z.py` and
`submit_uvlf_imf_compare.py` entry points are migration-only shims. Scientific
options that were formerly CLI flags now belong in the typed TOML config. In
historical reproduction, non-canonical modes with a non-`None`
birth-metallicity gate require both `enable_archived_metallicity = true` and
`metallicity_source = "regulator"` or `"mzr"`. Burst scatter requires
`enable_archived_burst_scatter = true` before its archived parameters can be
used under `[star_formation]`.

Legacy `.npz` UVLF products can be migrated with
`scripts/data/convert_uvlf_npz_to_v2_hdf5.py`. New production runs should not
write or consume the legacy production `.npz` format.

## `auroralf.uvlf.compute_dust_attenuated_uvlf()`

导入：

```python
from auroralf.uvlf import compute_dust_attenuated_uvlf
```

输入：

- `intrinsic_muv`
  intrinsic UVLF 的绝对星等网格
- `intrinsic_phi`
  intrinsic UVLF，单位通常为 `Mpc^-3 mag^-1`
- `z`
  观测红移
- `muv_obs`
  输出时使用的 observed magnitude 网格；未提供时默认使用 `intrinsic_muv`
- `c0`, `c1`, `m0`
  尘埃修正公式
  `A_UV = max(c1 + c0 * beta, 0)` 和 `beta = beta0 + dbeta * (M_UV^obs - m0)` 中的系数
- `clip_to_bounds`
  是否把映射后的 intrinsic magnitude 截断到输入网格边界内，默认 `False`
- `match_faint_end_after_intersection`
  保留的兼容接口参数；当前实现中不再使用旧的交点拼接逻辑
- `insert_transition_point`
  保留的兼容接口参数；当前实现中不再使用旧的交点插点逻辑

输出：

- 返回一个字典，常用字段包括：
  - `Muv_obs`
  - `Muv_intrinsic`
  - `A_uv`
  - `dMuv_dMuv_obs`
  - `phi_nodust_obs`
  - `phi_intrinsic_interp`
  - `phi_obs`
  - `phi_obs_eval`
  - `transition_index`

说明：

- 先按公式计算
  `phi_obs_raw(M_UV^obs) = phi_int(M_UV) * dM_UV / dM_UV^obs`
- 当前最终返回的 dust UVLF 采用物理裁剪：
  `phi_obs = min(phi_obs_raw, phi_nodust_obs)`
- 因此最终的 dust 曲线不会高于 no-dust 曲线
- `phi_obs_eval` 保留未经裁剪的原始 dust 结果，便于调试

最小调用：

```python
from auroralf.mah import Cosmology
from auroralf.uvlf import sample_uvlf_from_hmf, compute_dust_attenuated_uvlf
import numpy as np

cosmology = Cosmology()
result = sample_uvlf_from_hmf(
    z_obs=6.0,
    cosmology=cosmology,
    N_mass=3000,
    n_tracks=1000,
    bins=np.linspace(-28.0, -10.0, 21),
    pipeline_workers=32,
)

dust_result = compute_dust_attenuated_uvlf(
    intrinsic_muv=result.uvlf["bin_centers"],
    intrinsic_phi=result.uvlf["phi"],
    z=6.0,
    muv_obs=np.linspace(-28.0, -10.0, 400),
)

print(dust_result["phi_obs"])
```

## `auroralf.uvlf` 尘埃修正辅助函数

导入：

```python
from auroralf.uvlf import (
    intrinsic_muv_from_observed,
    intrinsic_muv_jacobian,
    uv_continuum_slope_beta,
    uv_dust_attenuation,
)
```

说明：

- `uv_continuum_slope_beta(muv_obs, z)`
  返回 Bouwens 型 UV continuum slope `beta`
- `uv_dust_attenuation(muv_obs, z, c0=2.10, c1=4.85, m0=-19.5)`
  返回 `A_UV`
- `intrinsic_muv_from_observed(muv_obs, z, ...)`
  返回 `M_UV = M_UV^obs - A_UV`
- `intrinsic_muv_jacobian(muv_obs, z, ...)`
  返回 `dM_UV / dM_UV^obs`

最小调用：

```python
from auroralf.uvlf import uv_dust_attenuation, intrinsic_muv_from_observed

muv_obs = [-22.0, -20.0, -18.0]
auv = uv_dust_attenuation(muv_obs, z=6.0)
muv_intrinsic = intrinsic_muv_from_observed(muv_obs, z=6.0)
```

## `auroralf.uvlf.compute_reed07_halo_mass_function_dndm()`
导入：

```python
from auroralf.uvlf import compute_reed07_halo_mass_function_dndm
```

输入：

- `halo_mass_msun`
  halo mass；支持标量或 `numpy.ndarray`
- `z_obs`
  红移
- `cosmology`
  keyword-only 必填的 `auroralf.mah.Cosmology`；其 `H0`、`omega_m` 与
  `omega_b` 会直接传给 `hmf.MassFunction`

输出：

- `dndm`
  Reed07 halo mass function `dn/dM`，单位为 `Mpc^-3 Msun^-1`

说明：

- 该接口是项目当前唯一的 HMF 生产接口
- 底层调用 `hmf.MassFunction(hmf_model="Reed07")`
- 质量从 `hmf` 的 `Msun/h` 转为项目内部的 `Msun`
- `dn/dM` 从 `h^4 Mpc^-3 Msun^-1` 转为 `Mpc^-3 Msun^-1`

最小调用：

```python
import numpy as np
from auroralf.mah import Cosmology
from auroralf.uvlf import compute_reed07_halo_mass_function_dndm

cosmology = Cosmology()
masses = np.logspace(8, 12, 100)
dndm = compute_reed07_halo_mass_function_dndm(
    masses,
    6.0,
    cosmology=cosmology,
)
```
