# realUVLF

## `mah.generate_halo_histories()`

导入：

```python
from mah import generate_halo_histories
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
  最低质量阈值；默认 `None` 时使用 `massfunc.SFRD().M_vir(mu=0.61, Tvir=1e4, z)`，也可以传标量、与红移网格同长度的数组，或 `M_min(z)` 形式的可调用对象。若研究 minihalo 成星，可直接传 `sfr.minihalo_mass_floor`，把 H(_2) 冷却下限 `M_cool_H2(z)` 作为成星下限质量
- `cosmology`
  `mah.Cosmology`；未提供时使用项目默认宇宙学
- `random_seed`
  随机种子
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
  `beta, gamma` 的抽样方式，支持 `"mcbride"` 和 `"gaussian"`
- `pilot_samples`
  当 `sampler="gaussian"` 时使用的 pilot sample 数目

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
- `dMh_dt`
  halo mass accretion rate
- `active_flag`
  是否仍处于有效区间；当 `Mh < M_min` 后为 `False`
- `termination_flag`
  终止状态标记；当前实现使用：
  - `"active"`
  - `"below_M_min"`
  - `"completed"`

## `sfr.compute_sfr_from_tracks()`

导入：

```python
from sfr import compute_sfr_from_tracks
```

输入：

- `tracks`
  `HaloHistoryResult.tracks` 风格的扁平 `dict[str, np.ndarray]`；至少需要：
  - `halo_id`
  - `step`
  - `z`
  - `t_gyr`
  - `Mh`
  - `dMh_dt`
  若提供 `active_flag`，则 SFR 只会在该轨道被标记为 active 的时间段内计算
- `enable_time_delay`
  是否启用基于 dynamical time 的 extended-burst 延迟核；默认 `False`
- `burst_kappa`
  extended-burst 核的 κ 参数；默认 `0.1`
- `burst_lookback_max_myr`
  延迟卷积最大回看时间，单位 `Myr`；默认 `100.0`
- `model_parameters`
  `SFRModelParameters` 实例；未提供时使用默认参数

输出：

- `dict[str, np.ndarray]`
  保留输入列，并新增：
  - `r_vir`
  - `V_c`
  - `T_vir`
  - `sigma_vbc_rms`
  - `V_cool_H2`
  - `M_cool_H2`
  - `M_atom`
  - `tau_del`
  - `td_burst`
  - `t_src`
  - `Mh_src`
  - `dMh_dt_src`
  - `fstar_src`
  - `fstar_now`
  - `pop2_active_flag`
  - `branch_active_flag`
  - `SFR_pop2`
  - `mdot_burst`
  - `SFR_total`
  - `SFR`

说明：

- `SFR` 单位为 `Msun/yr`
- 当 `enable_time_delay=False` 时，直接用当前时刻的 `Mh` 和 `dMh_dt`
- 当 `enable_time_delay=True` 时，使用
  `g(t-t') \propto (t-t') \exp[-(t-t')/(\kappa t_d)]`
  的 extended-burst 核对
  `fstar(Mh(t')) * dMh_dt(t')`
  做时间卷积
- 原子冷却阈值始终生效，不再提供关闭开关
- 若输入 `tracks` 含有 `active_flag`，则该标记与原子冷却阈值共同决定 `pop2_active_flag/branch_active_flag`
- `tau_del/t_src/Mh_src/dMh_dt_src` 仍保留，作为与旧单一延迟时间口径可对照的诊断量
- `mdot_burst` 仍保留，表示只对 `dMh/dt` 做 kernel 卷积后的诊断量；真正进入 delay-SFR 的是
  `kernel * fstar(Mh) * dMh_dt` 的积分
- 若 `T_vir < 1e4 K`，则 `SFR = 0`
- `sigma_vbc_rms/V_cool_H2/M_cool_H2` 是额外的 minihalo 诊断量，其中
  `V_cool_H2 = sqrt(a^2 + (b v_bc)^2)`，默认采用 `a=3.714 km/s`、`b=4.015` 和
  `v_bc(z)=sigma_bc(z)`

## 项目里 minihalo 的成星下限

项目中可把

```python
from sfr import minihalo_mass_floor
```

返回的 `M_cool_H2(z)` 作为 minihalo 可以成星的下限质量。

默认口径是：

- 采用 Fialkov et al. 的拟合
  `V_cool,H2 = sqrt(a^2 + (b v_bc)^2)`
- 默认参数 `a=3.714 km/s`、`b=4.015`
- 默认取 `v_bc(z)=sigma_bc(z)`
- 再用 Barkana & Loeb 的 virial 关系把 `V_cool,H2` 转成 virial mass

如果要把它真正用于轨道截断或活跃 halo 选择，推荐直接传给 `mah.generate_halo_histories()` 的 `M_min`：

```python
from mah import generate_halo_histories
from sfr import minihalo_mass_floor

result = generate_halo_histories(
    n_tracks=100,
    z_final=10.0,
    Mh_final=1e8,
    M_min=minihalo_mass_floor,
)
```

## `sfr.minihalo_mass_floor()`

导入：

```python
from sfr import minihalo_mass_floor
```

输入：

- `redshift`
  标量或数组红移
- `v_bc_kms`
  可选的 baryon-dark matter streaming velocity，单位 `km/s`；未提供时默认采用
  `sigma_bc(z)`
- `cosmology`
  可选宇宙学；未提供时使用项目默认宇宙学

输出：

- 对应 redshift 上的 H(_2) 冷却 minihalo 下限质量，单位 `Msun`

说明：

- 采用 Fialkov et al. 的拟合
  `V_cool,H2 = sqrt(a^2 + (b v_bc)^2)`
- 再用 Barkana & Loeb 的 virial 标度把 `V_cool,H2` 转成 `M_vir`

最小调用：

```python
from mah import generate_halo_histories
from sfr import compute_sfr_from_tracks

result = generate_halo_histories(n_tracks=100, z_final=6.0, Mh_final=1e11)
sfr_tracks = compute_sfr_from_tracks(result.tracks, enable_time_delay=True)
```

## `ssp.load_uv1600_table()`

导入：

```python
from ssp import load_uv1600_table
```

输入：

- `file_path`
  SSP 光谱文件路径，例如 `data/spectra-bin-imf135_300.BASEL.z001.a+00.dat`
  或 `data/SSP_Spectra_BPASSv2.2.1_bin-imf100_300.hdf5`
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

## `ssp.interpolate_uv1600_luminosity_per_msun()`

导入：

```python
from ssp import interpolate_uv1600_luminosity_per_msun
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
from ssp import interpolate_uv1600_luminosity_per_msun

lum_1600 = interpolate_uv1600_luminosity_per_msun(
    time_myr=10.0,
    file_path="data/spectra-bin-imf135_300.BASEL.z001.a+00.dat",
)
```

HDF5 示例：

```python
from ssp import load_uv1600_table

ages_myr, luv_per_msun = load_uv1600_table(
    file_path="data/SSP_Spectra_BPASSv2.2.1_bin-imf100_300.hdf5",
    metallicity=0.05,
)
```

## `ssp.compute_halo_uv_luminosity()`

导入：

```python
from ssp import compute_halo_uv_luminosity
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
- SSP 核采用与现有 `ssp` 一致的 `log10(age)` 插值风格
- 当年龄小于 SSP 最小年龄时取最小年龄值；大于最大年龄时返回 `0`
- 若 `load_uv1600_table()` 返回的是 `Myr` 年龄网格，而 `mah/sfr` 历史是 `Gyr`，请先做 `ssp_age_grid_gyr = ages_myr / 1e3`

最小调用：

```python
from mah import generate_halo_histories
from sfr import compute_sfr_from_tracks
from ssp import compute_halo_uv_luminosity, load_uv1600_table

histories = generate_halo_histories(n_tracks=1, z_final=6.0, Mh_final=1e11)
sfr_tracks = compute_sfr_from_tracks(histories.tracks)

ages_myr, luv_per_msun = load_uv1600_table(
    "data/spectra-bin-imf135_300.BASEL.z001.a+00.dat"
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

## `uvlf.run_halo_uv_pipeline()`

导入：

```python
from uvlf import run_halo_uv_pipeline
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
  SSP 光谱文件路径；默认使用 `data_save/ssp_uv1600_topheavy_imf100_300_z0005.npz`
- `cosmology`
  `mah.Cosmology`；未提供时使用项目默认宇宙学
- `random_seed`
  随机种子
- `sampler`
  `mah` 参数抽样方式，默认 `"mcbride"`
- `enable_time_delay`
  是否在 `sfr` 计算中启用基于 dynamical time 的 extended-burst 延迟核，默认 `False`
- `workers`
  保留的接口参数；当前实现中 `run_halo_uv_pipeline()` 内部 UV 卷积按串行执行
- `burst_lookback_max_myr`
  延迟卷积最大回看时间，单位 `Myr`；默认 `100.0`
- `ssp_lookback_max_myr`
  SSP UV 卷积最大回看时间，单位 `Myr`
- `sfr_model_parameters`
  `SFRModelParameters` 实例；默认使用 `DEFAULT_SFR_MODEL_PARAMETERS`

输出：

- `HaloUVPipelineResult`

## `HaloUVPipelineResult`

字段：

- `histories`
  `mah.generate_halo_histories()` 返回的原始 `HaloHistoryResult`
- `sfr_tracks`
  `sfr.compute_sfr_from_tracks()` 输出的扁平表格
- `uv_luminosities`
  每个 halo 在 `z_final` 的总 UV 光度，单位 `erg/s/Hz`
- `muv`
  每个 halo 在 `z_final` 的总 UV 星等
- `redshift_grid`
  这次计算使用的 redshift grid
- `floor_mass`
  从有效历史点反推出的有效 `M_min(z)` 下限，可直接用于画图
- `active_grid`
  每个 halo 每个时间步是否仍处于有效区间
- `metadata`
  包含 `n_tracks`、`steps_per_halo`、`workers`、`ssp_file`、`enable_time_delay` 和各阶段耗时

说明：

- 这个函数封装了完整主流程：`mah -> sfr -> SSP UV convolution`
- `mah` 部分使用默认 `M_min`，即 `massfunc.SFRD().M_vir(mu=0.61, Tvir=1e4, z)`
- 若某个质量样本在整条历史上都没有 active step，当前会合法返回全零 UV，
  `metadata["has_active_histories"]` 会记录这一点
- `load_uv1600_table()` 读出的 SSP 年龄网格会自动从 `Myr` 转成 `Gyr` 后再参与卷积

最小调用：

```python
from uvlf import run_halo_uv_pipeline

result = run_halo_uv_pipeline(
    n_tracks=10000,
    z_final=6.0,
    Mh_final=1e12,
    workers=32,
)

print(result.uv_luminosities.shape)
print(result.metadata["timing_seconds"])
```

## `uvlf.sample_uvlf_from_hmf()`

导入：

```python
from uvlf import sample_uvlf_from_hmf
```

输入：

- `z_obs`
  观测红移
- `N_mass`
  外层 Monte Carlo 抽取的 halo 终质量个数，默认 `3000`
- `n_tracks`
  每个质量点内层生成的 luminosity realization 个数，默认 `1000`
- `random_seed`
  随机种子
- `quantity`
  统计对象，支持 `"Muv"` 和 `"luminosity"`；默认 `"Muv"`
- `bins`
  histogram 的 bin 数或 bin edges
- `logM_min`
  外层均匀抽样的最低 `log10 Mh`，默认 `9`
- `logM_max`
  外层均匀抽样的最高 `log10 Mh`，默认 `13`
- `z_start_max`
  内层 `mah` 回溯的最高红移，默认 `50.0`
- `n_grid`
  内层 `mah/sfr` 使用的 redshift grid 点数，默认 `240`
- `sampler`
  `mah` 参数抽样方式，默认 `"mcbride"`
- `enable_time_delay`
  是否在 `sfr` 中启用时间延迟，默认 `False`
- `pipeline_workers`
  外层 `N_mass` 质量点采样使用的并行 worker 数
- `ssp_file`
  SSP 文件路径；默认使用 `data_save/ssp_uv1600_topheavy_imf100_300_z0005.npz`
- `progress_path`
  可选进度文件路径；若提供，会把外层 `N_mass` 循环进度持续写入该 txt 文件
- `sfr_model_parameters`
  `SFRModelParameters` 实例；默认使用 `DEFAULT_SFR_MODEL_PARAMETERS`

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
  - `Muv`
  - `sample_weight`
- `uvlf`
  UVLF histogram 结果，包含：
  - `quantity`
  - `bin_edges`
  - `bin_centers`
  - `bin_width`
  - `weighted_counts`
  - `phi`
- `metadata`
  运行参数和耗时信息

说明：

- 外层在 `log10 Mh in [9, 13]` 上均匀抽样
- 外层权重使用 ST halo mass function：
  - `dn/dlogM = M ln(10) dn/dM`
- 每个质量点的总权重会平均分配给其 `n_tracks` 个 luminosity realization
- 内层条件采样器直接复用 `uvlf.run_halo_uv_pipeline()`
- 当前并行层级放在外层 `N_mass` 循环；`run_halo_uv_pipeline()` 内部 UV 卷积保持串行，避免嵌套进程池
- 若设置 `progress_path`，外层 `N_mass` 进度条会实时写入文本文件

最小调用：

```python
from uvlf import sample_uvlf_from_hmf

result = sample_uvlf_from_hmf(
    z_obs=6.0,
    N_mass=3000,
    n_tracks=1000,
    pipeline_workers=32,
)

print(result.samples["Muv"].shape)
print(result.uvlf["phi"])
```

## `uvlf.compute_dust_attenuated_uvlf()`

导入：

```python
from uvlf import compute_dust_attenuated_uvlf
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
  是否把映射后的 intrinsic magnitude 截断到输入网格边界内，默认 `True`
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
from uvlf import sample_uvlf_from_hmf, compute_dust_attenuated_uvlf
import numpy as np

result = sample_uvlf_from_hmf(
    z_obs=6.0,
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

## `uvlf` 尘埃修正辅助函数

导入：

```python
from uvlf import (
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
from uvlf import uv_dust_attenuation, intrinsic_muv_from_observed

muv_obs = [-22.0, -20.0, -18.0]
auv = uv_dust_attenuation(muv_obs, z=6.0)
muv_intrinsic = intrinsic_muv_from_observed(muv_obs, z=6.0)
```

## `massfunc.Mass_func.dndmst()`
导入：

```python
from massfunc import Mass_func
```

输入：

- `M`
  halo mass；支持标量或 `numpy.ndarray`
- `z`
  红移

输出：

- `dndm_st`
  Sheth-Tormen 质量函数 `dn/dM`，量纲可按 `Mpc^-3 Msun^-1` 理解

说明：

- `Mass_func()` 默认使用包内默认宇宙学参数
- ST 质量函数接口名是 `dndmst(M, z)`
- 若需要减少首次调用开销，可先执行：
  - `sigma2_interpolation_set()`
  - `dsig2dm_interpolation_set()`

最小调用：

```python
import numpy as np
from massfunc import Mass_func

mf = Mass_func()
masses = np.logspace(8, 12, 100)
dndm_st = mf.dndmst(masses, 6.0)
```
