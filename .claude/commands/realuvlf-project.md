# realUVLF 项目指南

项目全称：紫外光度函数 (UV Luminosity Function) — 基于高度函数的半解析模型

## 项目概述

通过 Monte Carlo 方法计算高红移宇宙的紫外光度函数 (UVLF)。核心流程：
1. 生成暗晕增长历史 (MAH)
2. 从暗晕质量增长率计算恒星形成率 (SFR)
3. 通过单星族 (SSP) 模型将 SFR 转换为 UV 光度
4. 结合晕质量函数 (HMF) 抽样得到 UVLF
5. 可选：加入尘埃衰减和 Pop III 星族贡献

## 目录结构

```
realUVLF/
├── uvlf/           # 核心计算模块
│   ├── pipeline.py      # 主流水线：MAH → SFR → UV 光度
│   ├── hmf_sampling.py  # 从 HMF 抽样计算 UVLF
│   ├── dust.py          # 尘埃衰减模型 (UV continuum slope + attenuation)
│   └── uvlf.md          # UVLF 计算方案文档
├── mah/            # 暗晕增长历史生成器
│   ├── generator.py     # generate_halo_histories() 主入口
│   ├── models.py        # MAH 模型参数
│   ├── physics.py       # 物理常数与关系
│   └── sampling.py      # Monte Carlo 采样方法
├── sfr/            # 恒星形成率计算
│   ├── calculator.py    # compute_sfr_from_tracks() 主入口
│   ├── popiii_splitter.py  # Pop III/Pop II 分支逻辑
│   └── sfr.md           # SFR 模型文档
├── ssp/            # 单星族 (SSP) UV 光度
│   ├── uv1600.py        # 1600Å UV 光度表加载与插值
│   └── convolution.py   # SFR 卷积得到 UV 光度
├── popiii/         # Pop III 星族
│   ├── burst.py         # Pop III burst 处理
│   ├── hosts.py         # Pop III 宿主暗晕识别
│   ├── trigger.py       # Pop III 触发条件
│   ├── uv.py            # Pop III UV 光度计算
│   └── models.py        # Pop III 模型参数
├── cosm/           # 宇宙学参数与时间线
├── data/           # 光谱库 (Byrne+23 BPASS 等) + 观测数据 + Schaerer Pop III
├── data_save/      # 可复用的中间结果 (CSV, NPZ, 表格)
├── outputs/        # 临时输出 (日志, 快看图, 诊断)
├── slides/         # 演示文稿 (uvlf_status_deck.tex/.pdf)
│   └── assets/     # 幻灯片用图 (仅保留当前引用的)
├── plots/          # 绑图脚本
├── tests/          # 测试
└── .venv/          # Python 虚拟环境
```

## 核心代码入口

### 完整流水线
```python
from uvlf import run_halo_uv_pipeline, HaloUVPipelineResult
```
`run_halo_uv_pipeline()` 是主入口，执行 MAH → SFR → UV 光度的完整链路。

### HMF 抽样 → UVLF
```python
from uvlf import sample_uvlf_from_hmf, UVLFSamplingResult
```
`sample_uvlf_from_hmf()` 在连续质量空间上从 HMF 抽样，结合 pipeline 得到 UVLF。

### 暗晕历史生成
```python
from mah import generate_halo_histories, Cosmology, HaloHistoryResult
```
- 支持 `mcbride` 和 `gaussian` 两种采样器
- 时间网格支持 `uniform_in_z`, `uniform_in_t`, `custom`
- Minihalo 成星下限: `from sfr import minihalo_mass_floor`

### 恒星形成率
```python
from sfr import compute_sfr_from_tracks, SFRModelParameters, minihalo_mass_floor
```
- 支持时间延迟 (extended-burst kernel convolution)
- 支持 Pop III 分支 (`pop3_enabled=True`)
- Pop III IMF 类型: `"Sal"`, `"logA"`, `"logE"`

### 尘埃衰减
```python
from uvlf import uv_dust_attenuation, uv_continuum_slope_beta, compute_dust_attenuated_uvlf
```
- `A_UV(M_UV_obs, z)` 依赖红移和观测星等
- UV slope: `beta(M_UV, z) = -0.09z - 1.49 + (-0.007z - 0.09)(M_UV - M0)`

## 运行方式

### 顶层脚本
- `run_uvlf_pop3_compare_all_z.py` — Pop III vs Pop II 全红移对比
- `run_uvlf_compare_imf_no_delay_all_z.py` — IMF 对比 (无时间延迟)
- `run_uvlf_parallel_popiii_compare_z10_15.py` — Pop III 并行对比 z=10,15
- `replot_uvlf_imf_no_delay_compare.py` — 重绑 IMF 对比图

### Python 环境
使用项目 `.venv`：
```bash
.venv/bin/python script.py
```
或通过 uv:
```bash
uv run script.py
```

## 关键物理参数

- 质量范围: log10(Mh) ∈ [9, 16] Msun
- 默认宇宙学: FlatLambdaCDM (见 mah/Cosmology)
- 原子冷却温度: T_vir = 1e4 K
- 平均分子量: μ = 0.61
- AB 零点: AB_ZEROPOINT_LNU = 51.60
- Minihalo 冷却: V_cool_H2 = sqrt(a² + (b·v_bc)²), a=3.714, b=4.015

## 绑图约定

- 使用 `plt.style.use("apj")` 作为默认绑图样式
- 论文/幻灯片图: slides/assets/ (矢量 PDF 优先)
- 诊断/快看图: outputs/
- 可复用数据表: data_save/

## 观测数据

`data/` 下按红移目录存放观测 UVLF 数据:
- redshift_6, redshift_8, redshift_9, redshift_10, redshift_11, redshift_12p5
- bowler_uvlf_z6.npz, bowler_uvlf_z8.npz
- donnan_uvlf_z8.npz, mclure_uvlf_z8.npz
- universemachine_dr1 (宇宙机器数据)
