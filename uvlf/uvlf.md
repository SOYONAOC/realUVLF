下面是固定质量范围后的 **Codex 流程说明**。我已经把范围写死为：

[
M_{\min}=10^9,M_\odot,\qquad M_{\max}=10^{16},M_\odot
]

对应：

[
\log_{10} M_{\min}=9,\qquad \log_{10} M_{\max}=16
]

你可以直接发给 Codex。

---

# UVLF 计算方案

目标是用两层 Monte Carlo 方法计算 UVLF，不对 halo mass 分很多 bin，而是直接在连续质量空间中抽样。

## 总体思路

我们要计算的是：

[
\phi(L)=\int d\log M_{\rm h},\frac{dn}{d\log M_{\rm h}}(M_{\rm h},z_{\rm obs}),P(L\mid M_{\rm h})
]

其中：

* (\frac{dn}{d\log M_{\rm h}}) 由 halo mass function 给出
* (P(L\mid M_{\rm h})) 不做解析拟合，而是通过 `run_halo_uv_pipeline()` 直接抽样得到

数值实现采用两层结构：

1. 第一层在 (\log M_{\rm h}) 上均匀抽取一批 halo 终质量
2. 第二层对每个抽到的 (M_{\rm h}) 调用 `run_halo_uv_pipeline()`，得到该质量下的一组 luminosity realization
3. 最后结合 halo mass function 的权重，对所有 luminosity 样本做加权统计，得到 UVLF

---

## 固定参数

采用以下设置：

[
N_{\rm mass}=3000,\qquad n_{\rm tracks}=1000
]

并固定质量范围为：

[
\log_{10} M_{\rm h}\in[9,16]
]

也就是：

[
M_{\rm h}\in[10^9,10^{16}],M_\odot
]

含义是：

* 抽取 3000 个不同的 halo 终质量
* 对每个质量生成 1000 个 luminosity realization

---

## 第一层：抽取 halo 质量

在固定区间

[
\log_{10} M_{\rm h}\in[9,16]
]

上做均匀抽样：

[
\log_{10} M_{{\rm h},i}\sim \mathrm{Uniform}(9,16)
]

然后转成

[
M_{{\rm h},i}=10^{\log_{10} M_{{\rm h},i}}
]

这样做的原因是实现简单，而且高质量端不会太稀疏。

---

## 第二层：由 (M_{\rm h}) 生成 luminosity 样本

对每个抽到的 (M_{{\rm h},i})，直接调用：

```python
run_halo_uv_pipeline(Mh_final=Mh_i, z_final=z_obs, n_tracks=1000, ...)
```

把这个函数视为一个黑箱条件采样器，即：

[
M_{{\rm h},i}\rightarrow {L_{ij}}*{j=1}^{n*{\rm tracks}}
]

也就是说，对于每个质量点，都生成 1000 个 luminosity realization，用来表征该质量下的非高斯散布。

---

## 质量权重

由于第一层是在 (\log M_{\rm h}) 上均匀抽样，所以每个质量样本都需要用 halo mass function 加权。

对每个 (M_{{\rm h},i})，计算：

[
\frac{dn}{d\log M_{\rm h}}(M_{{\rm h},i},z_{\rm obs})
]

如果已有接口返回的是 (dn/dM)，则先转换成：

[
\frac{dn}{d\log M}=M\ln 10\frac{dn}{dM}
]

然后把这个量作为该质量样本的权重：

[
w_i \propto \frac{dn}{d\log M_{\rm h}}(M_{{\rm h},i},z_{\rm obs})
]

---

## 把质量权重分配到 luminosity 样本

如果某个质量样本 (M_{{\rm h},i}) 生成了 1000 个 luminosity realization，那么每个 realization 的权重取为：

[
w_{ij}=\frac{w_i}{1000}
]

这样该质量点的总权重保持不变。

---

## 最终统计 UVLF

把所有质量点生成的 luminosity 样本合并起来，形成总样本集合：

[
{L_{ij},,w_{ij}}
]

然后对 luminosity 或 UV magnitude 做加权 histogram，得到 UVLF。

如果使用 luminosity (L)，则输出 (\phi(L))；
如果使用 UV magnitude (M_{\rm UV})，则输出 (\phi(M_{\rm UV}))。

---

## 需要实现的主函数

请实现一个主函数，例如：

```python
sample_uvlf_from_hmf(
    z_obs,
    N_mass=3000,
    n_tracks=1000,
    random_seed=42,
    ...
)
```

其中质量范围固定为：

* `logM_min = 9`
* `logM_max = 13`

这个函数应完成以下步骤：

1. 在 ([9,16]) 上均匀抽取 3000 个 `logMh`
2. 转成 `Mh`
3. 计算每个 `Mh` 对应的 halo mass function 权重
4. 对每个 `Mh` 调用 `run_halo_uv_pipeline(Mh_final=Mh_i, z_final=z_obs, n_tracks=1000, ...)`
5. 从返回结果中提取 luminosity 或 UV magnitude 样本
6. 为每个 realization 分配对应权重
7. 汇总全部样本
8. 对全部样本做加权 histogram，得到 UVLF
9. 返回样本表和 UVLF 结果

---

## 结果要求

最终返回至少包括两部分：

### 1. 样本表

包含每个 luminosity realization 的信息，例如：

* `logMh`
* `Mh`
* `mass_weight`
* `track_index`
* `luminosity` 或 `Muv`
* `sample_weight`

### 2. UVLF

包含分 bin 统计后的结果，例如：

* bin edges
* bin centers
* weighted counts
* UVLF value

---

## 核心思想总结

这套方法的本质是：

* 外层对 halo mass function 做 Monte Carlo 积分
* 内层用 `run_halo_uv_pipeline()` 采样条件分布 (P(L\mid M_{\rm h}))
* 最后通过加权统计得到总体 UVLF

它不需要把 halo mass 划分成很多 bins，并且可以自然保留 (M_{\rm h}\to L) 映射中的非高斯散布。
