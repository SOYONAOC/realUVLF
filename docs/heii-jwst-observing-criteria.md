# He II 的 JWST 观测判据

2026-09-11。依据用户要求：观测预测必须对应有来源的波段、仪器模式和噪声条件。
本次更新观测判据及 slides，未运行 ETC，未新增仪器灵敏度或检出率结果。
原有 1e−20、1e−19、1e−18 erg s−1 cm−2 只描述通量分布，不作为 JWST 灵敏度。

## 波段与模式

He II 的观测波长为 λobs = 1640 Å × (1+z)。当前 z=12.5、14.5 对应
2.215、2.543 μm。先在这一实际波段评估响应，不把整台望远镜的名义覆盖
范围当作灵敏度相同的区间。

| 候选配置 | 标称范围 | 用途 |
| --- | --- | --- |
| NIRSpec PRISM/CLEAR | 0.60–5.30 μm | 匹配现有深场低分辨率光谱 |
| NIRSpec G235M/F170LP | 1.66–3.17 μm，R≈1000 | 评估相邻谱线分离及线宽影响 |

来源：[STScI NIRSpec Dispersers and Filters, Table 2](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters)，
核对日期 2026-09-11。PRISM 的分辨率随波长变化；需检查实际 MSA 位置的
有效波段、坏像元与谱线混合。不能预先宣称某模式更灵敏。

## 有物理与统计定义的计算标准

对每个目标记录红移、UV 测量及误差、透镜、线形和本征 FWHM、连续谱、
源形态/大小、MSA 或狭缝内位置。观测设置记录模式、读出、组数、积分数、
曝光数、有效曝光时间、背景场景、提取孔径及背景扣除方式。
曝光时间必须来自实际项目或明确的时间预算；探索性预算只标为方案假设。

工作统计量为积分谱线 S/N = F1640/σF。σF 是与该通量估计量相同口径的
不确定度，包含源与背景光子噪声、暗电流、读出噪声及噪声相关性。
由真实光谱的协方差和谱线拟合，或指定版本/参考数据的 ETC 计算取得。
ETC 的单波长、单像元 S/N 不直接等于整条线的积分 S/N。

计划观测预先定义积分线 S/N≥5 的工作标准；5 是统计选择，不是物理常数。
弱源且 σF 近似不随通量变化时，F5σ≈5σF；一般应求解 S/N(F)=5。
复现已发表观测时保留原文 3σ 或其他标准，不把上限擅自重新解释成 5σ。
相关噪声、红移搜索、连续谱拟合和混线会改变显著性，正式检验应沿用相同
测量流程，并用注入回收检验检出完备度与误报。

方法依据：[STScI NIRSpec Performance](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-performance)；
[JWST ETC Images and Plots](https://jwst-docs.stsci.edu/jwst-exposure-time-calculator-overview/jwst-etc-outputs-overview/jwst-etc-images-and-plots)。
STScI 推荐使用 ETC 比较具体源的信噪比；NIRSpec 的 ETC 1D 与 2D 信噪比
输出对相关读出噪声的处理不同，取用时需核对。

## 已有观测作为基准

GS-z14-1 的约 56 h PRISM 光谱给出 He II <7e−20 erg s−1 cm−2（3σ）。
论文采用仪器分辨率对应的未分辨高斯线，广义最小二乘使用谱线协方差，
并抽样红移不确定度。这个上限只约束匹配该对象的预测，不能当成统一的
“56 h JWST 门槛”，也不把上限/3 当作已恢复的完整噪声模型。
来源：[Wu et al. (2025), §III.3, Table 2](https://arxiv.org/html/2507.22858)。
现有两个对象的结果和近邻红移近似见 [观测对照记录](heii-observation-comparison.md)。

## 接下来的交付标准

1. 先复现真实对象的测量口径；正式比较前补齐目标红移与 UV 误差处理。
2. 使用同一源模型、明确曝光预算比较 PRISM 与 G235M。归档 ETC 配置、版本、
   参考数据、输出及积分线信噪比的提取方法；得到 F5σ(λ) 或所需曝光 t5σ(F)。
3. 将星系分布与检出完备度结合。对实际目标 j，以条件通量分布 p_j(F)
   计算 Pdet,j = ∫ C_j(F) p_j(F) dF，预测检出数 Nexp = Σj Pdet,j。
   C_j 必须由观测噪声/注入回收决定。HMF 丰度权重不能代替实际观测选择函数。
4. 未检出对象进入含噪声的似然；不把模型真实通量 CDF 当作未检出概率。

现有源模型和科学结果保持不变；新的仪器预测需要上述真实输入后才生成。
