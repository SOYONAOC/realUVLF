#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


FONT_PATH = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT_PATH)
FONT_NAME = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["font.family"] = FONT_NAME
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
SLIDE_PNG = OUTPUT_DIR / "uvlf_project_summary_slide.png"
SLIDE_PDF = OUTPUT_DIR / "uvlf_project_summary_slide.pdf"

RATIO_PLOT = OUTPUT_DIR / "ssp_vs_instant_same_sfh.png"
CONV_PLOT = OUTPUT_DIR / "ssp_convolution_explanation_1.png"


def add_panel(ax, x, y, w, h, title, body_lines, face="#ffffff", edge="#d0d7de"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.015,
        y + h - 0.035,
        title,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        va="top",
        color="#0f172a",
    )
    ax.text(
        x + 0.018,
        y + h - 0.085,
        "\n".join(body_lines),
        transform=ax.transAxes,
        fontsize=12.8,
        va="top",
        color="#111827",
        linespacing=1.45,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(
        0.04,
        0.95,
        "realUVLF 当前工作总结：UVLF / SSP / Dust 诊断",
        fontsize=24,
        fontweight="bold",
        color="#0f172a",
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.04,
        0.915,
        "目标：梳理目前已经完成的工作、核心结果，以及模型当前不稳妥的地方，方便向 supervisor 汇报。",
        fontsize=12.5,
        color="#475569",
        va="top",
        transform=ax.transAxes,
    )

    add_panel(
        ax,
        0.04,
        0.56,
        0.36,
        0.30,
        "1. 已完成的工作",
        [
            "• 已搭建并跑通完整链路：MAH → SFR → SSP 卷积 → UVLF。",
            "• 已统一并检查 dust 实现；当前保留物理截断：dust UVLF 不允许高于 no-dust UVLF。",
            "• 已做 apples-to-apples 测试：固定同一批 MAH+SFR，只比较 SSP 卷积和瞬时 UV 标定。",
            "• 已单独检查 SFR、UV 卷积窗口、K_UV,eff、以及不同红移下的 MAH/SFR 历史。",
        ],
        face="#ffffff",
    )

    add_panel(
        ax,
        0.04,
        0.19,
        0.36,
        0.32,
        "2. 关键结果",
        [
            "• 瞬时 SFR 与 standard 模型本身很接近：中位数只差约 0%–10%。",
            "• 真正的大差异出在 UV 标定：SSP / instant ≈",
            "  z=6: 2.43–2.48x,  z=8: 1.57–1.62x,",
            "  z=10: 1.23–1.29x, z=12.5: 1.07–1.12x。",
            "• SSP 的有效 UV 记忆时间主要是 30–100 Myr；300 Myr 之外贡献已很小。",
            "• 代表性轨道的等效 K_UV：z=6 ≈ 4.80e-29，z=12.5 ≈ 1.11e-28。",
        ],
        face="#ffffff",
    )

    add_panel(
        ax,
        0.42,
        0.19,
        0.22,
        0.67,
        "3. 当前模型的问题",
        [
            "• 这张 SSP 表在长期恒定 SFR 极限下给出",
            "  K_UV,SSP,long ≈ 6.1e-29，明显小于 standard 的 1.17e-28。",
            "• 结果是：在低红移，当前模型会系统性给出更亮的 UV。",
            "• 因此如果用同一套 f_star 参数去同时 fit 低 z 和高 z，",
            "  低 z 往往被压得更低，高 z tension 反而会变大。",
            "• 这不表示卷积公式本身有 bug；更像是当前 UV calibration",
            "  与 standard baseline 不是同一个归一化体系。",
            "",
            "当前判断：",
            "• 方法形式上更物理，但当前这套 UV 标定还不够稳妥，",
            "  不能直接当成绝对 UVLF 拟合的最终结论。",
        ],
        face="#fff7ed",
        edge="#fdba74",
    )

    add_panel(
        ax,
        0.66,
        0.19,
        0.30,
        0.15,
        "4. 建议下一步",
        [
            "• 若要与文献标准模型公平比较，需要明确选择统一的 UV calibration。",
            "• 建议把当前结果汇报为：internal-consistent，但 absolute UV normalization 尚不稳健。",
        ],
        face="#eff6ff",
        edge="#93c5fd",
    )

    ratio_img = mpimg.imread(RATIO_PLOT)
    conv_img = mpimg.imread(CONV_PLOT)

    ax_ratio = fig.add_axes([0.66, 0.54, 0.30, 0.28])
    ax_ratio.imshow(ratio_img)
    ax_ratio.set_title("SSP / instant 随红移变化", fontsize=14, pad=6)
    ax_ratio.axis("off")

    ax_conv = fig.add_axes([0.66, 0.19 + 0.17, 0.30, 0.17])
    ax_conv.imshow(conv_img)
    ax_conv.set_title("卷积图解：z=6 vs z=12.5", fontsize=14, pad=6)
    ax_conv.axis("off")

    ax.text(
        0.04,
        0.06,
        "一句话结论：当前项目在方法形式上更物理，但当前 SSP UV 标定会系统性抬高低红移 UV，"
        "使得跨红移联合拟合更紧张，因此现阶段更适合做趋势诊断，不适合直接作为绝对 UVLF 归一化的最终结论。",
        fontsize=13,
        color="#7c2d12",
        fontweight="bold",
        transform=ax.transAxes,
        va="bottom",
    )

    fig.savefig(SLIDE_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(SLIDE_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()
