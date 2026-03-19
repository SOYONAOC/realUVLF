#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


FONT_PATH = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT_PATH)
FONT_NAME = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["font.family"] = FONT_NAME
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
SLIDES_DIR = ROOT / "slides"
PDF_PATH = SLIDES_DIR / "uvlf_status_deck.pdf"


def newest(pattern: str) -> Path:
    matches = list(OUTPUT_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched pattern: {pattern}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def add_text_box(ax, x, y, w, h, title, lines, face="#ffffff", edge="#cbd5e1"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.018,
        y + h - 0.035,
        title,
        transform=ax.transAxes,
        va="top",
        fontsize=22,
        fontweight="bold",
    )
    ax.text(
        x + 0.020,
        y + h - 0.105,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        fontsize=19,
        linespacing=1.7,
        color="#0f172a",
    )


def add_image(fig, rect, path: Path, title: str):
    ax = fig.add_axes(rect)
    ax.imshow(mpimg.imread(path))
    ax.set_title(title, fontsize=18, pad=8)
    ax.axis("off")
    return ax


def page_title(fig, title: str, subtitle: str = ""):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(
        0.04,
        0.95,
        title,
        transform=ax.transAxes,
        va="top",
        fontsize=30,
        fontweight="bold",
        color="#0f172a",
    )
    if subtitle:
        ax.text(0.04, 0.905, subtitle, transform=ax.transAxes, va="top", fontsize=16, color="#475569")
    return ax


def build_deck() -> None:
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    z6 = newest("uvlf_compare_no_puv_z6*.png")
    z8 = newest("uvlf_compare_no_puv_z8*.png")
    z10 = newest("uvlf_compare_no_puv_z10*.png")
    z12p5 = newest("uvlf_compare_no_puv_z12p5*.png")
    ratio = newest("ssp_vs_instant_same_sfh.png")
    conv = newest("ssp_convolution_explanation_1.png")
    mah = newest("mah_sfr_four_z_2.png")

    with PdfPages(PDF_PATH) as pdf:
        fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
        ax = page_title(
            fig,
            "realUVLF：UVLF / SSP / Dust 结果总结",
            "",
        )
        add_text_box(
            ax,
            0.06,
            0.58,
            0.88,
            0.20,
            "工作概述",
            [
                "• 完成 MAH → SFR → SSP convolution → UVLF 全链路计算。",
                "• 统一 current 与 standard 对比时使用的 dust 实现。",
                "• 系统诊断 current / standard 偏差的来源与红移依赖。",
            ],
            face="#ffffff",
        )
        add_text_box(
            ax,
            0.06,
            0.22,
            0.88,
            0.26,
            "核心结论",
            [
                "• 当前方法形式上更物理，但当前 SSP UV calibration 会系统性抬高低红移 UV。",
                "• 这会放大跨红移联合拟合的 tension。",
                "• 因而现阶段更适合做机制诊断，不适合直接作为绝对 UVLF 归一化的最终结论。",
            ],
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
        ax = page_title(fig, "1. 已完成工作", "")
        add_text_box(
            ax,
            0.06,
            0.18,
            0.88,
            0.68,
            "目前已经完成",
            [
                "• 跑通完整链路：MAH → SFR → SSP convolution → UVLF。",
                "• 统一 current / standard 对比中的 dust 实现，并保留 dust ≤ no-dust 的物理截断。",
                "• 完成 apples-to-apples 诊断：固定同一批 MAH+SFR，只比较 SSP 卷积与瞬时 UV 标定。",
                "• 单独检查了 SFR、SSP 时间窗口、K_UV,eff，以及不同红移下的 MAH / SFR 历史。",
                "• 补齐 z=6, 8, 10, 12.5 的同参数 current vs standard 正式对比图。",
            ],
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
        ax = page_title(fig, "2. 关键结论", "")
        add_text_box(
            ax,
            0.06,
            0.18,
            0.88,
            0.68,
            "关键结论",
            [
                "• 瞬时 SFR 与 standard 本身很接近：中位数通常只差 0%–10%。",
                "• 主要差异来自 UV calibration，而不是 SFR 本身。",
                "• SSP / instant 的平均亮度比：",
                "  z=6: 2.43–2.48x；z=8: 1.57–1.62x",
                "  z=10: 1.23–1.29x；z=12.5: 1.07–1.12x",
                "• SSP 的有效 UV 记忆时间主要在 30–100 Myr。",
            ],
            face="#eff6ff",
            edge="#93c5fd",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
        ax = page_title(fig, "3. 当前模型存在的问题", "")
        add_text_box(
            ax,
            0.06,
            0.18,
            0.88,
            0.68,
            "问题在哪里",
            [
                "• 这张 SSP 表在长期恒定 SFR 极限下给出 K_UV,SSP,long ≈ 6.1e-29。",
                "• 该值明显小于 standard 的 1.17e-28，因此低红移 UV 会被系统性抬亮。",
                "• 高红移时 SFH 更陡、更年轻，K_UV,eff 更接近 standard，因此 SSP 增强变弱。",
                "• 用同一套 f_star 同时拟合低 z 和高 z 时，当前 UV calibration 会放大 high-z tension。",
                "• 目前这套模型更适合做趋势诊断，不适合作为绝对 UVLF 归一化的最终结论。",
            ],
            face="#fff7ed",
            edge="#fdba74",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
        page_title(fig, "4. 同参数对比：z = 6, 8", "")
        add_image(fig, [0.05, 0.16, 0.42, 0.70], z6, "z = 6")
        add_image(fig, [0.53, 0.16, 0.42, 0.70], z8, "z = 8")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
        page_title(fig, "5. 同参数对比：z = 10, 12.5", "")
        add_image(fig, [0.05, 0.16, 0.42, 0.70], z10, "z = 10")
        add_image(fig, [0.53, 0.16, 0.42, 0.70], z12p5, "z = 12.5")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
        page_title(fig, "6. SSP / instant 随红移变化", "")
        add_image(fig, [0.07, 0.18, 0.86, 0.66], ratio, "SSP / instant 的亮度比随红移变化")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
        ax = page_title(fig, "7. 卷积图解：z = 6 与 z = 12.5", "")
        add_image(fig, [0.05, 0.16, 0.56, 0.70], conv, "SFR、SSP kernel、积分贡献、K_UV,eff")
        add_text_box(
            ax,
            0.66,
            0.22,
            0.28,
            0.58,
            "关键点",
            [
                "• 高红移时当前 SFR 很强，但过去几十 Myr 的“年轻恒星库存”积得不够。",
                "• 因此 K_UV,eff 更接近 standard，SSP 相对 instant 的增强变弱。",
                "• 低红移时库存更接近稳态，因此 K_UV,eff 更小，UV 被显著抬亮。",
            ],
            face="#ffffff",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
        ax = page_title(fig, "8. MAH / SFR 历史诊断", "")
        add_image(fig, [0.05, 0.16, 0.56, 0.70], mah, "四个红移下的 MAH 与有效 SFR 历史")
        add_text_box(
            ax,
            0.66,
            0.22,
            0.28,
            0.58,
            "关键点",
            [
                "• 固定同一个终质量时，高红移 halo 的吸积率和瞬时 SFR 更高。",
                "• 但 SSP 增强不仅取决于当前 SFR，还取决于过去几十到一百 Myr 的库存。",
                "• z=12.5 更像“快速上升”的系统；z=6 更接近已经积累了一段时间的系统。",
            ],
            face="#ffffff",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
        ax = page_title(fig, "9. 结论", "")
        add_text_box(
            ax,
            0.09,
            0.26,
            0.82,
            0.46,
            "总结",
            [
                "• 当前 pipeline 在形式上更物理，因为它使用了完整的 SSP 卷积。",
                "• 但当前 SSP UV calibration 会在低红移给出更亮的 UV 归一化，从而放大跨红移联合拟合的 tension。",
                "• 现阶段这套模型更适合做机制与趋势诊断，尚不适合直接作为绝对 UVLF 归一化的最终结论。",
            ],
            face="#eff6ff",
            edge="#93c5fd",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    build_deck()
