#!/usr/bin/env python3
"""Plot a compact cosmic timeline from z=0 to z=2000."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18
from matplotlib import font_manager

FONT_PATH = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


OUTPUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUTPUT_DIR / "cosmic_timeline_z0_2000.png"
PDF_PATH = OUTPUT_DIR / "cosmic_timeline_z0_2000.pdf"


def transformed_z(z: np.ndarray | float) -> np.ndarray | float:
    """Use log10(1+z) so low- and high-z phases fit on one axis."""
    return np.log10(1.0 + np.asarray(z, dtype=float))


def age_myr(z: np.ndarray | float) -> np.ndarray | float:
    """Cosmic age in Myr."""
    return Planck18.age(np.asarray(z, dtype=float)).to_value("Myr")


def main() -> None:
    z = np.linspace(0.0, 2000.0, 4000)
    x = transformed_z(z)
    t_myr = age_myr(z)

    fig = plt.figure(figsize=(12.0, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[5.5, 1.5])
    ax = fig.add_subplot(gs[0])
    ax_stage = fig.add_subplot(gs[1], sharex=ax)

    ax.plot(x, t_myr, color="#1f4e79", lw=3, label="Planck18 宇宙年龄")
    ax.set_yscale("log")
    ax.set_ylabel("宇宙年龄 [Myr]")
    ax.set_title("宇宙时间线：z = 0 到 2000")
    ax.grid(True, which="both", alpha=0.25)

    stages = [
        ("今天 / 暗能量主导", 0.0, 0.7, "#d8ead3"),
        ("星系组装高峰", 0.7, 6.0, "#fff2cc"),
        ("再电离时期", 6.0, 10.0, "#fce5cd"),
        ("宇宙黎明", 10.0, 30.0, "#f4cccc"),
        ("黑暗时代", 30.0, 1089.0, "#d9d2e9"),
        ("电离等离子体时期", 1089.0, 2000.0, "#cfe2f3"),
    ]

    for name, z_min, z_max, color in stages:
        x0 = transformed_z(z_min)
        x1 = transformed_z(z_max)
        ax.axvspan(x0, x1, color=color, alpha=0.28, lw=0)
        ax_stage.axvspan(x0, x1, color=color, alpha=0.85, lw=0)
        ax_stage.text(
            0.5 * (x0 + x1),
            0.5,
            name,
            ha="center",
            va="center",
            fontsize=10,
            rotation=0,
        )

    event_specs = [
        ("今天", 0.0, "13.8 Gyr"),
        ("再电离基本结束", 6.0, "~0.94 Gyr"),
        ("第一代恒星", 20.0, "~0.18 Gyr"),
        ("开始具备恒星形成条件", 25.0, "~0.13 Gyr"),
        ("开始具备星系形成条件", 15.0, "~0.27 Gyr"),
        ("CMB 解耦", 1089.0, "~0.38 Myr"),
    ]

    for label, z_evt, text_age in event_specs:
        x_evt = transformed_z(z_evt)
        t_evt = float(age_myr(z_evt))
        ax.axvline(x_evt, color="#444444", lw=1.2, ls="--", alpha=0.7)
        ax.scatter([x_evt], [t_evt], color="#8b0000", s=28, zorder=5)
        ax.annotate(
            f"{label}\nz={z_evt:g}, {text_age}",
            xy=(x_evt, t_evt),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.8},
        )

    off_scale_note = (
        "更早的阶段如暴涨、原初核合成发生在远高于这里的红移\n"
        "(z >> 2000)，因此不在这张图的横轴范围内。"
    )
    ax.text(
        0.02,
        0.06,
        off_scale_note,
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        color="#333333",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#f5f5f5", "ec": "#cccccc", "alpha": 0.95},
    )

    tick_z = np.array([0, 1, 3, 6, 10, 30, 100, 300, 1000, 2000], dtype=float)
    tick_x = transformed_z(tick_z)
    ax_stage.set_xticks(tick_x)
    ax_stage.set_xticklabels([f"{int(v)}" if float(v).is_integer() else f"{v:g}" for v in tick_z])
    ax_stage.set_xlabel("红移 z")
    ax_stage.set_yticks([])
    ax_stage.set_ylim(0, 1)
    ax_stage.set_ylabel("阶段")

    age_tick_z = np.array([0, 1, 6, 10, 30, 100, 300, 1089, 2000], dtype=float)
    age_tick_x = transformed_z(age_tick_z)
    age_tick_labels = []
    for z_tick in age_tick_z:
        t_tick = float(age_myr(z_tick))
        if t_tick >= 1000.0:
            age_tick_labels.append(f"{t_tick / 1000.0:.1f} Gyr")
        else:
            age_tick_labels.append(f"{t_tick:.1f} Myr")
    top = ax.secondary_xaxis("top")
    top.set_xticks(age_tick_x)
    top.set_xticklabels(age_tick_labels, rotation=0)
    top.set_xlabel("若干红移对应的宇宙年龄")

    for spine in ("top", "right"):
        ax_stage.spines[spine].set_visible(False)

    fig.savefig(PNG_PATH, dpi=220)
    fig.savefig(PDF_PATH)
    print(f"saved_png={PNG_PATH}")
    print(f"saved_pdf={PDF_PATH}")


if __name__ == "__main__":
    main()
