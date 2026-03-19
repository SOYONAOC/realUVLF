#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from ssp import load_uv1600_table
from sfr.calculator import EXTENDED_BURST_KAPPA
from plot_extended_burst_kernel import burst_kernel, tdyn_y16_myr


FONT_PATH = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path("outputs")
PNG_PATH = OUTPUT_DIR / "effective_uv_kernel.png"
PDF_PATH = OUTPUT_DIR / "effective_uv_kernel.pdf"
TXT_PATH = OUTPUT_DIR / "effective_uv_kernel.txt"
SSP_FILE = "spectra-bin_byrne23/spectra-bin-imf135_300.BASEL.z001.a+00.dat"


def effective_kernel(delta_t_myr: np.ndarray, ssp_age_myr: np.ndarray, ssp_luv: np.ndarray, td_myr: float, kappa: float) -> np.ndarray:
    result = np.zeros_like(delta_t_myr, dtype=float)
    for i, dt in enumerate(delta_t_myr):
        tau = np.linspace(0.0, dt, max(10, int(min(1200, dt * 4 + 10))), dtype=float)
        g = burst_kernel(tau, td_myr, kappa=kappa)
        ssp = np.interp(dt - tau, ssp_age_myr, ssp_luv, left=ssp_luv[0], right=0.0)
        result[i] = np.trapezoid(g * ssp, x=tau)
    return result


def weighted_percentile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    cdf = np.cumsum(ws)
    if cdf[-1] <= 0.0:
        return float(xs[-1])
    cdf = cdf / cdf[-1]
    return float(np.interp(q, cdf, xs))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ages_myr, luv_per_msun = load_uv1600_table(SSP_FILE)
    ages_myr = np.asarray(ages_myr, dtype=float)
    luv_per_msun = np.asarray(luv_per_msun, dtype=float)

    delta_t = np.linspace(0.0, 500.0, 1200)
    ssp_kernel = np.interp(delta_t, ages_myr, luv_per_msun, left=luv_per_msun[0], right=0.0)

    z_values = [6.0, 12.0]
    colors = {6.0: "#c0392b", 12.0: "#1f4e79"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(delta_t, ssp_kernel / np.max(ssp_kernel), color="black", linewidth=2.3, label="原始 SSP kernel")

    summary_lines = [
        "Effective UV kernel diagnostics",
        f"kappa: {EXTENDED_BURST_KAPPA}",
        f"ssp_file: {SSP_FILE}",
        "",
    ]

    for z in z_values:
        td = float(tdyn_y16_myr(np.array([z]))[0])
        eff = effective_kernel(delta_t, ages_myr, luv_per_msun, td, EXTENDED_BURST_KAPPA)
        eff_norm = eff / np.max(eff)
        ax.plot(delta_t, eff_norm, color=colors[z], linewidth=2.2, label=rf"$K_{{\rm eff}}(\Delta t),\ z={z:g}$")

        ssp_p50 = weighted_percentile(delta_t, ssp_kernel, 0.5)
        eff_p50 = weighted_percentile(delta_t, eff, 0.5)
        ssp_p90 = weighted_percentile(delta_t, ssp_kernel, 0.9)
        eff_p90 = weighted_percentile(delta_t, eff, 0.9)
        summary_lines.extend(
            [
                f"z={z:g}: t_d = {td:.6f} Myr",
                f"z={z:g}: SSP_kernel_p50 = {ssp_p50:.6f} Myr",
                f"z={z:g}: K_eff_p50 = {eff_p50:.6f} Myr",
                f"z={z:g}: SSP_kernel_p90 = {ssp_p90:.6f} Myr",
                f"z={z:g}: K_eff_p90 = {eff_p90:.6f} Myr",
            ]
        )

    ax.set_xlabel(r"$\Delta t$ [Myr]")
    ax.set_ylabel("归一化核")
    ax.set_title("原始 SSP kernel 与合成后的 $K_{\\rm eff}(\\Delta t)$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=10)

    ax = axes[1]
    for z in z_values:
        td = float(tdyn_y16_myr(np.array([z]))[0])
        g = burst_kernel(delta_t, td, kappa=EXTENDED_BURST_KAPPA)
        eff = effective_kernel(delta_t, ages_myr, luv_per_msun, td, EXTENDED_BURST_KAPPA)
        ax.plot(delta_t, g / np.max(g), linestyle="--", color=colors[z], linewidth=1.7, label=rf"$g(\Delta t),\ z={z:g}$")
        ax.plot(delta_t, eff / np.max(eff), linestyle="-", color=colors[z], linewidth=2.2, label=rf"$K_{{\rm eff}},\ z={z:g}$")
    ax.set_xlabel(r"$\Delta t$ [Myr]")
    ax.set_ylabel("归一化核")
    ax.set_title(r"$g(\Delta t)$ 如何把 UV 记忆窗口向更长时间拉宽")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.savefig(PNG_PATH, dpi=220)
    fig.savefig(PDF_PATH)
    TXT_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(PNG_PATH.resolve())
    print(PDF_PATH.resolve())
    print(TXT_PATH.resolve())


if __name__ == "__main__":
    main()
