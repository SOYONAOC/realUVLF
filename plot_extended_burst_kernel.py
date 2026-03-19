#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from mah.models import CosmologySet, GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN, KM_PER_MPC, SECONDS_PER_GYR


FONT_PATH = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path("outputs")
PNG_PATH = OUTPUT_DIR / "extended_burst_kernel_tdyn.png"
PDF_PATH = OUTPUT_DIR / "extended_burst_kernel_tdyn.pdf"
TXT_PATH = OUTPUT_DIR / "extended_burst_kernel_tdyn.txt"


def virial_density_and_delta_vir(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cosmo = CosmologySet()
    hubble = cosmo.H0u * np.sqrt(cosmo.omegam * (1.0 + z) ** 3 + cosmo.omegalam)
    rho_crit = cosmo.rhocrit * (hubble / cosmo.H0u) ** 2
    omega_m_z = cosmo.omegam * (1.0 + z) ** 3 / (cosmo.omegam * (1.0 + z) ** 3 + cosmo.omegalam)
    delta = omega_m_z - 1.0
    delta_vir = 18.0 * np.pi**2 + 82.0 * delta - 39.0 * delta**2
    rho_vir = delta_vir * rho_crit
    return rho_vir, delta_vir


def tdyn_y16_myr(z: np.ndarray) -> np.ndarray:
    rho_vir, _ = virial_density_and_delta_vir(z)
    t_gyr = np.sqrt(3.0 * np.pi / (32.0 * GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN * rho_vir)) * (KM_PER_MPC / SECONDS_PER_GYR)
    return t_gyr * 1.0e3


def tau_del_current_myr(z: np.ndarray) -> np.ndarray:
    rho_vir, _ = virial_density_and_delta_vir(z)
    t_gyr = np.sqrt(3.0 / (4.0 * np.pi * GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN * rho_vir)) * (KM_PER_MPC / SECONDS_PER_GYR)
    return t_gyr * 1.0e3


def burst_kernel(delta_t_myr: np.ndarray, td_myr: float, kappa: float = 1.0) -> np.ndarray:
    x = np.asarray(delta_t_myr, dtype=float)
    kernel = x / (kappa**2 * td_myr**2) * np.exp(-x / (kappa * td_myr))
    kernel[x < 0.0] = 0.0
    return kernel


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    z_grid = np.linspace(0.0, 30.0, 500)
    td_myr = tdyn_y16_myr(z_grid)
    tau_myr = tau_del_current_myr(z_grid)

    z_examples = [6.0, 12.0, 20.0]
    colors = ["#1f4e79", "#c0392b", "#1d8348"]
    td_examples = {z: float(tdyn_y16_myr(np.array([z]))[0]) for z in z_examples}

    max_td = max(td_examples.values())
    delta_t = np.linspace(0.0, 6.0 * max_td, 1200)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(z_grid, td_myr, color="#c0392b", linewidth=2.4, label=r"$t_d(z_f)$ (Y16)")
    ax.plot(z_grid, tau_myr, color="#2e86de", linewidth=2.2, linestyle="--", label=r"$\tau_{\rm del}=r_{\rm vir}/V_c$ (current)")
    ax.set_yscale("log")
    ax.set_xlabel("红移 z")
    ax.set_ylabel("时间尺度 [Myr]")
    ax.set_title(r"$t_d(z)$ 与当前 $\tau_{\rm del}$ 的比较")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=10)

    ratio = tau_myr / td_myr
    ax.text(
        0.04,
        0.05,
        rf"$\tau_{{\rm del}}/t_d \approx {float(np.median(ratio)):.2f}$（几乎与 z 无关）",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    ax = axes[1]
    for z, color in zip(z_examples, colors, strict=True):
        td = td_examples[z]
        g = burst_kernel(delta_t, td, kappa=1.0)
        ax.plot(delta_t, g, color=color, linewidth=2.2, label=rf"$z_f={z:g}$, $t_d={td:.1f}\,\rm Myr$")
        ax.axvline(td, color=color, linestyle=":", alpha=0.8, linewidth=1.0)
    ax.set_xlabel(r"$\Delta t = t-t'$ [Myr]")
    ax.set_ylabel(r"$g(\Delta t)$ [Myr$^{-1}$]")
    ax.set_title(r"Extended-burst kernel $g(\Delta t)=\frac{\Delta t}{\kappa^2 t_d^2}e^{-\Delta t/(\kappa t_d)}$（$\kappa=1$）")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.text(
        0.04,
        0.05,
        "这个核从 0 开始上升，在 Δt = κ t_d 处达到峰值；\n"
        "对 κ=1，平均延迟时间是 2 t_d。",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    fig.savefig(PNG_PATH, dpi=220)
    fig.savefig(PDF_PATH)

    lines = [
        "Extended-burst delay kernel diagnostics",
        "g(Δt) = Δt / (κ^2 t_d^2) * exp[-Δt / (κ t_d)]",
        "For κ = 1: mode = t_d, mean = 2 t_d, integral = 1",
        "",
    ]
    for z in z_examples:
        td = td_examples[z]
        tau = float(tau_del_current_myr(np.array([z]))[0])
        lines.extend(
            [
                f"z={z:g}: t_d(Y16) = {td:.6f} Myr",
                f"z={z:g}: tau_del(current) = {tau:.6f} Myr",
                f"z={z:g}: tau_del / t_d = {tau / td:.6f}",
            ]
        )
    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(PNG_PATH.resolve())
    print(PDF_PATH.resolve())
    print(TXT_PATH.resolve())


if __name__ == "__main__":
    main()
