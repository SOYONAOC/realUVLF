#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mah.models import CosmologySet, GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN, KM_PER_MPC, SECONDS_PER_GYR
from sfr.calculator import EXTENDED_BURST_KAPPA

plt.style.use("apj")

OUTPUT_DIR = Path("outputs")
DEFAULT_OUTPUT_STEM = "extended_burst_kernel_tdyn"


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


def burst_kernel(delta_t_myr: np.ndarray, td_myr: float, kappa: float = EXTENDED_BURST_KAPPA) -> np.ndarray:
    x = np.asarray(delta_t_myr, dtype=float)
    kernel = x / (kappa**2 * td_myr**2) * np.exp(-x / (kappa * td_myr))
    kernel[x < 0.0] = 0.0
    return kernel


def build_output_paths(output_stem: str) -> tuple[Path, Path, Path]:
    return (
        OUTPUT_DIR / f"{output_stem}.png",
        OUTPUT_DIR / f"{output_stem}.pdf",
        OUTPUT_DIR / f"{output_stem}.txt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the extended-burst delay kernel g(Δt).")
    parser.add_argument(
        "--kappa",
        type=float,
        default=EXTENDED_BURST_KAPPA,
        help="Dimensionless burst-duration parameter κ.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Optional output filename stem under outputs/. Defaults to a κ-tagged stem when κ != 1.",
    )
    parser.add_argument(
        "--kernel-xmax-myr",
        type=float,
        default=None,
        help="Optional x-axis upper limit in Myr for the kernel panel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kappa = float(args.kappa)
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("kappa must be a positive finite number")
    kernel_xmax_myr = args.kernel_xmax_myr
    if kernel_xmax_myr is not None and (not np.isfinite(kernel_xmax_myr) or kernel_xmax_myr <= 0.0):
        raise ValueError("kernel-xmax-myr must be a positive finite number")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_stem
    if output_stem is None:
        if np.isclose(kappa, EXTENDED_BURST_KAPPA):
            output_stem = DEFAULT_OUTPUT_STEM
        else:
            output_stem = f"{DEFAULT_OUTPUT_STEM}_kappa{str(kappa).replace('.', 'p')}"
    png_path, pdf_path, txt_path = build_output_paths(output_stem)

    z_examples = [6.0, 12.0, 20.0]
    colors = ["#1f4e79", "#c0392b", "#1d8348"]
    td_examples = {z: float(tdyn_y16_myr(np.array([z]))[0]) for z in z_examples}

    max_td = max(td_examples.values())
    delta_t = np.linspace(0.0, 6.0 * max_td, 1200)

    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    for z, color in zip(z_examples, colors, strict=True):
        td = td_examples[z]
        g = burst_kernel(delta_t, td, kappa=kappa)
        ax.plot(delta_t, g, color=color, linewidth=2.2, label=rf"$z_f={z:g}$, $t_d={td:.1f}\,\rm Myr$")
        ax.axvline(kappa * td, color=color, linestyle=":", alpha=0.8, linewidth=1.0)
    ax.set_xlabel(r"$\Delta t = t-t'$ [Myr]")
    ax.set_ylabel(r"$g(\Delta t)$ [Myr$^{-1}$]")
    if kernel_xmax_myr is not None:
        ax.set_xlim(0.0, kernel_xmax_myr)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)

    lines = [
        "Extended-burst delay kernel diagnostics",
        "g(Δt) = Δt / (κ^2 t_d^2) * exp[-Δt / (κ t_d)]",
        f"For κ = {kappa:g}: mode = {kappa:g} t_d, mean = {2.0 * kappa:g} t_d, integral = 1",
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
                f"z={z:g}: kernel_mode = {kappa * td:.6f} Myr",
                f"z={z:g}: kernel_mean = {2.0 * kappa * td:.6f} Myr",
            ]
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(png_path.resolve())
    print(pdf_path.resolve())
    print(txt_path.resolve())


if __name__ == "__main__":
    main()
