#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot UVLF delay-comparison results from saved NPZ files."
    )
    parser.add_argument(
        "--npz-240",
        type=str,
        default="data_save/uvlf_delay_effect_compare_allz_fat_20260322_100myr_uniformt_bugfix_n240.npz",
    )
    parser.add_argument(
        "--npz-480",
        type=str,
        default="data_save/uvlf_delay_effect_compare_allz_fat_20260322_100myr_uniformt_bugfix_n480.npz",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outputs/uvlf_delay_effect_compare_allz_fat_20260322_100myr_uniformt_bugfix_replot",
    )
    return parser.parse_args()


def _tag_from_z(z_value: float) -> str:
    return f"z{str(float(z_value)).replace('.', 'p')}"


def _load_case(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    result: dict[str, np.ndarray] = {"z_values": np.asarray(data["z_values"], dtype=float)}
    for key in data.files:
        if key == "z_values":
            continue
        result[key] = np.asarray(data[key])
    return result


def main() -> None:
    args = _parse_args()
    npz_240_path = Path(args.npz_240).expanduser().resolve()
    npz_480_path = Path(args.npz_480).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    data_240 = _load_case(npz_240_path)
    data_480 = _load_case(npz_480_path)
    z_values = np.asarray(data_240["z_values"], dtype=float)

    fig, axes = plt.subplots(
        2,
        z_values.size,
        figsize=(5.0 * z_values.size, 7.2),
        constrained_layout=True,
        sharex="col",
    )
    if z_values.size == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for column, z_obs in enumerate(z_values):
        tag = _tag_from_z(float(z_obs))

        centers_240 = np.asarray(data_240[f"{tag}_no_delay_bin_centers"], dtype=float)
        phi_240_no = np.asarray(data_240[f"{tag}_no_delay_phi"], dtype=float)
        phi_240_delay = np.asarray(data_240[f"{tag}_delay_phi"], dtype=float)
        ratio_240 = np.divide(
            phi_240_delay,
            phi_240_no,
            out=np.full_like(phi_240_no, np.nan),
            where=phi_240_no > 0.0,
        )

        centers_480 = np.asarray(data_480[f"{tag}_no_delay_bin_centers"], dtype=float)
        phi_480_no = np.asarray(data_480[f"{tag}_no_delay_phi"], dtype=float)
        phi_480_delay = np.asarray(data_480[f"{tag}_delay_phi"], dtype=float)
        ratio_480 = np.divide(
            phi_480_delay,
            phi_480_no,
            out=np.full_like(phi_480_no, np.nan),
            where=phi_480_no > 0.0,
        )

        ax_top = axes[0, column]
        ax_top.plot(centers_240, phi_240_no, color="black", lw=1.8, ls="--", label="no delay, n=240")
        ax_top.plot(centers_240, phi_240_delay, color="#c44e52", lw=1.8, ls="--", label="delay, n=240")
        ax_top.plot(centers_480, phi_480_no, color="black", lw=2.2, ls="-", label="no delay, n=480")
        ax_top.plot(centers_480, phi_480_delay, color="#c44e52", lw=2.2, ls="-", label="delay, n=480")
        ax_top.set_yscale("log")
        ax_top.set_xlim(-25.5, -10.0)
        ax_top.set_ylim(1.0e-8, 1.0)
        ax_top.grid(alpha=0.22)
        ax_top.set_title(f"z = {z_obs:g}")
        if column == 0:
            ax_top.set_ylabel(r"$\phi(M_{\rm UV})$")
            ax_top.legend(fontsize=9, frameon=False, loc="lower left")

        ax_bottom = axes[1, column]
        ax_bottom.plot(centers_240, ratio_240, color="#1f77b4", lw=1.8, ls="--", label="n=240")
        ax_bottom.plot(centers_480, ratio_480, color="#ff7f0e", lw=2.0, ls="-", label="n=480")
        ax_bottom.axhline(1.0, color="0.35", ls="--", lw=1.0)
        ax_bottom.set_xlim(-25.5, -10.0)
        ax_bottom.set_ylim(0.2, 1.12)
        ax_bottom.grid(alpha=0.22)
        ax_bottom.set_xlabel(r"$M_{\rm UV}$")
        if column == 0:
            ax_bottom.set_ylabel("delay / no delay")
            ax_bottom.legend(fontsize=9, frameon=False, loc="lower right")

    fig.suptitle("Bugfix replot: UVLF delay effect, n_grid=240 vs 480", fontsize=16)

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")


if __name__ == "__main__":
    main()
