#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _try_use_apj_style() -> None:
    try:
        plt.style.use("apj")
    except OSError:
        pass


def _tag_from_z(z_value: float) -> str:
    return f"z{str(float(z_value)).replace('.', 'p')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot halo-mass and SFR comparison from saved NPZ.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data_save/delay_sfr_mh_four_z_compare_20260323_n480.npz",
    )
    parser.add_argument("--z-values", nargs="+", type=float, required=True)
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    _try_use_apj_style()

    data = np.load(Path(args.data_path).expanduser().resolve())
    z_values = [float(z) for z in args.z_values]
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    mh_final = float(data["mh_final"][0])
    lookback_max_myr = float(data["lookback_max_myr"][0])

    fig, axes = plt.subplots(
        2,
        len(z_values),
        figsize=(5.2 * len(z_values), 7.2),
        constrained_layout=True,
        sharex="col",
    )
    if len(z_values) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for column, z_obs in enumerate(z_values):
        tag = _tag_from_z(z_obs)
        lookback = np.asarray(data[f"{tag}_no_delay_lookback_myr"], dtype=float)
        recent_mask = np.asarray(data[f"{tag}_no_delay_recent_mask"], dtype=bool)

        mh_p16 = np.asarray(data[f"{tag}_no_delay_mh_p16"], dtype=float)
        mh_p50 = np.asarray(data[f"{tag}_no_delay_mh_p50"], dtype=float)
        mh_p84 = np.asarray(data[f"{tag}_no_delay_mh_p84"], dtype=float)

        no_p16 = np.asarray(data[f"{tag}_no_delay_sfr_p16"], dtype=float)
        no_p50 = np.asarray(data[f"{tag}_no_delay_sfr_p50"], dtype=float)
        no_p84 = np.asarray(data[f"{tag}_no_delay_sfr_p84"], dtype=float)
        de_p16 = np.asarray(data[f"{tag}_delay_sfr_p16"], dtype=float)
        de_p50 = np.asarray(data[f"{tag}_delay_sfr_p50"], dtype=float)
        de_p84 = np.asarray(data[f"{tag}_delay_sfr_p84"], dtype=float)

        ax_mass = axes[0, column]
        mass_mask = recent_mask & np.isfinite(mh_p16) & np.isfinite(mh_p50) & np.isfinite(mh_p84)
        ax_mass.fill_between(lookback[mass_mask], mh_p16[mass_mask], mh_p84[mass_mask], color="0.5", alpha=0.18)
        ax_mass.plot(lookback[mass_mask], mh_p50[mass_mask], color="black", lw=2.0, label=r"$M_{\rm h}(t)$")
        ax_mass.set_yscale("log")
        ax_mass.set_xlim(lookback_max_myr, 0.0)
        ax_mass.grid(alpha=0.22)
        ax_mass.set_title(f"z = {z_obs:g}")
        if column == 0:
            ax_mass.set_ylabel(r"$M_{\rm h}\ [{\rm M_\odot}]$")
            ax_mass.legend(frameon=False, fontsize=10, loc="lower left")

        ax_sfr = axes[1, column]
        no_mask = recent_mask & np.isfinite(no_p16) & np.isfinite(no_p50) & np.isfinite(no_p84)
        ax_sfr.fill_between(lookback[no_mask], no_p16[no_mask], no_p84[no_mask], color="black", alpha=0.12)
        ax_sfr.plot(lookback[no_mask], no_p50[no_mask], color="black", lw=2.0, label="no delay")
        de_mask = recent_mask & np.isfinite(de_p16) & np.isfinite(de_p50) & np.isfinite(de_p84)
        ax_sfr.fill_between(lookback[de_mask], de_p16[de_mask], de_p84[de_mask], color="#c44e52", alpha=0.16)
        ax_sfr.plot(lookback[de_mask], de_p50[de_mask], color="#c44e52", lw=2.0, label="delay")
        ax_sfr.set_yscale("log")
        ax_sfr.set_xlim(lookback_max_myr, 0.0)
        ax_sfr.grid(alpha=0.22)
        ax_sfr.set_xlabel("Lookback time before observation [Myr]")
        if column == 0:
            ax_sfr.set_ylabel(r"${\rm SFR}\ [{\rm M_\odot\,yr^{-1}}]$")
            ax_sfr.legend(frameon=False, fontsize=10, loc="lower left")

    fig.suptitle(
        f"Halo-mass and SFR history at fixed Mh,final = {mh_final:.0e} Msun",
        fontsize=16,
    )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")


if __name__ == "__main__":
    main()
