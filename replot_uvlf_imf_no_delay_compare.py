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
        description="Plot the no-delay UVLF comparison between the legacy IMF SSP and the top-heavy SSP."
    )
    parser.add_argument(
        "--npz-path",
        type=str,
        default="data_save/uvlf_imf_no_delay_compare_allz_20260324_140817.npz",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outputs/uvlf_imf_no_delay_compare_allz_20260324_140817",
    )
    return parser.parse_args()


def _z_tag(z_value: float) -> str:
    return f"z{str(float(z_value)).replace('.', 'p')}"


def _obs_dir_from_z(z_value: float) -> Path:
    return Path("obsdata") / f"redshift_{str(float(z_value)).replace('.', 'p').rstrip('0').rstrip('p')}"


def _load_observational_uvlf(z_value: float) -> list[dict[str, np.ndarray | str]]:
    obs_dir = _obs_dir_from_z(z_value)
    datasets: list[dict[str, np.ndarray | str]] = []
    for file_path in sorted(obs_dir.glob("*.npz")):
        payload = np.load(file_path)
        label_array = np.asarray(payload["label"])
        label = str(label_array[0]) if label_array.size > 0 else file_path.stem
        datasets.append(
            {
                "label": label,
                "Muv": np.asarray(payload["muverr"], dtype=float),
                "phi": np.asarray(payload["phierr"], dtype=float),
                "mag_err": np.asarray(payload["mag_err"], dtype=float),
                "phi_err_lo": np.asarray(payload["phi_err_lo"], dtype=float),
                "phi_err_up": np.asarray(payload["phi_err_up"], dtype=float),
            }
        )
    return datasets


def main() -> None:
    args = _parse_args()
    npz_path = Path(args.npz_path).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    z_values = [float(z) for z in np.asarray(data["z_values"], dtype=float)]
    old_ssp_file = str(np.asarray(data["old_ssp_file"])[0])
    topheavy_ssp_file = str(np.asarray(data["topheavy_ssp_file"])[0])

    fig, axes = plt.subplots(
        2,
        len(z_values),
        figsize=(5.2 * len(z_values), 7.6),
        constrained_layout=True,
        sharex="col",
    )
    if len(z_values) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    summary_lines = [
        f"npz_path: {npz_path}",
        f"old_ssp_file: {old_ssp_file}",
        f"topheavy_ssp_file: {topheavy_ssp_file}",
        "",
    ]
    obs_markers = ["o", "s", "^", "D", "P", "X"]

    for column, z_obs in enumerate(z_values):
        tag = _z_tag(z_obs)
        centers = np.asarray(data[f"{tag}_bin_centers"], dtype=float)
        phi_old = np.asarray(data[f"{tag}_old_phi"], dtype=float)
        phi_top = np.asarray(data[f"{tag}_topheavy_phi"], dtype=float)
        ratio = np.asarray(data[f"{tag}_phi_ratio_topheavy_over_old"], dtype=float)

        ax_top = axes[0, column]
        valid_old = np.isfinite(phi_old) & (phi_old > 0.0)
        valid_top = np.isfinite(phi_top) & (phi_top > 0.0)
        ax_top.plot(centers[valid_old], phi_old[valid_old], color="black", lw=2.2, label="legacy IMF")
        ax_top.plot(centers[valid_top], phi_top[valid_top], color="#c44e52", lw=2.2, label="top-heavy IMF")
        obs_sets = _load_observational_uvlf(z_obs)
        for obs_index, obs in enumerate(obs_sets):
            marker = obs_markers[obs_index % len(obs_markers)]
            muv = np.asarray(obs["Muv"], dtype=float)
            phi = np.asarray(obs["phi"], dtype=float)
            mag_err = np.asarray(obs["mag_err"], dtype=float)
            phi_err_lo = np.asarray(obs["phi_err_lo"], dtype=float)
            phi_err_up = np.asarray(obs["phi_err_up"], dtype=float)
            valid = np.isfinite(muv) & np.isfinite(phi) & (phi > 0.0)
            if not np.any(valid):
                continue
            ax_top.errorbar(
                muv[valid],
                phi[valid],
                xerr=mag_err[valid],
                yerr=np.vstack([phi_err_lo[valid], phi_err_up[valid]]),
                fmt=marker,
                ms=5.5,
                color="#1f4e79",
                mec="white",
                mew=0.6,
                elinewidth=1.0,
                capsize=2.0,
                alpha=0.92,
                label=str(obs["label"]),
            )
        ax_top.set_yscale("log")
        ax_top.set_xlim(-24.5, -15.0)
        ax_top.set_ylim(1.0e-7, 5.0e-1)
        ax_top.grid(alpha=0.22)
        ax_top.set_title(rf"$z={z_obs:g}$")
        if column == 0:
            ax_top.set_ylabel(r"$\phi(M_{\rm UV})$ [dex$^{-1}$ Mpc$^{-3}$]")
        ax_top.legend(frameon=False, fontsize=7.8, loc="lower left")

        ax_bottom = axes[1, column]
        valid_ratio = np.isfinite(ratio) & (ratio > 0.0)
        ax_bottom.plot(centers[valid_ratio], ratio[valid_ratio], color="#1f77b4", lw=2.1)
        ax_bottom.axhline(1.0, color="0.35", ls="--", lw=1.0)
        ax_bottom.set_xlim(-24.5, -15.0)
        ax_bottom.set_ylim(0.8, max(3.2, float(np.nanmax(ratio[valid_ratio])) * 1.08 if np.any(valid_ratio) else 3.2))
        ax_bottom.grid(alpha=0.22)
        ax_bottom.set_xlabel(r"$M_{\rm UV}$")
        if column == 0:
            ax_bottom.set_ylabel("top-heavy / legacy")

        overlap = np.isfinite(ratio) & np.isfinite(phi_old) & np.isfinite(phi_top)
        summary_lines.append(f"z={z_obs:g}")
        if np.any(overlap):
            summary_lines.append(f"  phi_ratio_median={float(np.nanmedian(ratio[overlap])):.6f}")
            summary_lines.append(f"  phi_ratio_min={float(np.nanmin(ratio[overlap])):.6f}")
            summary_lines.append(f"  phi_ratio_max={float(np.nanmax(ratio[overlap])):.6f}")
            bright = overlap & (centers <= -20.0)
            mid = overlap & (centers > -20.0) & (centers <= -18.0)
            faint = overlap & (centers > -18.0)
            if np.any(bright):
                summary_lines.append(f"  bright_ratio_median={float(np.nanmedian(ratio[bright])):.6f}")
            if np.any(mid):
                summary_lines.append(f"  mid_ratio_median={float(np.nanmedian(ratio[mid])):.6f}")
            if np.any(faint):
                summary_lines.append(f"  faint_ratio_median={float(np.nanmedian(ratio[faint])):.6f}")
        summary_lines.append(f"  n_obs_sets={len(obs_sets)}")
        summary_lines.append("")

    fig.suptitle("No-delay UVLF comparison with observational UVLF points", fontsize=16)

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    txt_path = output_prefix.parent / f"{output_prefix.name}_plot_summary.txt"
    fig.savefig(png_path, dpi=250)
    fig.savefig(pdf_path)
    plt.close(fig)
    txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")
    print(f"saved_txt={txt_path}")


if __name__ == "__main__":
    main()
