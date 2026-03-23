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
        description="Replot saved UVLF delay/no-delay curves and overlay observational UVLF points."
    )
    parser.add_argument(
        "--npz-path",
        type=str,
        default="data_save/uvlf_delay_effect_compare_allz_fat_20260322_100myr_uniformt_bugfix_n480.npz",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outputs/uvlf_delay_effect_compare_allz_fat_20260322_100myr_uniformt_bugfix_n480_with_obs",
    )
    parser.add_argument(
        "--z-values",
        nargs="+",
        type=float,
        default=[6.0, 8.0, 10.0, 12.5],
    )
    return parser.parse_args()


def _tag_from_z(z_value: float) -> str:
    return f"z{str(float(z_value)).replace('.', 'p')}"


def _obs_dir_from_z(z_value: float) -> Path:
    return Path("obsdata") / f"redshift_{str(float(z_value)).replace('.', 'p').rstrip('0').rstrip('p')}"


def _load_saved_uvlf(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    payload: dict[str, np.ndarray] = {}
    for key in data.files:
        payload[key] = np.asarray(data[key])
    return payload


def _load_observational_uvlf(z_value: float) -> list[dict[str, np.ndarray | str]]:
    obs_dir = _obs_dir_from_z(z_value)
    datasets: list[dict[str, np.ndarray | str]] = []
    for file_path in sorted(obs_dir.glob("*.npz")):
        data = np.load(file_path)
        label_array = np.asarray(data["label"])
        label = str(label_array[0]) if label_array.size > 0 else file_path.stem
        datasets.append(
            {
                "label": label,
                "Muv": np.asarray(data["muverr"], dtype=float),
                "phi": np.asarray(data["phierr"], dtype=float),
                "mag_err": np.asarray(data["mag_err"], dtype=float),
                "phi_err_lo": np.asarray(data["phi_err_lo"], dtype=float),
                "phi_err_up": np.asarray(data["phi_err_up"], dtype=float),
            }
        )
    return datasets


def main() -> None:
    args = _parse_args()
    npz_path = Path(args.npz_path).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    saved = _load_saved_uvlf(npz_path)
    z_values = [float(z) for z in args.z_values]
    colors = {
        "no_delay": "black",
        "delay": "#c44e52",
    }
    obs_markers = ["o", "s", "^", "D", "P", "X"]

    fig, axes = plt.subplots(
        2,
        len(z_values),
        figsize=(5.2 * len(z_values), 7.8),
        constrained_layout=True,
        sharex="col",
    )
    if len(z_values) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    summary_lines: list[str] = [
        f"npz_path: {npz_path}",
        f"z_values: {' '.join(str(z) for z in z_values)}",
        "",
    ]

    for column, z_obs in enumerate(z_values):
        tag = _tag_from_z(z_obs)
        centers = np.asarray(saved[f"{tag}_no_delay_bin_centers"], dtype=float)
        phi_no = np.asarray(saved[f"{tag}_no_delay_phi"], dtype=float)
        phi_delay = np.asarray(saved[f"{tag}_delay_phi"], dtype=float)
        ratio = np.divide(
            phi_delay,
            phi_no,
            out=np.full_like(phi_no, np.nan),
            where=phi_no > 0.0,
        )

        ax_top = axes[0, column]
        ax_top.plot(centers, phi_no, color=colors["no_delay"], lw=2.2, label="no delay")
        ax_top.plot(centers, phi_delay, color=colors["delay"], lw=2.2, label="delay")

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
        ax_top.legend(fontsize=7.8, frameon=False, loc="lower left")

        ax_bottom = axes[1, column]
        valid_ratio = np.isfinite(ratio) & (ratio > 0.0)
        ax_bottom.plot(centers[valid_ratio], ratio[valid_ratio], color="#1f77b4", lw=2.1)
        ax_bottom.axhline(1.0, color="0.35", ls="--", lw=1.0)
        ax_bottom.set_xlim(-24.5, -15.0)
        ax_bottom.set_ylim(0.2, 1.15)
        ax_bottom.grid(alpha=0.22)
        ax_bottom.set_xlabel(r"$M_{\rm UV}$")
        if column == 0:
            ax_bottom.set_ylabel("delay / no delay")

        overlap = np.isfinite(phi_no) & np.isfinite(phi_delay) & (phi_no > 0.0) & (phi_delay > 0.0)
        summary_lines.append(f"z={z_obs:g}")
        if np.any(overlap):
            summary_lines.append(f"  ratio_median={float(np.nanmedian(ratio[overlap])):.6f}")
            bright = overlap & (centers <= -20.0)
            mid = overlap & (centers > -20.0) & (centers <= -18.0)
            faint = overlap & (centers > -18.0)
            if np.any(bright):
                summary_lines.append(f"  ratio_bright_median={float(np.nanmedian(ratio[bright])):.6f}")
            if np.any(mid):
                summary_lines.append(f"  ratio_mid_median={float(np.nanmedian(ratio[mid])):.6f}")
            if np.any(faint):
                summary_lines.append(f"  ratio_faint_median={float(np.nanmedian(ratio[faint])):.6f}")
        summary_lines.append(f"  n_obs_sets={len(obs_sets)}")
        summary_lines.append("")

    fig.suptitle("Saved UVLF curves with observational UVLF points", fontsize=16)

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    txt_path = output_prefix.with_suffix(".txt")
    fig.savefig(png_path, dpi=250)
    fig.savefig(pdf_path)
    plt.close(fig)
    txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")
    print(f"saved_txt={txt_path}")


if __name__ == "__main__":
    main()
