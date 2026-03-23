#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sfr.calculator import EXTENDED_BURST_KAPPA
from uvlf import run_halo_uv_pipeline, uv_luminosity_to_muv


def _burst_kernel(delta_t_gyr: np.ndarray, td_gyr: float, kappa: float = EXTENDED_BURST_KAPPA) -> np.ndarray:
    delta_t_gyr = np.asarray(delta_t_gyr, dtype=float)
    kernel = np.zeros_like(delta_t_gyr, dtype=float)
    valid = delta_t_gyr >= 0.0
    x = delta_t_gyr[valid]
    kernel[valid] = x / (kappa**2 * td_gyr**2) * np.exp(-x / (kappa * td_gyr))
    return kernel


def _try_use_apj_style() -> None:
    try:
        plt.style.use("apj")
    except OSError:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose whether the delay kernel is resolved on the current time grid.")
    parser.add_argument("--z-values", nargs="+", type=float, default=[6.0, 12.5])
    parser.add_argument("--n-grid-values", nargs="+", type=int, default=[80, 240, 480, 960])
    parser.add_argument("--mh-final", type=float, default=1.0e11)
    parser.add_argument("--n-tracks", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outputs/uvlf_delay_resolution_diagnostic_20260322",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data_save/uvlf_delay_resolution_diagnostic_20260322.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path).expanduser().resolve()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    _try_use_apj_style()

    z_values = [float(z) for z in args.z_values]
    n_grid_values = [int(n) for n in args.n_grid_values]
    if len(z_values) != 2:
        raise ValueError("use exactly two z values so the figure stays readable")

    colors = {z_values[0]: "#1f4e79", z_values[1]: "#c44e52"}
    labels = {z: rf"$z={z:g}$" for z in z_values}
    summary_lines = [
        "Delay-kernel resolution diagnostic",
        f"kappa: {EXTENDED_BURST_KAPPA}",
        f"mh_final: {args.mh_final:.6e}",
        f"n_tracks: {args.n_tracks}",
        f"seed: {args.seed}",
        "",
    ]
    compact: dict[str, np.ndarray] = {
        "z_values": np.asarray(z_values, dtype=float),
        "n_grid_values": np.asarray(n_grid_values, dtype=int),
        "mh_final": np.array([float(args.mh_final)], dtype=float),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), constrained_layout=True)
    ax_kernel = axes[0, 0]
    ax_integral = axes[0, 1]
    ax_sfr = axes[1, 0]
    ax_muv = axes[1, 1]

    for z_obs in z_values:
        kernel_integrals = []
        dt_last10 = []
        sfr_ratio_median = []
        delta_muv_median = []

        for n_grid in n_grid_values:
            result_no = run_halo_uv_pipeline(
                n_tracks=int(args.n_tracks),
                z_final=z_obs,
                Mh_final=float(args.mh_final),
                n_grid=n_grid,
                random_seed=int(args.seed),
                enable_time_delay=False,
                workers=1,
            )
            result_delay = run_halo_uv_pipeline(
                n_tracks=int(args.n_tracks),
                z_final=z_obs,
                Mh_final=float(args.mh_final),
                n_grid=n_grid,
                random_seed=int(args.seed),
                enable_time_delay=True,
                workers=1,
            )

            t_grid = np.asarray(result_no.sfr_tracks["t_gyr"], dtype=float).reshape(int(args.n_tracks), n_grid)
            td_grid = np.asarray(result_no.sfr_tracks["td_burst"], dtype=float).reshape(int(args.n_tracks), n_grid)
            active_grid = np.asarray(result_no.sfr_tracks["active_flag"], dtype=bool).reshape(int(args.n_tracks), n_grid)
            sfr_no = np.asarray(result_no.sfr_tracks["SFR"], dtype=float).reshape(int(args.n_tracks), n_grid)
            sfr_delay = np.asarray(result_delay.sfr_tracks["SFR"], dtype=float).reshape(int(args.n_tracks), n_grid)

            time_row = t_grid[0]
            td_final = float(td_grid[0, -1])
            lookback = time_row[-1] - time_row
            kernel = _burst_kernel(lookback, td_final)
            kernel_integral = float(np.trapezoid(kernel, x=time_row))
            kernel_integrals.append(kernel_integral)
            dt_last10.append(float(np.median(np.diff(time_row)[-10:]) * 1.0e3))

            final_index = np.max(np.where(active_grid, np.arange(n_grid, dtype=int)[None, :], -1), axis=1)
            ratios = []
            for halo_id, idx in enumerate(final_index):
                if idx >= 0 and sfr_no[halo_id, idx] > 0.0:
                    ratios.append(sfr_delay[halo_id, idx] / sfr_no[halo_id, idx])
            sfr_ratio_median.append(float(np.nanmedian(np.asarray(ratios, dtype=float))))

            muv_no = np.asarray(uv_luminosity_to_muv(result_no.uv_luminosities), dtype=float)
            muv_delay = np.asarray(uv_luminosity_to_muv(result_delay.uv_luminosities), dtype=float)
            delta_muv_median.append(float(np.nanmedian(muv_delay - muv_no)))

            summary_lines.append(
                f"z={z_obs:g}, n_grid={n_grid}: "
                f"kernel_integral={kernel_integral:.6f}, "
                f"median_last10_dt_myr={dt_last10[-1]:.6f}, "
                f"kappa_td_myr={EXTENDED_BURST_KAPPA * td_final * 1.0e3:.6f}, "
                f"median_final_SFR_ratio={sfr_ratio_median[-1]:.6f}, "
                f"median_delta_Muv={delta_muv_median[-1]:.6f}"
            )

            if n_grid == n_grid_values[1]:
                dense_lookback_myr = np.linspace(0.0, 120.0, 600, dtype=float)
                dense_kernel = _burst_kernel(dense_lookback_myr / 1.0e3, td_final)
                sample_mask = (lookback * 1.0e3) <= 120.0
                ax_kernel.plot(dense_lookback_myr, dense_kernel, color=colors[z_obs], lw=2.2, label=labels[z_obs])
                ax_kernel.scatter(
                    (lookback[sample_mask] * 1.0e3),
                    kernel[sample_mask],
                    color=colors[z_obs],
                    s=20,
                    alpha=0.9,
                )

        z_key = f"z{str(z_obs).replace('.', 'p')}"
        compact[f"{z_key}_kernel_integrals"] = np.asarray(kernel_integrals, dtype=float)
        compact[f"{z_key}_median_last10_dt_myr"] = np.asarray(dt_last10, dtype=float)
        compact[f"{z_key}_median_final_sfr_ratio"] = np.asarray(sfr_ratio_median, dtype=float)
        compact[f"{z_key}_median_delta_muv"] = np.asarray(delta_muv_median, dtype=float)

        ax_integral.plot(n_grid_values, kernel_integrals, marker="o", color=colors[z_obs], lw=2.2, label=labels[z_obs])
        ax_sfr.plot(n_grid_values, sfr_ratio_median, marker="o", color=colors[z_obs], lw=2.2, label=labels[z_obs])
        ax_muv.plot(n_grid_values, delta_muv_median, marker="o", color=colors[z_obs], lw=2.2, label=labels[z_obs])

    ax_kernel.set_xlabel(r"Lookback $\Delta t$ [Myr]")
    ax_kernel.set_ylabel(r"$g(\Delta t)$ [Gyr$^{-1}$]")
    ax_kernel.set_xlim(0.0, 120.0)
    ax_kernel.set_title(rf"Kernel shape and the actual $n_{{\rm grid}}={n_grid_values[1]}$ sample points")
    ax_kernel.grid(alpha=0.25)
    ax_kernel.legend(frameon=False)

    ax_integral.axhline(1.0, color="0.35", ls="--", lw=1.2)
    ax_integral.set_xscale("log")
    ax_integral.set_xlabel(r"$n_{\rm grid}$")
    ax_integral.set_ylabel(r"$\int g(\Delta t)\,d\Delta t$ on the discrete grid")
    ax_integral.set_ylim(0.0, 1.08)
    ax_integral.set_title("If this is far below 1, the kernel is under-resolved")
    ax_integral.grid(alpha=0.25)
    ax_integral.legend(frameon=False)

    ax_sfr.set_xscale("log")
    ax_sfr.set_xlabel(r"$n_{\rm grid}$")
    ax_sfr.set_ylabel(r"Median ${\rm SFR}_{\rm delay}/{\rm SFR}_{\rm no\,delay}$")
    ax_sfr.set_ylim(0.0, 1.02)
    ax_sfr.set_title(rf"Final-SFR ratio at fixed $M_h={args.mh_final:.0e}\,M_\odot$")
    ax_sfr.grid(alpha=0.25)
    ax_sfr.legend(frameon=False)

    ax_muv.set_xscale("log")
    ax_muv.set_xlabel(r"$n_{\rm grid}$")
    ax_muv.set_ylabel(r"Median $\Delta M_{\rm UV}$")
    ax_muv.set_title(rf"Median UV dimming at fixed $M_h={args.mh_final:.0e}\,M_\odot$")
    ax_muv.grid(alpha=0.25)
    ax_muv.legend(frameon=False)

    summary_lines.extend(
        [
            "",
            "note_1: top-left compares the continuous kernel to the actual lookback samples used by the code",
            "note_2: top-right should be close to 1 if the delay kernel is numerically resolved",
            "note_3: bottom panels show how much the physics result moves when only n_grid changes",
        ]
    )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    txt_path = output_prefix.with_suffix(".txt")
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)

    txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    np.savez_compressed(data_path, **compact)

    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")
    print(f"saved_txt={txt_path}")
    print(f"saved_npz={data_path}")


if __name__ == "__main__":
    main()
