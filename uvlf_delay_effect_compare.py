#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sfr.calculator import DEFAULT_SFR_MODEL_PARAMETERS
from uvlf.hmf_sampling import UVLFSamplingResult, sample_uvlf_from_hmf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare UVLFs with and without the extended-burst time-delay branch."
    )
    parser.add_argument("--z-values", nargs="+", type=float, default=[6.0, 12.0])
    parser.add_argument("--n-mass", type=int, default=800)
    parser.add_argument("--n-tracks", type=int, default=300)
    parser.add_argument("--n-grid", type=int, default=180)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outputs/uvlf_delay_effect_compare",
    )
    return parser.parse_args()


def _run_case(
    z_obs: float,
    enable_time_delay: bool,
    n_mass: int,
    n_tracks: int,
    n_grid: int,
    workers: int,
    seed: int,
    progress_path: Path,
) -> UVLFSamplingResult:
    bins = np.arange(-26.0, -9.5, 0.5, dtype=float)
    return sample_uvlf_from_hmf(
        z_obs=z_obs,
        N_mass=n_mass,
        n_tracks=n_tracks,
        random_seed=seed,
        quantity="Muv",
        bins=bins,
        n_grid=n_grid,
        enable_time_delay=enable_time_delay,
        pipeline_workers=workers,
        progress_path=progress_path,
        sfr_model_parameters=DEFAULT_SFR_MODEL_PARAMETERS,
    )


def _write_summary(
    summary_path: Path,
    z_values: list[float],
    no_delay_results: list[UVLFSamplingResult],
    delay_results: list[UVLFSamplingResult],
) -> None:
    lines = []
    for z_obs, no_delay, delay in zip(z_values, no_delay_results, delay_results, strict=True):
        phi_no = np.asarray(no_delay.uvlf["phi"], dtype=float)
        phi_delay = np.asarray(delay.uvlf["phi"], dtype=float)
        centers = np.asarray(no_delay.uvlf["bin_centers"], dtype=float)
        ratio = np.divide(phi_delay, phi_no, out=np.full_like(phi_no, np.nan), where=phi_no > 0.0)
        valid = np.isfinite(ratio)
        lines.append(f"z={z_obs:g}")
        if np.any(valid):
            lines.append(f"  median(phi_delay/phi_no_delay) = {float(np.nanmedian(ratio[valid])):.4f}")
            bright = valid & (centers <= -20.0)
            mid = valid & (centers > -20.0) & (centers <= -18.0)
            faint = valid & (centers > -18.0)
            if np.any(bright):
                lines.append(f"  bright(Muv<=-20) median ratio = {float(np.nanmedian(ratio[bright])):.4f}")
            if np.any(mid):
                lines.append(f"  mid(-20<Muv<=-18) median ratio = {float(np.nanmedian(ratio[mid])):.4f}")
            if np.any(faint):
                lines.append(f"  faint(Muv>-18) median ratio = {float(np.nanmedian(ratio[faint])):.4f}")
        else:
            lines.append("  no valid overlapping bins")
        lines.append(
            "  timing_no_delay="
            f"{float(no_delay.metadata['sampling_seconds']):.2f}s "
            "timing_delay="
            f"{float(delay.metadata['sampling_seconds']):.2f}s"
        )
        lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    z_values = [float(value) for value in args.z_values]
    no_delay_results: list[UVLFSamplingResult] = []
    delay_results: list[UVLFSamplingResult] = []

    for z_obs in z_values:
        base_name = f"{output_prefix.name}_z{str(z_obs).replace('.', 'p')}"
        no_delay_results.append(
            _run_case(
                z_obs=z_obs,
                enable_time_delay=False,
                n_mass=args.n_mass,
                n_tracks=args.n_tracks,
                n_grid=args.n_grid,
                workers=args.workers,
                seed=args.seed,
                progress_path=output_prefix.parent / f"{base_name}_no_delay_progress.txt",
            )
        )
        delay_results.append(
            _run_case(
                z_obs=z_obs,
                enable_time_delay=True,
                n_mass=args.n_mass,
                n_tracks=args.n_tracks,
                n_grid=args.n_grid,
                workers=args.workers,
                seed=args.seed,
                progress_path=output_prefix.parent / f"{base_name}_delay_progress.txt",
            )
        )

    fig, axes = plt.subplots(
        2,
        len(z_values),
        figsize=(6.0 * len(z_values), 8.0),
        constrained_layout=True,
        sharex="col",
    )
    if len(z_values) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for column, (z_obs, no_delay, delay) in enumerate(
        zip(z_values, no_delay_results, delay_results, strict=True)
    ):
        centers = np.asarray(no_delay.uvlf["bin_centers"], dtype=float)
        phi_no = np.asarray(no_delay.uvlf["phi"], dtype=float)
        phi_delay = np.asarray(delay.uvlf["phi"], dtype=float)
        ratio = np.divide(phi_delay, phi_no, out=np.full_like(phi_no, np.nan), where=phi_no > 0.0)

        ax_top = axes[0, column]
        ax_top.plot(centers, phi_no, color="black", lw=2.2, label="No delay")
        ax_top.plot(centers, phi_delay, color="#c44e52", lw=2.2, label="Extended burst")
        ax_top.set_yscale("log")
        ax_top.set_ylim(1.0e-8, 1.0)
        ax_top.set_xlim(-25.5, -10.0)
        ax_top.grid(alpha=0.25)
        ax_top.set_title(f"z = {z_obs:g}")
        if column == 0:
            ax_top.set_ylabel(r"$\phi(M_{\rm UV})$")
        ax_top.legend(fontsize=11, frameon=False, loc="lower left")

        ax_bottom = axes[1, column]
        ax_bottom.plot(centers, ratio, color="#1f77b4", lw=2.2)
        ax_bottom.axhline(1.0, color="0.35", ls="--", lw=1.2)
        ax_bottom.set_ylim(0.2, 1.2)
        ax_bottom.grid(alpha=0.25)
        ax_bottom.set_xlabel(r"$M_{\rm UV}$")
        if column == 0:
            ax_bottom.set_ylabel("Delay / no delay")

    fig.suptitle(
        "UVLF effect of the extended-burst delay kernel",
        fontsize=17,
        fontweight="bold",
    )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    summary_path = output_prefix.with_suffix(".txt")
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)

    _write_summary(summary_path, z_values, no_delay_results, delay_results)
    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")
    print(f"saved_txt={summary_path}")


if __name__ == "__main__":
    main()
