#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uvlf import intrinsic_muv_from_observed, sample_uvlf_from_hmf


DEFAULT_LOGMH_EDGES = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5]
OBS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def format_z_tag(z_value: float) -> str:
    return f"{z_value:g}".replace(".", "p")


def _mass_bin_labels(logmh_edges: np.ndarray) -> list[str]:
    labels: list[str] = []
    for left, right in zip(logmh_edges[:-1], logmh_edges[1:], strict=True):
        left_label = f"{left:.1f}" if not float(left).is_integer() else f"{left:.0f}"
        right_label = f"{right:.1f}" if not float(right).is_integer() else f"{right:.0f}"
        labels.append(rf"$10^{{{left_label}}}\!-\!10^{{{right_label}}}\,M_\odot$")
    return labels


def _muv_edges(sample_muv: np.ndarray, bin_width: float) -> np.ndarray:
    finite = sample_muv[np.isfinite(sample_muv)]
    if finite.size == 0:
        raise RuntimeError("No finite Muv samples available.")
    left = np.floor(np.min(finite) / bin_width) * bin_width
    right = np.ceil(np.max(finite) / bin_width) * bin_width
    return np.arange(left, right + bin_width, bin_width, dtype=float)


def _format_redshift_tag(z_value: float) -> str:
    return f"{z_value:g}".replace(".", "p")


def _mass_range_tag(logmh_edges: np.ndarray) -> str:
    left = float(logmh_edges[0])
    right = float(logmh_edges[-1])

    def _fmt(value: float) -> str:
        return f"{int(round(value))}" if float(value).is_integer() else f"{value:g}".replace(".", "p")

    return f"mh{_fmt(left)}_{_fmt(right)}"


def _load_observational_uvlf(z_value: float) -> list[dict[str, np.ndarray | str]]:
    obs_dir = Path("obsdata") / f"redshift_{_format_redshift_tag(z_value)}"
    datasets: list[dict[str, np.ndarray | str]] = []
    if not obs_dir.exists():
        return datasets

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


def _mass_colors(n_bins: int) -> list[tuple[float, float, float, float]]:
    cmap = plt.get_cmap("turbo")
    if n_bins == 1:
        return [cmap(0.5)]
    return [cmap(x) for x in np.linspace(0.08, 0.92, n_bins)]


def _observed_muv_from_intrinsic(intrinsic_muv: np.ndarray, z: float) -> np.ndarray:
    intrinsic_muv = np.asarray(intrinsic_muv, dtype=float)
    finite = intrinsic_muv[np.isfinite(intrinsic_muv)]
    if finite.size == 0:
        return np.full_like(intrinsic_muv, np.nan, dtype=float)

    obs_left = np.floor(np.min(finite) - 8.0)
    obs_right = np.ceil(np.max(finite) + 4.0)
    observed_grid = np.linspace(obs_left, obs_right, 12000, dtype=float)
    intrinsic_grid = np.asarray(intrinsic_muv_from_observed(observed_grid, z), dtype=float)
    order = np.argsort(intrinsic_grid)
    intrinsic_sorted = intrinsic_grid[order]
    observed_sorted = observed_grid[order]
    unique_mask = np.concatenate(([True], np.diff(intrinsic_sorted) > 0.0))
    intrinsic_sorted = intrinsic_sorted[unique_mask]
    observed_sorted = observed_sorted[unique_mask]
    return np.interp(
        intrinsic_muv,
        intrinsic_sorted,
        observed_sorted,
        left=observed_sorted[0],
        right=observed_sorted[-1],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full UVLF sampling and plot per-Muv-bin halo-mass composition."
    )
    parser.add_argument("--z", type=float, default=12.5)
    parser.add_argument("--N-mass", type=int, default=3000)
    parser.add_argument("--n-tracks", type=int, default=1000)
    parser.add_argument("--pipeline-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--tail-threshold", type=float, default=-19.0)
    parser.add_argument("--logmh-edges", type=float, nargs="+", default=DEFAULT_LOGMH_EDGES)
    parser.add_argument("--x-min", type=float, default=-25.0)
    parser.add_argument("--x-max", type=float, default=-15.0)
    parser.add_argument("--dust", action="store_true", help="Use dust-attenuated observed MUV instead of intrinsic MUV.")
    parser.add_argument("--progress-path", type=str, default="")
    args = parser.parse_args()

    if args.N_mass <= 0 or args.n_tracks <= 0 or args.bin_width <= 0.0:
        raise ValueError("N-mass, n-tracks, and bin-width must be positive.")

    logmh_edges = np.asarray(args.logmh_edges, dtype=float)
    if logmh_edges.ndim != 1 or logmh_edges.size < 2 or not np.all(np.diff(logmh_edges) > 0):
        raise ValueError("logmh-edges must be a strictly increasing 1D array.")

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data_save")
    data_dir.mkdir(parents=True, exist_ok=True)

    z_tag = format_z_tag(args.z)
    progress_path = (
        Path(args.progress_path)
        if args.progress_path
        else outputs_dir / f"uvlf_full_mass_composition_z{z_tag}_progress.txt"
    )

    result = sample_uvlf_from_hmf(
        z_obs=float(args.z),
        N_mass=int(args.N_mass),
        n_tracks=int(args.n_tracks),
        random_seed=int(args.random_seed),
        quantity="Muv",
        bins=80,
        logM_min=float(logmh_edges[0]),
        logM_max=float(logmh_edges[-1]),
        progress_path=progress_path,
        pipeline_workers=int(args.pipeline_workers),
    )

    sample_muv_intrinsic = np.asarray(result.samples["Muv"], dtype=float)
    sample_logmh = np.asarray(result.samples["logMh"], dtype=float)
    sample_weight = np.asarray(result.samples["sample_weight"], dtype=float)
    if args.dust:
        sample_muv = _observed_muv_from_intrinsic(sample_muv_intrinsic, float(args.z))
    else:
        sample_muv = sample_muv_intrinsic

    valid = np.isfinite(sample_muv) & np.isfinite(sample_logmh) & np.isfinite(sample_weight)
    sample_muv = sample_muv[valid]
    sample_logmh = sample_logmh[valid]
    sample_weight = sample_weight[valid]

    muv_edges = _muv_edges(sample_muv, float(args.bin_width))
    muv_centers = 0.5 * (muv_edges[:-1] + muv_edges[1:])
    bin_width = np.diff(muv_edges)

    total_weighted_counts, _ = np.histogram(sample_muv, bins=muv_edges, weights=sample_weight)
    total_phi = total_weighted_counts / bin_width

    n_mass_bins = logmh_edges.size - 1
    contrib_counts = np.zeros((n_mass_bins, muv_centers.size), dtype=float)
    for i in range(n_mass_bins):
        mask = (sample_logmh >= logmh_edges[i]) & (sample_logmh < logmh_edges[i + 1])
        if np.any(mask):
            counts, _ = np.histogram(sample_muv[mask], bins=muv_edges, weights=sample_weight[mask])
            contrib_counts[i] = counts

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.divide(
            contrib_counts,
            total_weighted_counts[np.newaxis, :],
            out=np.zeros_like(contrib_counts),
            where=total_weighted_counts[np.newaxis, :] > 0.0,
        )

    labels = _mass_bin_labels(logmh_edges)
    mass_colors = _mass_colors(n_mass_bins)
    obs_datasets = _load_observational_uvlf(float(args.z))

    x_mask = (muv_centers >= float(args.x_min)) & (muv_centers <= float(args.x_max))
    if not np.any(x_mask):
        raise RuntimeError("No UVLF bins fall within the requested x-range.")

    visible_centers = muv_centers[x_mask]
    visible_phi = total_phi[x_mask]
    visible_frac = frac[:, x_mask]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.8, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
        constrained_layout=True,
    )
    ax_phi, ax_frac = axes

    positive = visible_phi > 0.0
    if np.any(positive):
        ax_phi.step(visible_centers[positive], visible_phi[positive], where="mid", color="black", linewidth=2.2)
    for dataset_index, dataset in enumerate(obs_datasets):
        obs_muv = np.asarray(dataset["Muv"], dtype=float)
        obs_phi = np.asarray(dataset["phi"], dtype=float)
        obs_mag_err = np.asarray(dataset["mag_err"], dtype=float)
        obs_phi_err_lo = np.asarray(dataset["phi_err_lo"], dtype=float)
        obs_phi_err_up = np.asarray(dataset["phi_err_up"], dtype=float)
        visible_obs = (
            np.isfinite(obs_muv)
            & np.isfinite(obs_phi)
            & (obs_muv >= float(args.x_min))
            & (obs_muv <= float(args.x_max))
            & (obs_phi > 0.0)
        )
        if not np.any(visible_obs):
            continue
        ax_phi.errorbar(
            obs_muv[visible_obs],
            obs_phi[visible_obs],
            xerr=obs_mag_err[visible_obs],
            yerr=np.vstack([obs_phi_err_lo[visible_obs], obs_phi_err_up[visible_obs]]),
            fmt="o",
            ms=5.8,
            lw=1.1,
            capsize=2.3,
            color=OBS_COLORS[dataset_index % len(OBS_COLORS)],
            label=str(dataset["label"]),
            zorder=3,
        )
    ax_phi.axvline(float(args.tail_threshold), color="0.35", linestyle="--", linewidth=1.2)
    ax_phi.set_yscale("log")
    positive_values: list[np.ndarray] = [visible_phi[visible_phi > 0.0]]
    for dataset in obs_datasets:
        obs_phi = np.asarray(dataset["phi"], dtype=float)
        obs_muv = np.asarray(dataset["Muv"], dtype=float)
        visible_obs = np.isfinite(obs_phi) & np.isfinite(obs_muv) & (obs_muv >= float(args.x_min)) & (obs_muv <= float(args.x_max)) & (obs_phi > 0.0)
        if np.any(visible_obs):
            positive_values.append(obs_phi[visible_obs])
    positive_concat = np.concatenate([arr for arr in positive_values if arr.size > 0]) if positive_values else np.array([])
    ymin = 10.0 ** np.floor(np.log10(np.min(positive_concat))) if positive_concat.size else 1.0e-8
    ymax = 10.0 ** np.ceil(np.log10(np.max(positive_concat))) * 1.2 if positive_concat.size else 1.0
    ax_phi.set_ylim(max(1.0e-8, ymin), ymax)
    ax_phi.set_xlim(float(args.x_min), float(args.x_max))
    ax_phi.set_ylabel(r"$\phi(M_{\rm UV})$")
    quantity_label = "Dust UVLF" if args.dust else "Intrinsic UVLF"
    ax_phi.set_title(rf"{quantity_label} and halo-mass composition at $z={args.z:g}$")
    ax_phi.grid(True, which="both", alpha=0.22)
    ax_phi.legend(frameon=False, fontsize=9, loc="lower left", ncol=2)

    bottoms = np.zeros_like(visible_centers)
    width = float(args.bin_width) * 0.92
    for i in range(n_mass_bins):
        ax_frac.bar(
            visible_centers,
            visible_frac[i],
            width=width,
            bottom=bottoms,
            color=mass_colors[i],
            edgecolor="white",
            linewidth=0.35,
            label=labels[i],
            align="center",
        )
        bottoms += visible_frac[i]

    ax_frac.axvline(float(args.tail_threshold), color="0.35", linestyle="--", linewidth=1.2)
    ax_frac.set_ylim(0.0, 1.0)
    ax_frac.set_ylabel("Fraction in each $M_{\\rm UV}$ bin")
    ax_frac.set_xlabel(r"$M_{\rm UV}$")
    ax_frac.grid(True, axis="y", alpha=0.22)
    ax_frac.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")
    ax_frac.set_xlim(float(args.x_min), float(args.x_max))

    range_tag = f"m{abs(int(args.x_min))}_m{abs(int(args.x_max))}"
    mass_tag = _mass_range_tag(logmh_edges)
    dust_tag = "_dust" if args.dust else ""
    png_path = outputs_dir / f"uvlf_full_mass_composition_z{z_tag}{dust_tag}_{range_tag}_{mass_tag}.png"
    pdf_path = outputs_dir / f"uvlf_full_mass_composition_z{z_tag}{dust_tag}_{range_tag}_{mass_tag}.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    frac_rows = [
        "bin_left\tbin_right\tbin_center\ttotal_weighted_count\tphi\t"
        + "\t".join(f"frac_{int(logmh_edges[i])}_{int(logmh_edges[i+1])}" for i in range(n_mass_bins))
    ]
    for j, (left, right, center, count, phi_value) in enumerate(
        zip(muv_edges[:-1], muv_edges[1:], muv_centers, total_weighted_counts, total_phi, strict=True)
    ):
        frac_rows.append(
            "\t".join(
                [
                    f"{left:.8f}",
                    f"{right:.8f}",
                    f"{center:.8f}",
                    f"{float(count):.12e}",
                    f"{float(phi_value):.12e}",
                ]
                + [f"{float(frac[i, j]):.12e}" for i in range(n_mass_bins)]
            )
        )

    summary_lines = [
        "Full UVLF halo-mass composition",
        f"quantity: {'dust' if args.dust else 'intrinsic'}",
        f"z_obs: {args.z}",
        f"N_mass: {args.N_mass}",
        f"n_tracks: {args.n_tracks}",
        f"pipeline_workers: {args.pipeline_workers}",
        f"random_seed: {args.random_seed}",
        f"logmh_edges: {', '.join(f'{x:g}' for x in logmh_edges)}",
        f"bin_width: {args.bin_width}",
        f"tail_threshold: {args.tail_threshold}",
        f"x_range: [{args.x_min}, {args.x_max}]",
        f"png_path: {png_path.resolve()}",
        f"pdf_path: {pdf_path.resolve()}",
        f"progress_path: {progress_path.resolve()}",
    ]

    tail_mask = muv_centers < float(args.tail_threshold)
    if np.any(tail_mask) and np.sum(total_weighted_counts[tail_mask]) > 0.0:
        tail_contrib = np.sum(contrib_counts[:, tail_mask], axis=1)
        tail_frac = tail_contrib / np.sum(total_weighted_counts[tail_mask])
        summary_lines.append("Bright-end fractional composition (weighted):")
        for label, value in zip(labels, tail_frac, strict=True):
            summary_lines.append(f"  {label}: {float(value):.4f}")

    frac_path = data_dir / f"uvlf_full_mass_composition_z{z_tag}{dust_tag}_{range_tag}_{mass_tag}.tsv"
    summary_path = data_dir / f"uvlf_full_mass_composition_z{z_tag}{dust_tag}_{range_tag}_{mass_tag}.txt"
    frac_path.write_text("\n".join(frac_rows) + "\n", encoding="utf-8")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(png_path.resolve())
    print(pdf_path.resolve())
    print(frac_path.resolve())
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
