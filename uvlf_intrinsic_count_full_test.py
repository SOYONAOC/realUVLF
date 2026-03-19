#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uvlf import sample_uvlf_from_hmf


MUV_MIN = -28.0
MUV_MAX = -10.0
BIN_WIDTH = 0.5


def reserve_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Count intrinsic UV samples per 0.5 mag bin.")
    parser.add_argument("--z-obs", type=float, default=6.0)
    parser.add_argument("--N-mass", type=int, default=3000)
    parser.add_argument("--n-tracks", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    plot_path = reserve_output_path(outputs_dir / "uvlf_intrinsic_count_z6.png")
    table_path = reserve_output_path(outputs_dir / "uvlf_intrinsic_count_z6.tsv")
    summary_path = reserve_output_path(outputs_dir / "uvlf_intrinsic_count_z6.txt")
    progress_path = reserve_output_path(outputs_dir / "uvlf_intrinsic_count_z6_progress.txt")

    bin_edges = np.arange(MUV_MIN, MUV_MAX + BIN_WIDTH, BIN_WIDTH, dtype=float)
    if not np.isclose(bin_edges[-1], MUV_MAX):
        bin_edges = np.append(bin_edges, MUV_MAX)

    t0 = time.perf_counter()
    result = sample_uvlf_from_hmf(
        z_obs=float(args.z_obs),
        N_mass=args.N_mass,
        n_tracks=args.n_tracks,
        bins=bin_edges,
        pipeline_workers=max(1, args.workers),
        random_seed=args.random_seed,
        progress_path=progress_path,
    )
    muv = np.asarray(result.samples["Muv"], dtype=float)
    finite = np.isfinite(muv)
    counts, used_edges = np.histogram(muv[finite], bins=bin_edges)
    centers = 0.5 * (used_edges[:-1] + used_edges[1:])
    elapsed = time.perf_counter() - t0

    lines = ["bin_left\tbin_right\tbin_center\tcount"]
    for left, right, center, count in zip(used_edges[:-1], used_edges[1:], centers, counts, strict=True):
        lines.append(f"{left:.1f}\t{right:.1f}\t{center:.2f}\t{int(count)}")
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_path.write_text(
        "\n".join(
            [
                f"z_obs: {args.z_obs}",
                f"N_mass: {args.N_mass}",
                f"n_tracks: {args.n_tracks}",
                f"workers: {args.workers}",
                f"random_seed: {args.random_seed}",
                f"total_samples: {muv.size}",
                f"finite_samples: {int(np.count_nonzero(finite))}",
                f"sampling_seconds: {result.metadata['sampling_seconds']}",
                f"total_seconds: {elapsed}",
                f"table_path: {table_path.resolve()}",
                f"plot_path: {plot_path.resolve()}",
                f"progress_path: {progress_path.resolve()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.bar(centers, counts, width=BIN_WIDTH * 0.92, color="#2C5AA0", edgecolor="white", linewidth=0.3)
    ax.set_xlabel(r"$M_{\rm UV}$")
    ax.set_ylabel("Sample count")
    ax.set_title(f"Intrinsic UV sample counts per 0.5 mag bin at z = {args.z_obs:g}")
    ax.set_xlim(MUV_MIN, MUV_MAX)
    ax.tick_params(direction="in", top=True, right=True)
    ax.minorticks_on()
    fig.savefig(plot_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(table_path.resolve())
    print(plot_path.resolve())
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
