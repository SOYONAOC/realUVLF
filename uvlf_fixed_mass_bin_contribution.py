#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uvlf import run_halo_uv_pipeline, uv_luminosity_to_muv


DEFAULT_REFERENCE_TABLE = Path("outputs/uvlf_intrinsic_count_z6.tsv")


def reserve_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def load_reference_table(table_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, int]] = []
    with table_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append((float(row["bin_left"]), float(row["bin_right"]), int(row["count"])))
    if not rows:
        raise RuntimeError(f"no rows found in reference table: {table_path}")
    bin_edges = np.array([rows[0][0], *[right for _, right, _ in rows]], dtype=float)
    reference_counts = np.array([count for _, _, count in rows], dtype=np.int64)
    return bin_edges, reference_counts


def _run_chunk(
    args: tuple[float, float, int, np.ndarray, int],
) -> np.ndarray:
    z_obs, mh_final, n_tracks, bin_edges, seed = args
    result = run_halo_uv_pipeline(
        n_tracks=n_tracks,
        z_final=z_obs,
        Mh_final=mh_final,
        workers=1,
        random_seed=seed,
    )
    muv = np.asarray(uv_luminosity_to_muv(result.uv_luminosities), dtype=float)
    finite = np.isfinite(muv)
    counts, _ = np.histogram(muv[finite], bins=bin_edges)
    return counts.astype(np.int64, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count fixed-Mh UV samples per bin and compare to full-sample counts.")
    parser.add_argument("--z-obs", type=float, default=6.0)
    parser.add_argument("--mh-final", type=float, default=1.0e10)
    parser.add_argument("--n-tracks", type=int, default=3_000_000)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--reference-table", type=Path, default=DEFAULT_REFERENCE_TABLE)
    args = parser.parse_args()

    if args.n_tracks <= 0 or args.chunk_size <= 0:
        raise ValueError("n-tracks and chunk-size must be positive")

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    table_path = reserve_output_path(outputs_dir / "uvlf_fixed_mass_1e10_contribution.tsv")
    plot_path = reserve_output_path(outputs_dir / "uvlf_fixed_mass_1e10_contribution.png")
    summary_path = reserve_output_path(outputs_dir / "uvlf_fixed_mass_1e10_contribution.txt")

    bin_edges, reference_counts = load_reference_table(args.reference_table)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    n_chunks = int(np.ceil(args.n_tracks / args.chunk_size))
    chunk_sizes = np.full(n_chunks, args.chunk_size, dtype=int)
    chunk_sizes[-1] = args.n_tracks - args.chunk_size * (n_chunks - 1)
    tasks = [
        (float(args.z_obs), float(args.mh_final), int(chunk_n), bin_edges, int(args.random_seed + i))
        for i, chunk_n in enumerate(chunk_sizes)
    ]

    total_counts = np.zeros(bin_centers.size, dtype=np.int64)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [executor.submit(_run_chunk, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            total_counts += np.asarray(future.result(), dtype=np.int64)
            if completed == 1 or completed % max(1, n_chunks // 20) == 0 or completed == n_chunks:
                elapsed = time.perf_counter() - start
                print(f"chunks {completed}/{n_chunks} elapsed={elapsed:.1f}s", flush=True)

    ratios = np.zeros_like(total_counts, dtype=float)
    positive_ref = reference_counts > 0
    ratios[positive_ref] = total_counts[positive_ref] / reference_counts[positive_ref]
    elapsed = time.perf_counter() - start

    lines = ["bin_left\tbin_right\tbin_center\tfixed_mass_count\tfull_count\tratio_to_full"]
    for left, right, center, fixed_count, full_count, ratio in zip(
        bin_edges[:-1],
        bin_edges[1:],
        bin_centers,
        total_counts,
        reference_counts,
        ratios,
        strict=True,
    ):
        lines.append(
            f"{left:.1f}\t{right:.1f}\t{center:.2f}\t{int(fixed_count)}\t{int(full_count)}\t{ratio:.8f}"
        )
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_path.write_text(
        "\n".join(
            [
                f"z_obs: {args.z_obs}",
                f"mh_final: {args.mh_final}",
                f"n_tracks: {args.n_tracks}",
                f"chunk_size: {args.chunk_size}",
                f"workers: {args.workers}",
                f"random_seed: {args.random_seed}",
                f"reference_table: {args.reference_table.resolve()}",
                f"elapsed_seconds: {elapsed}",
                f"table_path: {table_path.resolve()}",
                f"plot_path: {plot_path.resolve()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), constrained_layout=True, sharex=True)
    axes[0].bar(bin_centers, total_counts, width=np.diff(bin_edges) * 0.92, color="#A04D2C", edgecolor="white", linewidth=0.25)
    axes[0].set_ylabel(r"Count for $M_h=10^{10} M_\odot$")
    axes[0].set_title(r"Fixed-$M_h$ contribution by UV bin at $z=6$")
    axes[0].tick_params(direction="in", top=True, right=True)
    axes[0].minorticks_on()

    axes[1].bar(bin_centers, ratios, width=np.diff(bin_edges) * 0.92, color="#2C5AA0", edgecolor="white", linewidth=0.25)
    axes[1].set_xlabel(r"$M_{\rm UV}$")
    axes[1].set_ylabel("Fixed/full ratio")
    axes[1].tick_params(direction="in", top=True, right=True)
    axes[1].minorticks_on()
    axes[1].set_xlim(bin_edges[0], bin_edges[-1])

    fig.savefig(plot_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(table_path.resolve())
    print(plot_path.resolve())
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
