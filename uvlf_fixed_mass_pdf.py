#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uvlf import run_halo_uv_pipeline, uv_luminosity_to_muv


def reserve_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _run_chunk(args: tuple[float, float, int, int]) -> np.ndarray:
    z_final, mh_final, n_tracks, seed = args
    result = run_halo_uv_pipeline(
        n_tracks=n_tracks,
        z_final=z_final,
        Mh_final=mh_final,
        workers=1,
        random_seed=seed,
    )
    muv = np.asarray(uv_luminosity_to_muv(result.uv_luminosities), dtype=float)
    return muv[np.isfinite(muv)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate the Muv PDF for a fixed halo mass using Monte Carlo MAH sampling.")
    parser.add_argument("--z-final", type=float, default=6.0)
    parser.add_argument("--mh-final", type=float, default=1.0e10)
    parser.add_argument("--n-tracks", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=120)
    args = parser.parse_args()

    if args.n_tracks <= 0 or args.chunk_size <= 0 or args.n_bins <= 1:
        raise ValueError("n-tracks, chunk-size, and n-bins must be positive")

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"uvlf_fixed_mass_pdf_z{args.z_final:g}_mh1e10"
    png_path = reserve_output_path(outputs_dir / f"{base_name}.png")
    tsv_path = reserve_output_path(outputs_dir / f"{base_name}.tsv")
    txt_path = reserve_output_path(outputs_dir / f"{base_name}.txt")

    n_chunks = math.ceil(args.n_tracks / args.chunk_size)
    chunk_sizes = [args.chunk_size] * n_chunks
    chunk_sizes[-1] = args.n_tracks - args.chunk_size * (n_chunks - 1)
    tasks = [
        (float(args.z_final), float(args.mh_final), int(chunk_n), int(args.random_seed + i))
        for i, chunk_n in enumerate(chunk_sizes)
    ]

    samples: list[np.ndarray] = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [executor.submit(_run_chunk, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            samples.append(future.result())
            if completed == 1 or completed % max(1, n_chunks // 20) == 0 or completed == n_chunks:
                elapsed = time.perf_counter() - start
                print(f"chunks {completed}/{n_chunks} elapsed={elapsed:.1f}s", flush=True)

    muv = np.concatenate(samples) if samples else np.empty(0, dtype=float)
    if muv.size == 0:
        raise RuntimeError("no finite Muv samples were generated")

    data_min = float(np.min(muv))
    data_max = float(np.max(muv))
    bin_edges = np.linspace(data_min, data_max, int(args.n_bins) + 1, dtype=float)
    density, edges = np.histogram(muv, bins=bin_edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    elapsed = time.perf_counter() - start
    mean = float(np.mean(muv))
    std = float(np.std(muv))
    p16, p50, p84 = np.percentile(muv, [16.0, 50.0, 84.0])
    p01, p99 = np.percentile(muv, [1.0, 99.0])

    lines = ["bin_left\tbin_right\tbin_center\tpdf"]
    for left, right, center, pdf in zip(edges[:-1], edges[1:], centers, density, strict=True):
        lines.append(f"{left:.6f}\t{right:.6f}\t{center:.6f}\t{float(pdf):.8e}")
    tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    txt_path.write_text(
        "\n".join(
            [
                "Fixed-Mh Muv PDF from Monte Carlo MAH sampling",
                f"z_final: {args.z_final}",
                f"mh_final: {args.mh_final}",
                f"n_tracks: {args.n_tracks}",
                f"chunk_size: {args.chunk_size}",
                f"workers: {args.workers}",
                f"random_seed: {args.random_seed}",
                f"n_bins: {args.n_bins}",
                f"muv_min: {data_min}",
                f"muv_max: {data_max}",
                f"mean_muv: {mean}",
                f"std_muv: {std}",
                f"p01: {float(p01)}",
                f"p16: {float(p16)}",
                f"p50: {float(p50)}",
                f"p84: {float(p84)}",
                f"p99: {float(p99)}",
                f"elapsed_seconds: {elapsed}",
                f"png_path: {png_path.resolve()}",
                f"tsv_path: {tsv_path.resolve()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.step(centers, density, where="mid", color="#1f4e79", linewidth=2.2)
    ax.fill_between(centers, density, step="mid", alpha=0.18, color="#1f4e79")
    ax.axvline(mean, color="#B03A2E", linestyle="--", linewidth=1.5, label=rf"mean = {mean:.2f}")
    ax.axvline(p50, color="#1D8348", linestyle="-.", linewidth=1.5, label=rf"median = {float(p50):.2f}")
    ax.set_xlabel(r"$M_{\rm UV}$")
    ax.set_ylabel("PDF")
    ax.set_title(rf"$M_h = 10^{{10}}\,M_\odot$ at $z={args.z_final:g}$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=10)
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(png_path.resolve())
    print(tsv_path.resolve())
    print(txt_path.resolve())


if __name__ == "__main__":
    main()
