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


MASS_COLORS = {
    1.0e8: "#6C5CE7",
    1.0e9: "#00A8B5",
    1.0e10: "#E67E22",
    1.0e11: "#C0392B",
}


def reserve_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _run_chunk(args: tuple[float, float, int, int]) -> tuple[float, np.ndarray]:
    z_final, mh_final, n_tracks, seed = args
    result = run_halo_uv_pipeline(
        n_tracks=n_tracks,
        z_final=z_final,
        Mh_final=mh_final,
        workers=1,
        random_seed=seed,
    )
    muv = np.asarray(uv_luminosity_to_muv(result.uv_luminosities), dtype=float)
    return mh_final, muv[np.isfinite(muv)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed-Mh Muv PDFs from Monte Carlo MAH sampling.")
    parser.add_argument("--z-final", type=float, default=6.0)
    parser.add_argument("--mh-values", type=float, nargs="+", default=[1.0e8, 1.0e9, 1.0e10, 1.0e11])
    parser.add_argument("--n-tracks", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=160)
    parser.add_argument("--ymin", type=float, default=1.0e-4)
    args = parser.parse_args()

    if args.n_tracks <= 0 or args.chunk_size <= 0 or args.n_bins <= 1:
        raise ValueError("n-tracks, chunk-size, and n-bins must be positive")

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"uvlf_fixed_mass_pdf_multi_z{args.z_final:g}"
    png_path = reserve_output_path(outputs_dir / f"{base_name}.png")
    tsv_path = reserve_output_path(outputs_dir / f"{base_name}.tsv")
    txt_path = reserve_output_path(outputs_dir / f"{base_name}.txt")

    mh_values = [float(x) for x in args.mh_values]
    n_chunks = math.ceil(args.n_tracks / args.chunk_size)
    chunk_sizes = [args.chunk_size] * n_chunks
    chunk_sizes[-1] = args.n_tracks - args.chunk_size * (n_chunks - 1)

    tasks: list[tuple[float, float, int, int]] = []
    task_index = 0
    for mh_final in mh_values:
        for chunk_n in chunk_sizes:
            tasks.append((float(args.z_final), mh_final, int(chunk_n), int(args.random_seed + task_index)))
            task_index += 1

    samples_by_mass: dict[float, list[np.ndarray]] = {mh: [] for mh in mh_values}
    completed_by_mass: dict[float, int] = {mh: 0 for mh in mh_values}
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [executor.submit(_run_chunk, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            mh_final, muv_chunk = future.result()
            samples_by_mass[mh_final].append(muv_chunk)
            completed_by_mass[mh_final] += 1
            if completed == 1 or completed % max(1, len(tasks) // 20) == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - start
                status = ", ".join(
                    f"{mh:.0e}:{completed_by_mass[mh]}/{n_chunks}" for mh in mh_values
                )
                print(f"chunks {completed}/{len(tasks)} elapsed={elapsed:.1f}s | {status}", flush=True)

    concatenated: dict[float, np.ndarray] = {}
    global_min = np.inf
    global_max = -np.inf
    for mh_final in mh_values:
        muv = np.concatenate(samples_by_mass[mh_final]) if samples_by_mass[mh_final] else np.empty(0, dtype=float)
        if muv.size == 0:
            raise RuntimeError(f"no finite Muv samples were generated for Mh={mh_final:.3e}")
        concatenated[mh_final] = muv
        global_min = min(global_min, float(np.min(muv)))
        global_max = max(global_max, float(np.max(muv)))

    edges = np.linspace(global_min, global_max, int(args.n_bins) + 1, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])

    elapsed = time.perf_counter() - start
    summary_lines = [
        "Fixed-Mh Muv PDF comparison from Monte Carlo MAH sampling",
        f"z_final: {args.z_final}",
        "mh_values: " + ", ".join(f"{mh:.0e}" for mh in mh_values),
        f"n_tracks_each: {args.n_tracks}",
        f"chunk_size: {args.chunk_size}",
        f"workers: {args.workers}",
        f"random_seed: {args.random_seed}",
        f"n_bins: {args.n_bins}",
        f"ymin: {args.ymin}",
        f"global_muv_min: {global_min}",
        f"global_muv_max: {global_max}",
        f"elapsed_seconds: {elapsed}",
        f"png_path: {png_path.resolve()}",
        f"tsv_path: {tsv_path.resolve()}",
        "",
    ]

    tsv_lines = ["mh_final\tbin_left\tbin_right\tbin_center\tpdf"]
    fig, ax = plt.subplots(figsize=(7.6, 5.4), constrained_layout=True)
    for mh_final in mh_values:
        muv = concatenated[mh_final]
        density, hist_edges = np.histogram(muv, bins=edges, density=True)
        positive = density > 0.0
        color = MASS_COLORS.get(mh_final, None)
        label = rf"$M_h=10^{{{int(np.log10(mh_final))}}}\,M_\odot$"
        ax.step(centers[positive], density[positive], where="mid", linewidth=2.0, color=color, label=label)

        mean = float(np.mean(muv))
        std = float(np.std(muv))
        p16, p50, p84 = np.percentile(muv, [16.0, 50.0, 84.0])
        summary_lines.extend(
            [
                f"Mh={mh_final:.0e}: mean_muv={mean}",
                f"Mh={mh_final:.0e}: std_muv={std}",
                f"Mh={mh_final:.0e}: p16={float(p16)}",
                f"Mh={mh_final:.0e}: p50={float(p50)}",
                f"Mh={mh_final:.0e}: p84={float(p84)}",
            ]
        )
        for left, right, center, pdf in zip(hist_edges[:-1], hist_edges[1:], centers, density, strict=True):
            tsv_lines.append(f"{mh_final:.6e}\t{left:.6f}\t{right:.6f}\t{center:.6f}\t{float(pdf):.8e}")

    ax.set_yscale("log")
    ax.set_ylim(float(args.ymin), 1.0)
    ax.set_xlabel(r"$M_{\rm UV}$")
    ax.set_ylabel("PDF")
    ax.set_title(rf"Fixed-$M_h$ MUV PDF at $z={args.z_final:g}$")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, fontsize=10)
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
    txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(png_path.resolve())
    print(tsv_path.resolve())
    print(txt_path.resolve())


if __name__ == "__main__":
    main()
