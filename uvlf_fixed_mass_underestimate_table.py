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


def _truth_chunk(
    args: tuple[float, float, int, tuple[float, ...], int],
) -> tuple[int, float, float, np.ndarray]:
    z_final, mh_final, n_tracks, thresholds, seed = args
    result = run_halo_uv_pipeline(
        n_tracks=n_tracks,
        z_final=z_final,
        Mh_final=mh_final,
        workers=1,
        random_seed=seed,
    )
    muv = np.asarray(uv_luminosity_to_muv(result.uv_luminosities), dtype=float)
    finite = muv[np.isfinite(muv)]
    threshold_counts = np.array([np.count_nonzero(finite < thr) for thr in thresholds], dtype=np.int64)
    return int(finite.size), float(np.sum(finite)), float(np.sum(finite**2)), threshold_counts


def _repeat_sample(
    args: tuple[float, float, int, tuple[float, ...], int],
) -> np.ndarray:
    z_final, mh_final, n_tracks, thresholds, seed = args
    result = run_halo_uv_pipeline(
        n_tracks=n_tracks,
        z_final=z_final,
        Mh_final=mh_final,
        workers=1,
        random_seed=seed,
    )
    muv = np.asarray(uv_luminosity_to_muv(result.uv_luminosities), dtype=float)
    finite = muv[np.isfinite(muv)]
    return np.array([np.count_nonzero(finite < thr) for thr in thresholds], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify fixed-Mh tail underestimation from limited sampling.")
    parser.add_argument("--z-final", type=float, default=6.0)
    parser.add_argument("--mh-final", type=float, default=2.0e10)
    parser.add_argument("--truth-tracks", type=int, default=1_000_000)
    parser.add_argument("--truth-chunk-size", type=int, default=10_000)
    parser.add_argument("--repeat-sizes", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[-18.0, -19.0, -20.0])
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    table_path = reserve_output_path(outputs_dir / "uvlf_fixed_mass_2e10_underestimate.tsv")
    summary_path = reserve_output_path(outputs_dir / "uvlf_fixed_mass_2e10_underestimate.txt")
    plot_path = reserve_output_path(outputs_dir / "uvlf_fixed_mass_2e10_underestimate.png")

    thresholds = tuple(float(x) for x in args.thresholds)
    n_workers = max(1, int(args.workers))
    t0 = time.perf_counter()

    truth_n_chunks = math.ceil(args.truth_tracks / args.truth_chunk_size)
    truth_chunk_sizes = [args.truth_chunk_size] * truth_n_chunks
    truth_chunk_sizes[-1] = args.truth_tracks - args.truth_chunk_size * (truth_n_chunks - 1)
    truth_tasks = [
        (float(args.z_final), float(args.mh_final), int(chunk_n), thresholds, int(args.random_seed + i))
        for i, chunk_n in enumerate(truth_chunk_sizes)
    ]

    truth_total_n = 0
    truth_sum = 0.0
    truth_sum_sq = 0.0
    truth_threshold_counts = np.zeros(len(thresholds), dtype=np.int64)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        truth_futures = [executor.submit(_truth_chunk, task) for task in truth_tasks]
        for completed, future in enumerate(as_completed(truth_futures), start=1):
            n, sample_sum, sample_sum_sq, threshold_counts = future.result()
            truth_total_n += n
            truth_sum += sample_sum
            truth_sum_sq += sample_sum_sq
            truth_threshold_counts += threshold_counts
            if completed == 1 or completed % max(1, truth_n_chunks // 20) == 0 or completed == truth_n_chunks:
                elapsed = time.perf_counter() - t0
                print(f"truth chunks {completed}/{truth_n_chunks} elapsed={elapsed:.1f}s", flush=True)

        truth_mean = truth_sum / truth_total_n
        truth_var = max(truth_sum_sq / truth_total_n - truth_mean**2, 0.0)
        truth_std = float(np.sqrt(truth_var))
        truth_prob = truth_threshold_counts.astype(float) / float(truth_total_n)

        repeat_results: list[dict[str, float | int]] = []
        plot_expected: list[float] = []
        plot_means: list[float] = []
        plot_stds: list[float] = []
        plot_labels: list[str] = []

        for size_index, n_tracks in enumerate(args.repeat_sizes):
            repeat_tasks = [
                (
                    float(args.z_final),
                    float(args.mh_final),
                    int(n_tracks),
                    thresholds,
                    int(args.random_seed + 100_000 * (size_index + 1) + repeat_index),
                )
                for repeat_index in range(args.repeats)
            ]
            counts_matrix = np.empty((args.repeats, len(thresholds)), dtype=np.int64)
            repeat_futures = [executor.submit(_repeat_sample, task) for task in repeat_tasks]
            for completed, future in enumerate(as_completed(repeat_futures), start=1):
                counts_matrix[completed - 1] = future.result()
            for threshold_index, threshold in enumerate(thresholds):
                expected_count = float(n_tracks * truth_prob[threshold_index])
                counts = counts_matrix[:, threshold_index].astype(float)
                mean_count = float(np.mean(counts))
                std_count = float(np.std(counts))
                frac_zero = float(np.mean(counts == 0.0))
                mean_underestimate_percent = (
                    float((expected_count - mean_count) / expected_count * 100.0) if expected_count > 0.0 else 0.0
                )
                repeat_results.append(
                    {
                        "n_tracks": int(n_tracks),
                        "threshold": float(threshold),
                        "truth_probability": float(truth_prob[threshold_index]),
                        "expected_count": expected_count,
                        "mean_count": mean_count,
                        "std_count": std_count,
                        "frac_zero": frac_zero,
                        "min_count": int(np.min(counts)),
                        "max_count": int(np.max(counts)),
                        "mean_underestimate_percent": mean_underestimate_percent,
                    }
                )
                plot_expected.append(expected_count)
                plot_means.append(mean_count)
                plot_stds.append(std_count)
                plot_labels.append(f"n={n_tracks:g}\nM<{threshold:g}")

    elapsed = time.perf_counter() - t0

    lines = [
        "n_tracks\tthreshold\ttruth_probability\texpected_count\tmean_count\tstd_count\tfrac_zero\tmin_count\tmax_count\tmean_underestimate_percent"
    ]
    for row in repeat_results:
        lines.append(
            "\t".join(
                [
                    str(int(row["n_tracks"])),
                    f"{float(row['threshold']):.1f}",
                    f"{float(row['truth_probability']):.8f}",
                    f"{float(row['expected_count']):.6f}",
                    f"{float(row['mean_count']):.6f}",
                    f"{float(row['std_count']):.6f}",
                    f"{float(row['frac_zero']):.6f}",
                    str(int(row["min_count"])),
                    str(int(row["max_count"])),
                    f"{float(row['mean_underestimate_percent']):.6f}",
                ]
            )
        )
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_path.write_text(
        "\n".join(
            [
                f"z_final: {args.z_final}",
                f"mh_final: {args.mh_final}",
                f"truth_tracks: {args.truth_tracks}",
                f"truth_chunk_size: {args.truth_chunk_size}",
                f"repeat_sizes: {','.join(str(x) for x in args.repeat_sizes)}",
                f"repeats: {args.repeats}",
                f"thresholds: {','.join(str(x) for x in thresholds)}",
                f"workers: {n_workers}",
                f"random_seed: {args.random_seed}",
                f"truth_mean_muv: {truth_mean}",
                f"truth_std_muv: {truth_std}",
                *[
                    f"truth_probability_Muv<{thr:g}: {prob}"
                    for thr, prob in zip(thresholds, truth_prob, strict=True)
                ],
                f"elapsed_seconds: {elapsed}",
                f"table_path: {table_path.resolve()}",
                f"plot_path: {plot_path.resolve()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    x = np.arange(len(plot_labels), dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.bar(x - 0.18, plot_expected, width=0.36, color="#C7D4EA", label="Expected from truth")
    ax.bar(x + 0.18, plot_means, width=0.36, color="#2C5AA0", label="Mean from finite sampling")
    ax.errorbar(x + 0.18, plot_means, yerr=plot_stds, fmt="none", ecolor="black", capsize=2.5, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, fontsize=8)
    ax.set_ylabel("Count below threshold")
    ax.set_title(r"Finite-sampling underestimation for fixed $M_h=2\times10^{10} M_\odot$ at $z=6$")
    ax.tick_params(direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.legend(frameon=False)
    fig.savefig(plot_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(table_path.resolve())
    print(summary_path.resolve())
    print(plot_path.resolve())


if __name__ == "__main__":
    main()
