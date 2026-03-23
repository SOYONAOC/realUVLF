#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from uvlf import sample_uvlf_from_hmf


def format_redshift_tag(z_value: float) -> str:
    return f"{z_value:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the vectorized mah -> sfr -> SSP -> UVLF sampling pipeline."
    )
    parser.add_argument("--z-values", type=float, nargs="+", default=[6.0, 8.0, 10.0, 12.5])
    parser.add_argument("--N-mass", type=int, default=3000)
    parser.add_argument("--n-tracks", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--z-start-max", type=float, default=50.0)
    parser.add_argument("--n-grid", type=int, default=240)
    parser.add_argument("--sampler", type=str, default="mcbride")
    parser.add_argument("--enable-time-delay", action="store_true", default=True)
    parser.add_argument(
        "--output-stem",
        type=str,
        default="vectorized_uvlf_sampling_benchmark",
        help="Filename stem under outputs/ for the benchmark summary.",
    )
    args = parser.parse_args()

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = outputs_dir / f"{args.output_stem}.txt"

    lines = [
        "Vectorized UVLF sampling benchmark",
        f"z_values: {' '.join(f'{z:g}' for z in args.z_values)}",
        f"N_mass: {args.N_mass}",
        f"n_tracks: {args.n_tracks}",
        f"bins: {args.bins}",
        f"workers: {args.workers}",
        f"random_seed: {args.random_seed}",
        f"z_start_max: {args.z_start_max}",
        f"n_grid: {args.n_grid}",
        f"sampler: {args.sampler}",
        f"enable_time_delay: {args.enable_time_delay}",
        "",
    ]

    benchmark_start = time.perf_counter()
    for z_value in args.z_values:
        z_tag = format_redshift_tag(z_value)
        progress_path = outputs_dir / f"{args.output_stem}_z{z_tag}_progress.txt"
        print(
            f"Starting z={z_value:g} with N_mass={args.N_mass}, n_tracks={args.n_tracks}, workers={args.workers}",
            flush=True,
        )
        t0 = time.perf_counter()
        result = sample_uvlf_from_hmf(
            z_obs=float(z_value),
            N_mass=int(args.N_mass),
            n_tracks=int(args.n_tracks),
            random_seed=int(args.random_seed),
            bins=int(args.bins),
            z_start_max=float(args.z_start_max),
            n_grid=int(args.n_grid),
            sampler=args.sampler,
            enable_time_delay=bool(args.enable_time_delay),
            pipeline_workers=int(args.workers),
            progress_path=progress_path,
            print_progress=True,
        )
        elapsed = time.perf_counter() - t0
        finite_phi = int((result.uvlf["phi"] > 0.0).sum())
        median_pipeline = float(result.metadata["per_mass_pipeline_seconds"].mean())
        print(
            f"Finished z={z_value:g}: total={elapsed:.2f}s, sampling={result.metadata['sampling_seconds']:.2f}s, "
            f"mean_per_mass={median_pipeline:.4f}s, nonzero_bins={finite_phi}",
            flush=True,
        )
        lines.extend(
            [
                f"z={z_value:g}",
                f"  sampling_seconds: {float(result.metadata['sampling_seconds']):.6f}",
                f"  wall_seconds: {elapsed:.6f}",
                f"  mean_per_mass_pipeline_seconds: {median_pipeline:.6f}",
                f"  uvlf_nonzero_bins: {finite_phi}",
                f"  progress_path: {progress_path.resolve()}",
                "",
            ]
        )

    total_elapsed = time.perf_counter() - benchmark_start
    lines.extend(
        [
            f"benchmark_total_seconds: {total_elapsed:.6f}",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(summary_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
