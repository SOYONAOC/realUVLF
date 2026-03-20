#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dust-only UVLF compare plots for z=6,8,10,12.5.")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--N-mass", type=int, default=3000)
    parser.add_argument("--n-tracks", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--z-start-max", type=float, default=50.0)
    parser.add_argument("--n-grid", type=int, default=240)
    parser.add_argument("--sampler", type=str, default="mcbride")
    parser.add_argument("--enable-time-delay", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    python = sys.executable
    script = project_root / "uvlf_compare_no_puv_to_dust.py"

    for z in (6.0, 8.0, 10.0, 12.5):
        cmd = [
            python,
            str(script),
            "--z-obs",
            str(z),
            "--dust-only",
            "--workers",
            str(args.workers),
            "--N-mass",
            str(args.N_mass),
            "--n-tracks",
            str(args.n_tracks),
            "--bins",
            str(args.bins),
            "--random-seed",
            str(args.random_seed),
            "--z-start-max",
            str(args.z_start_max),
            "--n-grid",
            str(args.n_grid),
            "--sampler",
            str(args.sampler),
        ]
        if args.enable_time_delay:
            cmd.append("--enable-time-delay")
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
