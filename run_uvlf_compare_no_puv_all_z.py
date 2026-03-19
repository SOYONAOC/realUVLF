#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys


Z_VALUES = [6.0, 8.0, 10.0, 12.5]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run uvlf_compare_no_puv_to_dust.py for z=6,8,10,12.5 with consistent settings."
    )
    parser.add_argument("--N-mass", type=int, default=3000)
    parser.add_argument("--n-tracks", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    for z_value in Z_VALUES:
        cmd = [
            sys.executable,
            "uvlf_compare_no_puv_to_dust.py",
            "--z-obs",
            f"{z_value:g}",
            "--N-mass",
            str(args.N_mass),
            "--n-tracks",
            str(args.n_tracks),
            "--bins",
            str(args.bins),
            "--workers",
            str(args.workers),
            "--random-seed",
            str(args.random_seed),
        ]
        print("running", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
