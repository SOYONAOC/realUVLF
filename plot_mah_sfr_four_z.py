#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mah import Cosmology
from mah.generator import generate_halo_histories
from sfr.calculator import DEFAULT_SFR_MODEL_PARAMETERS, compute_sfr_from_tracks


Z_VALUES = [6.0, 8.0, 10.0, 12.5]
Z_START_MAX = 50.0
N_GRID = 240
N_TRACKS = 1000
MH_FINAL = 1.0e11
LOOKBACK_MAX_MYR = 300.0


def reserve_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def summarize_histories(
    z_final: float,
    *,
    mh_final: float,
    n_tracks: int,
    z_start_max: float,
    n_grid: int,
    lookback_max_myr: float,
) -> dict[str, np.ndarray | float]:
    cosmology = Cosmology()
    redshift_grid = np.linspace(z_start_max, z_final, n_grid, dtype=float)
    histories = generate_halo_histories(
        n_tracks=n_tracks,
        z_final=z_final,
        Mh_final=mh_final,
        z_start_max=z_start_max,
        cosmology=cosmology,
        random_seed=42,
        time_grid_mode="custom",
        custom_grid=redshift_grid,
        store_inactive_history=True,
        sampler="mcbride",
    )
    sfr_tracks = compute_sfr_from_tracks(
        histories.tracks,
        model_parameters=DEFAULT_SFR_MODEL_PARAMETERS,
    )

    halo_ids = np.asarray(sfr_tracks["halo_id"], dtype=int)
    n_halos = np.unique(halo_ids).size
    steps = redshift_grid.size

    z_grid = np.asarray(sfr_tracks["z"], dtype=float).reshape(n_halos, steps)
    t_grid = np.asarray(sfr_tracks["t_gyr"], dtype=float).reshape(n_halos, steps)
    mh_grid = np.asarray(sfr_tracks["Mh"], dtype=float).reshape(n_halos, steps)
    sfr_grid = np.asarray(sfr_tracks["SFR"], dtype=float).reshape(n_halos, steps)
    active_grid = np.asarray(sfr_tracks["active_flag"], dtype=bool).reshape(n_halos, steps)

    lookback_myr = (t_grid[:, -1][:, None] - t_grid) * 1.0e3
    recent_mask = lookback_myr <= lookback_max_myr

    sfr_active_only = np.where(active_grid, sfr_grid, np.nan)

    return {
        "z_final": z_final,
        "z_grid": z_grid[0],
        "lookback_myr": lookback_myr[0],
        "mah_p16": np.percentile(mh_grid, 16.0, axis=0),
        "mah_p50": np.percentile(mh_grid, 50.0, axis=0),
        "mah_p84": np.percentile(mh_grid, 84.0, axis=0),
        "sfr_p16": np.nanpercentile(sfr_active_only, 16.0, axis=0),
        "sfr_p50": np.nanpercentile(sfr_active_only, 50.0, axis=0),
        "sfr_p84": np.nanpercentile(sfr_active_only, 84.0, axis=0),
        "recent_mask": recent_mask[0],
        "t_obs_gyr": float(t_grid[0, -1]),
    }


def _summarize_worker(task: tuple[float, float, int, float, int, float]) -> dict[str, np.ndarray | float]:
    z_final, mh_final, n_tracks, z_start_max, n_grid, lookback_max_myr = task
    return summarize_histories(
        z_final,
        mh_final=mh_final,
        n_tracks=n_tracks,
        z_start_max=z_start_max,
        n_grid=n_grid,
        lookback_max_myr=lookback_max_myr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MAH and recent SFR histories for four redshifts.")
    parser.add_argument("--mh-final", type=float, default=MH_FINAL)
    parser.add_argument("--n-tracks", type=int, default=N_TRACKS)
    parser.add_argument("--z-start-max", type=float, default=Z_START_MAX)
    parser.add_argument("--n-grid", type=int, default=N_GRID)
    parser.add_argument("--lookback-max-myr", type=float, default=LOOKBACK_MAX_MYR)
    parser.add_argument("--mah-ymin", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = parser.parse_args()

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = reserve_output_path(output_dir / "mah_sfr_four_z.png")
    pdf_path = reserve_output_path(output_dir / "mah_sfr_four_z.pdf")
    txt_path = reserve_output_path(output_dir / "mah_sfr_four_z.txt")

    t0 = time.perf_counter()
    tasks = [
        (
            float(z_value),
            float(args.mh_final),
            int(args.n_tracks),
            float(args.z_start_max),
            int(args.n_grid),
            float(args.lookback_max_myr),
        )
        for z_value in Z_VALUES
    ]

    summaries: list[dict[str, np.ndarray | float]] = []
    max_workers = max(1, min(int(args.workers), len(tasks)))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_summarize_worker, task): task[0] for task in tasks}
        for future in as_completed(future_map):
            summaries.append(future.result())
    summaries.sort(key=lambda item: float(item["z_final"]))

    fig, (ax_mah, ax_sfr) = plt.subplots(
        2,
        1,
        figsize=(9.0, 8.5),
        constrained_layout=True,
    )
    colors = {
        6.0: "#1f77b4",
        8.0: "#ff7f0e",
        10.0: "#2ca02c",
        12.5: "#d62728",
    }

    for summary in summaries:
        z_final = float(summary["z_final"])
        color = colors[z_final]

        z_grid = np.asarray(summary["z_grid"], dtype=float)
        mah_p16 = np.asarray(summary["mah_p16"], dtype=float)
        mah_p50 = np.asarray(summary["mah_p50"], dtype=float)
        mah_p84 = np.asarray(summary["mah_p84"], dtype=float)
        mah_finite = np.isfinite(mah_p16) & np.isfinite(mah_p50) & np.isfinite(mah_p84)

        ax_mah.fill_between(z_grid[mah_finite], mah_p16[mah_finite], mah_p84[mah_finite], color=color, alpha=0.18)
        ax_mah.plot(z_grid[mah_finite], mah_p50[mah_finite], color=color, lw=2.2, label=rf"$z_{{\rm obs}}={z_final:g}$")

        lookback_myr = np.asarray(summary["lookback_myr"], dtype=float)
        recent_mask = np.asarray(summary["recent_mask"], dtype=bool)
        sfr_p16 = np.asarray(summary["sfr_p16"], dtype=float)
        sfr_p50 = np.asarray(summary["sfr_p50"], dtype=float)
        sfr_p84 = np.asarray(summary["sfr_p84"], dtype=float)
        sfr_finite = recent_mask & np.isfinite(sfr_p16) & np.isfinite(sfr_p50) & np.isfinite(sfr_p84)

        ax_sfr.fill_between(
            lookback_myr[sfr_finite],
            sfr_p16[sfr_finite],
            sfr_p84[sfr_finite],
            color=color,
            alpha=0.18,
        )
        ax_sfr.plot(
            lookback_myr[sfr_finite],
            sfr_p50[sfr_finite],
            color=color,
            lw=2.2,
            label=rf"$z_{{\rm obs}}={z_final:g}$",
        )

    ax_mah.set_yscale("log")
    ax_mah.set_xlim(float(args.z_start_max), min(Z_VALUES))
    ax_mah.set_ylim(float(args.mah_ymin), None)
    ax_mah.set_ylabel(r"$M_{\rm h}(z)\ [{\rm M}_\odot]$")
    ax_mah.set_xlabel("Redshift z")
    ax_mah.set_title("Median MAH and recent SFR histories for " + rf"$M_{{\rm h,final}}={float(args.mh_final):.0e}\,{{\rm M_\odot}}$")
    ax_mah.grid(True, which="both", alpha=0.25)
    ax_mah.legend(frameon=True)

    ax_sfr.set_yscale("log")
    ax_sfr.set_xlim(float(args.lookback_max_myr), 0.0)
    ax_sfr.set_ylabel(r"${\rm SFR}\ [{\rm M_\odot\,yr^{-1}}]$")
    ax_sfr.set_xlabel("Lookback time before observation [Myr]")
    ax_sfr.grid(True, which="both", alpha=0.25)
    ax_sfr.legend(frameon=True)

    fig.savefig(png_path, dpi=500)
    fig.savefig(pdf_path)

    elapsed = time.perf_counter() - t0
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write(f"MH_FINAL: {float(args.mh_final)}\n")
        handle.write(f"Z_VALUES: {Z_VALUES}\n")
        handle.write(f"N_TRACKS: {int(args.n_tracks)}\n")
        handle.write(f"Z_START_MAX: {float(args.z_start_max)}\n")
        handle.write(f"N_GRID: {int(args.n_grid)}\n")
        handle.write(f"LOOKBACK_MAX_MYR: {float(args.lookback_max_myr)}\n")
        handle.write(f"MAH_YMIN: {float(args.mah_ymin)}\n")
        handle.write(f"workers: {max_workers}\n")
        handle.write(f"elapsed_seconds: {elapsed:.6f}\n")
        handle.write(f"png_path: {png_path.resolve()}\n")
        handle.write(f"pdf_path: {pdf_path.resolve()}\n")
        for summary in summaries:
            handle.write(f"z={float(summary['z_final']):g}, t_obs_gyr={float(summary['t_obs_gyr']):.6f}\n")

    print(f"saved_png={png_path.resolve()}")
    print(f"saved_pdf={pdf_path.resolve()}")
    print(f"saved_txt={txt_path.resolve()}")


if __name__ == "__main__":
    main()
