#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from mah import Cosmology
from mah.generator import generate_halo_histories
from sfr.calculator import DEFAULT_SFR_MODEL_PARAMETERS, compute_sfr_from_tracks
from ssp import compute_halo_uv_luminosity, load_uv1600_table
from uvlf import uv_luminosity_to_muv
from uvlf.pipeline import DEFAULT_SSP_FILE


FONT_PATH = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path("data_save")
OUTPUT_DIR = Path("outputs")
TSV_PATH = DATA_DIR / "ssp_vs_instant_ssplong_grid.tsv"
TXT_PATH = DATA_DIR / "ssp_vs_instant_ssplong_grid.txt"
PNG_PATH = OUTPUT_DIR / "ssp_vs_instant_ssplong_grid.png"
PROGRESS_PATH = OUTPUT_DIR / "ssp_vs_instant_ssplong_grid_progress.txt"

DEFAULT_Z_VALUES = [6.0, 8.0, 10.0, 12.5]
DEFAULT_LOG_MASS_VALUES = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
Z_START_MAX = 50.0
N_GRID = 240
KUV_SSP_LONG = 6.1e-29


def write_progress(path: Path, completed: int, total: int, elapsed_seconds: float) -> None:
    fraction = completed / total if total > 0 else 1.0
    filled = int(round(30 * fraction))
    bar = "#" * filled + "-" * (30 - filled)
    rate = completed / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    remaining = total - completed
    eta_seconds = remaining / rate if rate > 0.0 else float("inf")
    eta_text = f"{eta_seconds:.1f}s" if np.isfinite(eta_seconds) else "inf"
    text = (
        f"[{bar}] {completed}/{total} "
        f"({fraction * 100.0:.2f}%) "
        f"elapsed={elapsed_seconds:.1f}s "
        f"eta={eta_text}\n"
    )
    path.write_text(text, encoding="utf-8")


def summarize_same_sfh(
    z_final: float,
    mh_final: float,
    n_tracks: int,
    random_seed: int,
) -> dict[str, float]:
    redshift_grid = np.linspace(Z_START_MAX, z_final, N_GRID, dtype=float)
    cosmology = Cosmology()
    histories = generate_halo_histories(
        n_tracks=n_tracks,
        z_final=z_final,
        Mh_final=mh_final,
        z_start_max=Z_START_MAX,
        cosmology=cosmology,
        random_seed=random_seed,
        time_grid_mode="custom",
        custom_grid=redshift_grid,
        store_inactive_history=True,
        sampler="mcbride",
    )
    sfr_tracks = compute_sfr_from_tracks(
        histories.tracks,
        model_parameters=DEFAULT_SFR_MODEL_PARAMETERS,
    )

    ages_myr, luv_per_msun = load_uv1600_table(DEFAULT_SSP_FILE)
    ssp_age_grid_gyr = ages_myr / 1.0e3

    halo_ids = np.asarray(sfr_tracks["halo_id"], dtype=int)
    t_gyr = np.asarray(sfr_tracks["t_gyr"], dtype=float)
    mh = np.asarray(sfr_tracks["Mh"], dtype=float)
    sfr = np.asarray(sfr_tracks["SFR"], dtype=float)
    active = np.asarray(sfr_tracks["active_flag"], dtype=bool)

    ssp_luv_values: list[float] = []
    instant_luv_values: list[float] = []

    for halo_id in np.unique(halo_ids):
        mask = (halo_ids == halo_id) & active
        if not np.any(mask):
            continue

        t_used = t_gyr[mask]
        mh_used = mh[mask]
        sfr_used = sfr[mask]
        t_obs = float(t_used[-1])

        ssp_luv = compute_halo_uv_luminosity(
            t_obs=t_obs,
            t_history=t_used,
            mh_history=mh_used,
            sfr_history=sfr_used,
            ssp_age_grid=ssp_age_grid_gyr,
            ssp_luv_grid=luv_per_msun,
            M_min=0.0,
            t_z50=float(t_used[0]),
            time_unit_in_years=1.0e9,
        )
        instant_luv = float(sfr_used[-1] / KUV_SSP_LONG)

        ssp_luv_values.append(ssp_luv)
        instant_luv_values.append(instant_luv)

    ssp_luv_array = np.asarray(ssp_luv_values, dtype=float)
    instant_luv_array = np.asarray(instant_luv_values, dtype=float)
    ratio = ssp_luv_array / instant_luv_array
    delta_mag = np.asarray(uv_luminosity_to_muv(instant_luv_array), dtype=float) - np.asarray(
        uv_luminosity_to_muv(ssp_luv_array), dtype=float
    )

    return {
        "z": z_final,
        "Mh_final": mh_final,
        "n_valid_tracks": int(ssp_luv_array.size),
        "mean_luv_ratio": float(np.mean(ratio)),
        "median_luv_ratio": float(np.median(ratio)),
        "p16_luv_ratio": float(np.percentile(ratio, 16.0)),
        "p84_luv_ratio": float(np.percentile(ratio, 84.0)),
        "mean_delta_mag": float(np.mean(delta_mag)),
        "median_delta_mag": float(np.median(delta_mag)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute SSP / instant-long UV ratios on the same SFH with a wider halo-mass grid "
            "to verify whether the z=6 ~ 1.3 result is robust."
        )
    )
    parser.add_argument("--n-tracks", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    z_values = list(DEFAULT_Z_VALUES)
    log_mass_values = list(DEFAULT_LOG_MASS_VALUES)
    mass_values = [10.0 ** value for value in log_mass_values]

    tasks = [(z_value, mh_value, args.n_tracks, args.random_seed) for z_value in z_values for mh_value in mass_values]
    total = len(tasks)
    rows: list[dict[str, float]] = []
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_task = {
            executor.submit(summarize_same_sfh, z_value, mh_value, args.n_tracks, args.random_seed): (z_value, mh_value)
            for z_value, mh_value, _, _ in tasks
        }
        completed = 0
        for future in as_completed(future_to_task):
            rows.append(future.result())
            completed += 1
            write_progress(PROGRESS_PATH, completed, total, time.perf_counter() - t0)

    rows.sort(key=lambda item: (item["z"], item["Mh_final"]))

    header = [
        "z",
        "Mh_final",
        "n_valid_tracks",
        "mean_luv_ratio",
        "median_luv_ratio",
        "p16_luv_ratio",
        "p84_luv_ratio",
        "mean_delta_mag",
        "median_delta_mag",
    ]
    with TSV_PATH.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write(
                "\t".join(
                    [
                        f"{row['z']:g}",
                        f"{row['Mh_final']:.6e}",
                        f"{int(row['n_valid_tracks'])}",
                        f"{row['mean_luv_ratio']:.6f}",
                        f"{row['median_luv_ratio']:.6f}",
                        f"{row['p16_luv_ratio']:.6f}",
                        f"{row['p84_luv_ratio']:.6f}",
                        f"{row['mean_delta_mag']:.6f}",
                        f"{row['median_delta_mag']:.6f}",
                    ]
                )
                + "\n"
            )

    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(mass_values)))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), constrained_layout=True)

    for mass_value, color in zip(mass_values, colors):
        subset = [row for row in rows if np.isclose(row["Mh_final"], mass_value)]
        subset.sort(key=lambda item: item["z"])
        axes[0].plot(
            [row["z"] for row in subset],
            [row["mean_luv_ratio"] for row in subset],
            marker="o",
            lw=1.8,
            color=color,
            label=rf"$\log M_h={np.log10(mass_value):.1f}$",
        )
        axes[1].plot(
            [row["z"] for row in subset],
            [row["median_luv_ratio"] for row in subset],
            marker="o",
            lw=1.8,
            color=color,
            label=rf"$\log M_h={np.log10(mass_value):.1f}$",
        )

    for ax, title, ylabel in [
        (axes[0], "均值比：$L_{\\rm SSP}/L_{\\rm inst,long}$", "均值亮度比"),
        (axes[1], "中位数比：$L_{\\rm SSP}/L_{\\rm inst,long}$", "中位数亮度比"),
    ]:
        ax.set_xlabel("红移 z")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(min(z_values) - 0.2, max(z_values) + 0.2)
        ax.set_ylim(0.45, 1.45)
        ax.axhline(1.0, color="0.45", lw=1.1, ls="--", alpha=0.8)

    axes[1].legend(frameon=True, fontsize=8.5, ncol=1, loc="best")
    fig.suptitle("同一条 SFH 下：SSP 卷积相对 Instant-long 基准的偏离", fontsize=14)
    fig.savefig(PNG_PATH, dpi=220)

    elapsed = time.perf_counter() - t0
    z6_rows = [row for row in rows if np.isclose(row["z"], 6.0)]
    z125_rows = [row for row in rows if np.isclose(row["z"], 12.5)]
    with TXT_PATH.open("w", encoding="utf-8") as handle:
        handle.write(f"n_tracks: {args.n_tracks}\n")
        handle.write(f"workers: {args.workers}\n")
        handle.write(f"z_values: {z_values}\n")
        handle.write(f"log_mass_values: {log_mass_values}\n")
        handle.write(f"ssp_file: {DEFAULT_SSP_FILE}\n")
        handle.write(f"KUV_SSP_LONG: {KUV_SSP_LONG:.6e}\n")
        handle.write(f"elapsed_seconds: {elapsed:.6f}\n")
        handle.write(f"tsv_path: {TSV_PATH.resolve()}\n")
        handle.write(f"png_path: {PNG_PATH.resolve()}\n")
        handle.write("\n[z=6]\n")
        for row in z6_rows:
            handle.write(
                f"logMh={np.log10(row['Mh_final']):.1f}, "
                f"mean={row['mean_luv_ratio']:.6f}, "
                f"median={row['median_luv_ratio']:.6f}, "
                f"p16={row['p16_luv_ratio']:.6f}, "
                f"p84={row['p84_luv_ratio']:.6f}, "
                f"n_valid={int(row['n_valid_tracks'])}\n"
            )
        handle.write("\n[z=12.5]\n")
        for row in z125_rows:
            handle.write(
                f"logMh={np.log10(row['Mh_final']):.1f}, "
                f"mean={row['mean_luv_ratio']:.6f}, "
                f"median={row['median_luv_ratio']:.6f}, "
                f"p16={row['p16_luv_ratio']:.6f}, "
                f"p84={row['p84_luv_ratio']:.6f}, "
                f"n_valid={int(row['n_valid_tracks'])}\n"
            )

    print(f"saved_tsv={TSV_PATH.resolve()}")
    print(f"saved_txt={TXT_PATH.resolve()}")
    print(f"saved_png={PNG_PATH.resolve()}")


if __name__ == "__main__":
    main()
