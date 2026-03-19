#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from ssp import compute_halo_uv_luminosity, load_uv1600_table
from uvlf.pipeline import DEFAULT_SSP_FILE


FONT_PATH = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path("data_save")
OUTPUT_DIR = Path("outputs")
TSV_PATH = DATA_DIR / "ssp_kuv_constant_sfr_convergence.tsv"
TXT_PATH = DATA_DIR / "ssp_kuv_constant_sfr_convergence.txt"
PNG_PATH = OUTPUT_DIR / "ssp_kuv_constant_sfr_convergence.png"

KUV_REFERENCE = 6.1e-29


def constant_sfr_kuv(duration_myr: float, n_time: int, ssp_age_grid_gyr: np.ndarray, ssp_luv_grid: np.ndarray) -> tuple[float, float]:
    duration_gyr = duration_myr / 1.0e3
    t_history = np.linspace(0.0, duration_gyr, n_time, dtype=float)
    mh_history = np.full_like(t_history, 1.0e12, dtype=float)
    sfr_history = np.ones_like(t_history, dtype=float)
    luv = compute_halo_uv_luminosity(
        t_obs=float(t_history[-1]),
        t_history=t_history,
        mh_history=mh_history,
        sfr_history=sfr_history,
        ssp_age_grid=ssp_age_grid_gyr,
        ssp_luv_grid=ssp_luv_grid,
        M_min=0.0,
        t_z50=0.0,
        time_unit_in_years=1.0e9,
    )
    kuv = 1.0 / luv
    return luv, kuv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute K_UV(T) for strictly constant SFR using the exact same compute_halo_uv_luminosity code path."
    )
    parser.add_argument("--n-time", type=int, default=4000)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ages_myr, luv_per_msun = load_uv1600_table(DEFAULT_SSP_FILE)
    ssp_age_grid_gyr = ages_myr / 1.0e3

    duration_myr_grid = np.array(
        [1, 3, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300, 500, 800, 1000, 1500, 2000],
        dtype=float,
    )

    t0 = time.perf_counter()
    rows: list[tuple[float, float, float]] = []
    for duration_myr in duration_myr_grid:
        luv, kuv = constant_sfr_kuv(duration_myr, args.n_time, ssp_age_grid_gyr, luv_per_msun)
        rows.append((duration_myr, luv, kuv))
    elapsed = time.perf_counter() - t0

    with TSV_PATH.open("w", encoding="utf-8") as handle:
        handle.write("duration_myr\tluv\tkuv\tkuv_over_reference\n")
        for duration_myr, luv, kuv in rows:
            handle.write(f"{duration_myr:.6f}\t{luv:.6e}\t{kuv:.6e}\t{kuv / KUV_REFERENCE:.6f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)

    duration = np.asarray([item[0] for item in rows], dtype=float)
    kuv = np.asarray([item[2] for item in rows], dtype=float)

    axes[0].plot(duration, kuv, marker="o", lw=2.0, color="#1f77b4")
    axes[0].axhline(KUV_REFERENCE, color="#d62728", ls="--", lw=1.6, label=rf"$6.1\times10^{{-29}}$")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("恒定 SFR 持续时间 T [Myr]")
    axes[0].set_ylabel(r"$K_{\rm UV}(T)$")
    axes[0].set_title("严格恒定 SFR 下的 $K_{\\rm UV}(T)$")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=True, fontsize=10)

    axes[1].plot(duration, kuv / KUV_REFERENCE, marker="o", lw=2.0, color="#2ca02c")
    axes[1].axhline(1.0, color="0.45", ls="--", lw=1.2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("恒定 SFR 持续时间 T [Myr]")
    axes[1].set_ylabel(r"$K_{\rm UV}(T) / 6.1\times10^{-29}$")
    axes[1].set_title("相对参考值的收敛")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].set_ylim(0.8, 2.6)

    fig.suptitle("与项目完全同一路径的 SSP 卷积：恒定 SFR 基准检查", fontsize=14)
    fig.savefig(PNG_PATH, dpi=220)

    with TXT_PATH.open("w", encoding="utf-8") as handle:
        handle.write(f"ssp_file: {DEFAULT_SSP_FILE}\n")
        handle.write(f"n_time: {args.n_time}\n")
        handle.write(f"elapsed_seconds: {elapsed:.6f}\n")
        handle.write(f"KUV_reference: {KUV_REFERENCE:.6e}\n")
        handle.write(f"tsv_path: {TSV_PATH.resolve()}\n")
        handle.write(f"png_path: {PNG_PATH.resolve()}\n")
        handle.write("\n")
        for duration_myr, luv, kuv_value in rows:
            handle.write(
                f"T={duration_myr:.1f} Myr, "
                f"LUV={luv:.6e}, "
                f"KUV={kuv_value:.6e}, "
                f"KUV/reference={kuv_value / KUV_REFERENCE:.6f}\n"
            )

    print(f"saved_tsv={TSV_PATH.resolve()}")
    print(f"saved_txt={TXT_PATH.resolve()}")
    print(f"saved_png={PNG_PATH.resolve()}")


if __name__ == "__main__":
    main()
