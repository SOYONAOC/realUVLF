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

from uvlf_compare_no_puv_to_dust import (
    KUV_SSP_LONG,
    LOGM_MAX,
    LOGM_MIN,
    MUV_MAX,
    MUV_MIN,
    build_intrinsic_uvlf,
    compute_dust_attenuated_uvlf,
    run_single_mass_compare,
)
from massfunc import Mass_func


FONT_PATH = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


def reserve_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def local_slope(muv: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    muv = np.asarray(muv, dtype=float)
    phi = np.asarray(phi, dtype=float)
    mask = np.isfinite(muv) & np.isfinite(phi) & (phi > 0.0)
    if np.count_nonzero(mask) < 3:
        return np.array([], dtype=float), np.array([], dtype=float)
    x = muv[mask]
    y = np.log10(phi[mask])
    return x, np.gradient(y, x)


def interpolate_to_grid(x_src: np.ndarray, y_src: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    mask = np.isfinite(x_src) & np.isfinite(y_src)
    if np.count_nonzero(mask) < 2:
        return np.full_like(x_grid, np.nan, dtype=float)
    order = np.argsort(x_src[mask])
    x = x_src[mask][order]
    y = y_src[mask][order]
    return np.interp(x_grid, x, y, left=np.nan, right=np.nan)


def compute_uvlf_arrays(
    z_obs: float,
    N_mass: int,
    n_tracks: int,
    bins: np.ndarray,
    workers: int,
    random_seed: int,
    z_start_max: float,
    n_grid: int,
    sampler: str,
    enable_time_delay: bool,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    hmf = Mass_func()
    hmf.sigma2_interpolation_set()
    hmf.dsig2dm_interpolation_set()

    log_mh = rng.uniform(LOGM_MIN, LOGM_MAX, size=N_mass)
    mh = np.power(10.0, log_mh)
    dndm = np.asarray(hmf.dndmst(mh, z_obs), dtype=float)
    dndlogm = mh * np.log(10.0) * dndm
    mass_weight = (LOGM_MAX - LOGM_MIN) * dndlogm / N_mass

    total_samples = N_mass * n_tracks
    sample_weight = np.empty(total_samples, dtype=float)
    ssp_luminosity = np.empty(total_samples, dtype=float)
    instant_luminosity = np.empty(total_samples, dtype=float)

    tasks = [
        (
            mass_index,
            float(log_mass),
            float(mass),
            float(z_obs),
            int(n_tracks),
            float(z_start_max),
            int(n_grid),
            sampler,
            bool(enable_time_delay),
            "spectra-bin_byrne23/spectra-bin-imf135_300.BASEL.z001.a+00.dat",
            int(random_seed + mass_index),
        )
        for mass_index, (log_mass, mass) in enumerate(zip(log_mh, mh, strict=True))
    ]

    max_workers = max(1, int(workers))
    if max_workers == 1:
        iterator = (run_single_mass_compare(task) for task in tasks)
        for mass_index, _, ssp_chunk, inst_chunk, _, _ in iterator:
            start = mass_index * n_tracks
            stop = start + n_tracks
            sample_weight[start:stop] = mass_weight[mass_index] / n_tracks
            ssp_luminosity[start:stop] = ssp_chunk
            instant_luminosity[start:stop] = inst_chunk
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(run_single_mass_compare, task): task[0] for task in tasks}
            for future in as_completed(future_map):
                mass_index, _, ssp_chunk, inst_chunk, _, _ = future.result()
                start = mass_index * n_tracks
                stop = start + n_tracks
                sample_weight[start:stop] = mass_weight[mass_index] / n_tracks
                ssp_luminosity[start:stop] = ssp_chunk
                instant_luminosity[start:stop] = inst_chunk

    ssp_intrinsic_muv, ssp_intrinsic_phi = build_intrinsic_uvlf(ssp_luminosity, sample_weight, bins)
    inst_intrinsic_muv, inst_intrinsic_phi = build_intrinsic_uvlf(instant_luminosity, sample_weight, bins)

    dust_grid = np.linspace(MUV_MIN, MUV_MAX, 400, dtype=float)
    dust_result = compute_dust_attenuated_uvlf(
        intrinsic_muv=ssp_intrinsic_muv,
        intrinsic_phi=ssp_intrinsic_phi,
        z=z_obs,
        muv_obs=dust_grid,
        clip_to_bounds=False,
    )
    inst_dust_result = compute_dust_attenuated_uvlf(
        intrinsic_muv=inst_intrinsic_muv,
        intrinsic_phi=inst_intrinsic_phi,
        z=z_obs,
        muv_obs=dust_grid,
        clip_to_bounds=False,
    )

    return {
        "dust_muv": np.asarray(dust_result["Muv_obs"], dtype=float),
        "dust_phi_our": np.asarray(dust_result["phi_obs"], dtype=float),
        "dust_phi_inst": np.asarray(inst_dust_result["phi_obs"], dtype=float),
        "intrinsic_muv_our": np.asarray(ssp_intrinsic_muv, dtype=float),
        "intrinsic_phi_our": np.asarray(ssp_intrinsic_phi, dtype=float),
        "intrinsic_muv_inst": np.asarray(inst_intrinsic_muv, dtype=float),
        "intrinsic_phi_inst": np.asarray(inst_intrinsic_phi, dtype=float),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose why SSP-vs-instant differences map differently onto UVLFs at different redshifts."
    )
    parser.add_argument("--z-values", type=float, nargs="+", default=[6.0, 12.5])
    parser.add_argument("--N-mass", type=int, default=3000)
    parser.add_argument("--n-tracks", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--z-start-max", type=float, default=50.0)
    parser.add_argument("--n-grid", type=int, default=240)
    parser.add_argument("--sampler", type=str, default="mcbride")
    parser.add_argument("--enable-time-delay", action="store_true")
    args = parser.parse_args()

    data_dir = Path("data_save")
    outputs_dir = Path("outputs")
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    base_name = "uvlf_ratio_slope_diagnostic"
    if args.enable_time_delay:
        base_name += "_delay"

    tsv_path = reserve_output_path(data_dir / f"{base_name}.tsv")
    txt_path = reserve_output_path(data_dir / f"{base_name}.txt")
    png_path = reserve_output_path(outputs_dir / f"{base_name}.png")

    bins = np.linspace(MUV_MIN, MUV_MAX, args.bins + 1, dtype=float)
    grid_common = np.linspace(MUV_MIN, MUV_MAX, 400, dtype=float)

    t0 = time.perf_counter()
    rows: list[str] = []
    summary_lines: list[str] = [
        f"KUV_SSP_LONG: {KUV_SSP_LONG:.6e}",
        f"z_values: {list(args.z_values)}",
        f"N_mass: {args.N_mass}",
        f"n_tracks: {args.n_tracks}",
        f"workers: {args.workers}",
    ]

    fig, axes = plt.subplots(2, len(args.z_values), figsize=(5.8 * len(args.z_values), 7.2), constrained_layout=True)
    if len(args.z_values) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, z_obs in enumerate(args.z_values):
        result = compute_uvlf_arrays(
            z_obs=z_obs,
            N_mass=args.N_mass,
            n_tracks=args.n_tracks,
            bins=bins,
            workers=args.workers,
            random_seed=args.random_seed,
            z_start_max=args.z_start_max,
            n_grid=args.n_grid,
            sampler=args.sampler,
            enable_time_delay=args.enable_time_delay,
        )

        dust_ratio = result["dust_phi_our"] / result["dust_phi_inst"]
        dust_ratio[~np.isfinite(dust_ratio)] = np.nan

        intrinsic_phi_our_on_grid = interpolate_to_grid(result["intrinsic_muv_our"], result["intrinsic_phi_our"], grid_common)
        intrinsic_phi_inst_on_grid = interpolate_to_grid(result["intrinsic_muv_inst"], result["intrinsic_phi_inst"], grid_common)
        intrinsic_ratio = intrinsic_phi_our_on_grid / intrinsic_phi_inst_on_grid
        intrinsic_ratio[~np.isfinite(intrinsic_ratio)] = np.nan

        dust_slope_x, dust_slope = local_slope(result["dust_muv"], result["dust_phi_inst"])
        intrinsic_slope_x, intrinsic_slope = local_slope(grid_common, intrinsic_phi_inst_on_grid)

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        ax_top.plot(result["dust_muv"], dust_ratio, color="#C0392B", lw=2.2, label="Dusty UVLF ratio")
        ax_top.plot(grid_common, intrinsic_ratio, color="#2E86DE", lw=2.0, ls="--", label="Intrinsic UVLF ratio")
        ax_top.axhline(1.0, color="0.3", lw=1.0, alpha=0.7)
        ax_top.set_xlim(MUV_MIN, MUV_MAX)
        ax_top.set_ylim(0.25, 1.6)
        ax_top.set_title(f"z = {z_obs:g}")
        ax_top.set_ylabel(r"$\phi_{\rm our}/\phi_{\rm inst}$")
        ax_top.grid(True, alpha=0.25)
        ax_top.legend(frameon=False, fontsize=10.0, loc="best")

        ax_bot.plot(dust_slope_x, dust_slope, color="#111111", lw=2.0, label="Dusty local slope")
        ax_bot.plot(intrinsic_slope_x, intrinsic_slope, color="#555555", lw=1.8, ls="--", label="Intrinsic local slope")
        ax_bot.set_xlim(MUV_MIN, MUV_MAX)
        ax_bot.set_ylabel(r"$d\log_{10}\phi / dM$")
        ax_bot.set_xlabel(r"$M_{\rm UV}$")
        ax_bot.grid(True, alpha=0.25)
        ax_bot.legend(frameon=False, fontsize=10.0, loc="best")

        sample_mags = (-20.0, -19.0, -18.0, -17.0)
        for M in sample_mags:
            idx_d = np.argmin(np.abs(result["dust_muv"] - M))
            idx_i = np.argmin(np.abs(grid_common - M))
            rows.append(
                "\t".join(
                    [
                        f"{z_obs:g}",
                        f"{M:.2f}",
                        f"{dust_ratio[idx_d]:.6f}",
                        f"{intrinsic_ratio[idx_i]:.6f}",
                        f"{dust_slope[idx_d] if idx_d < len(dust_slope) else np.nan:.6f}",
                        f"{intrinsic_slope[idx_i] if idx_i < len(intrinsic_slope) else np.nan:.6f}",
                    ]
                )
            )

        summary_lines.extend(
            [
                f"z={z_obs:g} median_dust_ratio: {np.nanmedian(dust_ratio):.6f}",
                f"z={z_obs:g} median_intrinsic_ratio: {np.nanmedian(intrinsic_ratio):.6f}",
            ]
        )

    fig.suptitle("UVLF ratio and local slope diagnostics", fontsize=16)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    with tsv_path.open("w", encoding="utf-8") as handle:
        handle.write("z\tMuv\tdust_ratio\tintrinsic_ratio\tdust_slope\tintrinsic_slope\n")
        for row in rows:
            handle.write(row + "\n")

    elapsed = time.perf_counter() - t0
    summary_lines.extend(
        [
            f"elapsed_seconds: {elapsed:.6f}",
            f"tsv_path: {tsv_path.resolve()}",
            f"png_path: {png_path.resolve()}",
        ]
    )
    txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"saved_tsv={tsv_path.resolve()}")
    print(f"saved_txt={txt_path.resolve()}")
    print(f"saved_png={png_path.resolve()}")


if __name__ == "__main__":
    main()
