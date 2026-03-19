#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from massfunc import Mass_func

from uvlf import compute_dust_attenuated_uvlf
from uvlf_compare_no_puv_to_dust import (
    LOGM_MAX,
    LOGM_MIN,
    MUV_MAX,
    MUV_MIN,
    build_intrinsic_uvlf,
    run_single_mass_compare,
)


def format_redshift_tag(z_value: float) -> str:
    return f"{z_value:g}".replace(".", "p")


def compute_uvlf_pair(
    z_obs: float,
    n_mass: int,
    n_tracks: int,
    bins: np.ndarray,
    workers: int,
    random_seed: int,
    z_start_max: float,
    n_grid: int,
    sampler: str,
    enable_time_delay: bool,
    ssp_file: str,
) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(random_seed)
    hmf = Mass_func()
    hmf.sigma2_interpolation_set()
    hmf.dsig2dm_interpolation_set()

    log_mh = rng.uniform(LOGM_MIN, LOGM_MAX, size=n_mass)
    mh = np.power(10.0, log_mh)
    dndm = np.asarray(hmf.dndmst(mh, z_obs), dtype=float)
    dndlogm = mh * np.log(10.0) * dndm
    mass_weight = (LOGM_MAX - LOGM_MIN) * dndlogm / n_mass

    total_samples = n_mass * n_tracks
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
            ssp_file,
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
            futures = [executor.submit(run_single_mass_compare, task) for task in tasks]
            for future in as_completed(futures):
                mass_index, _, ssp_chunk, inst_chunk, _, _ = future.result()
                start = mass_index * n_tracks
                stop = start + n_tracks
                sample_weight[start:stop] = mass_weight[mass_index] / n_tracks
                ssp_luminosity[start:stop] = ssp_chunk
                instant_luminosity[start:stop] = inst_chunk

    ssp_intrinsic_muv, ssp_intrinsic_phi = build_intrinsic_uvlf(ssp_luminosity, sample_weight, bins)
    inst_intrinsic_muv, inst_intrinsic_phi = build_intrinsic_uvlf(instant_luminosity, sample_weight, bins)

    dust_grid = np.linspace(MUV_MIN, MUV_MAX, 400, dtype=float)
    ssp_dust = compute_dust_attenuated_uvlf(
        intrinsic_muv=ssp_intrinsic_muv,
        intrinsic_phi=ssp_intrinsic_phi,
        z=z_obs,
        muv_obs=dust_grid,
        clip_to_bounds=False,
    )
    inst_dust = compute_dust_attenuated_uvlf(
        intrinsic_muv=inst_intrinsic_muv,
        intrinsic_phi=inst_intrinsic_phi,
        z=z_obs,
        muv_obs=dust_grid,
        clip_to_bounds=False,
    )

    dust_phi_our = np.asarray(ssp_dust["phi_obs"], dtype=float)
    dust_phi_inst = np.asarray(inst_dust["phi_obs"], dtype=float)
    dust_ratio = np.full_like(dust_phi_our, np.nan)
    valid_dust = np.isfinite(dust_phi_our) & np.isfinite(dust_phi_inst) & (dust_phi_inst > 0.0)
    dust_ratio[valid_dust] = dust_phi_our[valid_dust] / dust_phi_inst[valid_dust]

    intrinsic_ratio = np.full_like(ssp_intrinsic_phi, np.nan)
    valid_intrinsic = np.isfinite(ssp_intrinsic_phi) & np.isfinite(inst_intrinsic_phi) & (inst_intrinsic_phi > 0.0)
    intrinsic_ratio[valid_intrinsic] = ssp_intrinsic_phi[valid_intrinsic] / inst_intrinsic_phi[valid_intrinsic]

    return {
        "z_obs": float(z_obs),
        "dust_muv": dust_grid,
        "dust_phi_our": dust_phi_our,
        "dust_phi_inst": dust_phi_inst,
        "dust_ratio": dust_ratio,
        "intrinsic_muv": ssp_intrinsic_muv,
        "intrinsic_phi_our": ssp_intrinsic_phi,
        "intrinsic_phi_inst": inst_intrinsic_phi,
        "intrinsic_ratio": intrinsic_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether the dusty UVLF compare plots are visually misleading or numerically inconsistent."
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

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data_save")
    data_dir.mkdir(parents=True, exist_ok=True)

    z_tags = "_".join(format_redshift_tag(z) for z in args.z_values)
    plot_path = outputs_dir / f"uvlf_plot_consistency_check_z{z_tags}.png"
    table_path = data_dir / f"uvlf_plot_consistency_check_z{z_tags}.tsv"
    summary_path = data_dir / f"uvlf_plot_consistency_check_z{z_tags}.txt"

    bins = np.linspace(MUV_MIN, MUV_MAX, args.bins + 1, dtype=float)
    ssp_file = "spectra-bin_byrne23/spectra-bin-imf135_300.BASEL.z001.a+00.dat"

    t0 = time.perf_counter()
    results = [
        compute_uvlf_pair(
            z_obs=float(z_value),
            n_mass=args.N_mass,
            n_tracks=args.n_tracks,
            bins=bins,
            workers=args.workers,
            random_seed=args.random_seed + 1000 * index,
            z_start_max=args.z_start_max,
            n_grid=args.n_grid,
            sampler=args.sampler,
            enable_time_delay=args.enable_time_delay,
            ssp_file=ssp_file,
        )
        for index, z_value in enumerate(args.z_values)
    ]
    elapsed = time.perf_counter() - t0

    fig, axes = plt.subplots(len(results), 2, figsize=(11.8, 4.4 * len(results)), constrained_layout=True)
    if len(results) == 1:
        axes = np.asarray([axes], dtype=object)

    rows: list[str] = ["z\tkind\tMuv\tour_phi\tinst_phi\tratio"]
    summary_lines = [
        f"z_values: {list(map(float, args.z_values))}",
        f"N_mass: {args.N_mass}",
        f"n_tracks: {args.n_tracks}",
        f"bins: {args.bins}",
        f"workers: {args.workers}",
        f"elapsed_seconds: {elapsed:.3f}",
    ]

    for row_index, result in enumerate(results):
        z_obs = float(result["z_obs"])
        dust_muv = np.asarray(result["dust_muv"], dtype=float)
        dust_phi_our = np.asarray(result["dust_phi_our"], dtype=float)
        dust_phi_inst = np.asarray(result["dust_phi_inst"], dtype=float)
        dust_ratio = np.asarray(result["dust_ratio"], dtype=float)
        intrinsic_muv = np.asarray(result["intrinsic_muv"], dtype=float)
        intrinsic_phi_our = np.asarray(result["intrinsic_phi_our"], dtype=float)
        intrinsic_phi_inst = np.asarray(result["intrinsic_phi_inst"], dtype=float)
        intrinsic_ratio = np.asarray(result["intrinsic_ratio"], dtype=float)

        ax_curve = axes[row_index, 0]
        finite_our = np.isfinite(dust_phi_our) & (dust_phi_our > 0.0)
        finite_inst = np.isfinite(dust_phi_inst) & (dust_phi_inst > 0.0)
        ax_curve.plot(dust_muv[finite_our], dust_phi_our[finite_our], color="#C0392B", linewidth=2.1, label="Our dusty UVLF")
        ax_curve.plot(dust_muv[finite_inst], dust_phi_inst[finite_inst], color="#2E86DE", linestyle="--", linewidth=2.0, label="Instant dusty UVLF")
        ax_curve.set_yscale("log")
        ax_curve.set_xlim(MUV_MIN, MUV_MAX)
        ax_curve.set_ylim(1.0e-8, 1.0)
        ax_curve.set_xlabel(r"$M_{\rm UV}^{\rm obs}$")
        ax_curve.set_ylabel(r"$\phi(M_{\rm UV})$")
        ax_curve.set_title(f"z = {z_obs:g}: dusty curves used in slide")
        ax_curve.tick_params(direction="in", top=True, right=True)
        ax_curve.minorticks_on()
        ax_curve.legend(frameon=False, fontsize=10.2, loc="lower left")

        ax_ratio = axes[row_index, 1]
        finite_dust_ratio = np.isfinite(dust_ratio)
        finite_intr_ratio = np.isfinite(intrinsic_ratio)
        ax_ratio.plot(dust_muv[finite_dust_ratio], dust_ratio[finite_dust_ratio], color="#C0392B", linewidth=2.1, label="dusty ratio")
        ax_ratio.plot(
            intrinsic_muv[finite_intr_ratio],
            intrinsic_ratio[finite_intr_ratio],
            color="#2E86DE",
            linestyle="--",
            linewidth=2.0,
            label="intrinsic ratio",
        )
        ax_ratio.axhline(1.0, color="0.3", linestyle=":", linewidth=1.2)
        ax_ratio.set_xlim(MUV_MIN, MUV_MAX)
        ax_ratio.set_ylim(0.2, 1.6)
        ax_ratio.set_xlabel(r"$M_{\rm UV}$")
        ax_ratio.set_ylabel("Our / Instant")
        ax_ratio.set_title(f"z = {z_obs:g}: ratio diagnostic")
        ax_ratio.tick_params(direction="in", top=True, right=True)
        ax_ratio.legend(frameon=False, fontsize=10.2, loc="best")

        median_dust_ratio = float(np.nanmedian(dust_ratio[finite_dust_ratio]))
        median_intr_ratio = float(np.nanmedian(intrinsic_ratio[finite_intr_ratio]))
        summary_lines.extend(
            [
                f"",
                f"z={z_obs:g}",
                f"median_dust_ratio: {median_dust_ratio:.6f}",
                f"median_intrinsic_ratio: {median_intr_ratio:.6f}",
            ]
        )

        for muv, our_phi, inst_phi, ratio in zip(dust_muv, dust_phi_our, dust_phi_inst, dust_ratio, strict=True):
            rows.append(f"{z_obs}\tdust\t{muv:.6f}\t{our_phi:.12e}\t{inst_phi:.12e}\t{ratio:.12e}")
        for muv, our_phi, inst_phi, ratio in zip(intrinsic_muv, intrinsic_phi_our, intrinsic_phi_inst, intrinsic_ratio, strict=True):
            rows.append(f"{z_obs}\tintrinsic\t{muv:.6f}\t{our_phi:.12e}\t{inst_phi:.12e}\t{ratio:.12e}")

    fig.suptitle("Dusty UVLF plot-consistency check", fontsize=15)
    fig.savefig(plot_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    table_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(plot_path.resolve())
    print(table_path.resolve())
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
