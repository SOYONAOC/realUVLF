#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uvlf import compute_dust_attenuated_uvlf, sample_uvlf_from_hmf


MUV_MIN = -28.0
MUV_MAX = -10.0
DEFAULT_Z_LIST = (6.0, 8.0, 10.0, 12.5)


def format_redshift_tag(z_value: float) -> str:
    text = f"{z_value:g}"
    return text.replace(".", "p")


def observational_directory_for_redshift(z_value: float) -> Path:
    return Path("obsdata") / f"redshift_{format_redshift_tag(z_value)}"


def reserve_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def load_observational_uvlf(obs_dir: Path) -> list[dict[str, np.ndarray | str]]:
    datasets: list[dict[str, np.ndarray | str]] = []
    for file_path in sorted(obs_dir.glob("*.npz")):
        data = np.load(file_path)
        label_array = np.asarray(data["label"])
        label = str(label_array[0]) if label_array.size > 0 else file_path.stem
        datasets.append(
            {
                "label": label,
                "Muv": np.asarray(data["muverr"], dtype=float),
                "phi": np.asarray(data["phierr"], dtype=float),
                "mag_err": np.asarray(data["mag_err"], dtype=float),
                "phi_err_lo": np.asarray(data["phi_err_lo"], dtype=float),
                "phi_err_up": np.asarray(data["phi_err_up"], dtype=float),
            }
        )
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full dusty UVLF test and save the plot.")
    parser.add_argument("--z-obs", type=float, nargs="+", default=list(DEFAULT_Z_LIST))
    parser.add_argument("--N-mass", type=int, default=3000)
    parser.add_argument("--n-tracks", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for z_obs in args.z_obs:
        z_obs = float(z_obs)
        z_tag = format_redshift_tag(z_obs)
        obs_dir = observational_directory_for_redshift(z_obs)
        if not obs_dir.exists():
            raise FileNotFoundError(f"observational directory not found for z={z_obs:g}: {obs_dir}")

        plot_path = reserve_output_path(outputs_dir / f"dust_uvlf_z{z_tag}_full_test.png")
        txt_path = reserve_output_path(outputs_dir / f"dust_uvlf_z{z_tag}_full_test.txt")
        progress_path = reserve_output_path(outputs_dir / f"dust_uvlf_z{z_tag}_full_test_progress.txt")

        t0 = time.perf_counter()
        result = sample_uvlf_from_hmf(
            z_obs=z_obs,
            N_mass=args.N_mass,
            n_tracks=args.n_tracks,
            bins=np.linspace(MUV_MIN, MUV_MAX, args.bins + 1, dtype=float),
            pipeline_workers=max(1, args.workers),
            random_seed=args.random_seed,
            progress_path=progress_path,
        )
        intrinsic_muv = np.asarray(result.uvlf["bin_centers"], dtype=float)
        intrinsic_phi = np.asarray(result.uvlf["phi"], dtype=float)
        sfr_params = dict(result.metadata["sfr_model_parameters"])

        dust_result = compute_dust_attenuated_uvlf(
            intrinsic_muv=intrinsic_muv,
            intrinsic_phi=intrinsic_phi,
            z=z_obs,
            muv_obs=np.linspace(MUV_MIN, MUV_MAX, 400, dtype=float),
            clip_to_bounds=False,
        )
        dust_muv = np.asarray(dust_result["Muv_obs"], dtype=float)
        dust_phi = np.asarray(dust_result["phi_obs"], dtype=float)
        dust_positive = np.isfinite(dust_phi) & (dust_phi > 0.0)
        intrinsic_positive = np.isfinite(intrinsic_phi) & (intrinsic_phi > 0.0)

        fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
        ax.plot(
            intrinsic_muv[intrinsic_positive],
            intrinsic_phi[intrinsic_positive],
            color="#2C5AA0",
            linewidth=1.3,
            linestyle="--",
            label="Intrinsic",
        )
        ax.plot(
            dust_muv[dust_positive],
            dust_phi[dust_positive],
            color="#A04D2C",
            linewidth=1.9,
            label="Dust-corrected",
        )

        for obs in load_observational_uvlf(obs_dir):
            ax.errorbar(
                np.asarray(obs["Muv"], dtype=float),
                np.asarray(obs["phi"], dtype=float),
                xerr=np.asarray(obs["mag_err"], dtype=float),
                yerr=np.vstack(
                    [
                        np.asarray(obs["phi_err_lo"], dtype=float),
                        np.asarray(obs["phi_err_up"], dtype=float),
                    ]
                ),
                fmt="o",
                markersize=4.0,
                capsize=2.5,
                linewidth=0.9,
                alpha=0.9,
                label=str(obs["label"]),
            )

        ax.set_yscale("log")
        ax.set_xlim(MUV_MIN, MUV_MAX)
        ax.set_ylim(1.0e-8, 1.0)
        ax.set_xlabel(r"$M_{\rm UV}^{\rm obs}$")
        ax.set_ylabel(r"$\phi(M_{\rm UV})$")
        ax.set_title(f"Full dusty UVLF test at z = {z_obs:g}")
        ax.tick_params(direction="in", top=True, right=True)
        ax.minorticks_on()
        ax.text(
            0.03,
            0.97,
            "\n".join(
                [
                    rf"$\epsilon_0 = {sfr_params['epsilon_0']:.3f}$",
                    rf"$M_c = 10^{{{np.log10(sfr_params['characteristic_mass']):.2f}}}\,M_\odot$",
                    rf"$\beta_\star = {sfr_params['beta_star']:.2f}$",
                    rf"$\gamma_\star = {sfr_params['gamma_star']:.2f}$",
                ]
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.0,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
        )
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
        fig.savefig(plot_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

        elapsed = time.perf_counter() - t0
        txt_path.write_text(
            "\n".join(
                [
                    f"z_obs: {z_obs}",
                    f"N_mass: {args.N_mass}",
                    f"n_tracks: {args.n_tracks}",
                    f"bins: {args.bins}",
                    f"workers: {args.workers}",
                    f"random_seed: {args.random_seed}",
                    f"epsilon_0: {sfr_params['epsilon_0']}",
                    f"characteristic_mass: {sfr_params['characteristic_mass']}",
                    f"beta_star: {sfr_params['beta_star']}",
                    f"gamma_star: {sfr_params['gamma_star']}",
                    f"sampling_seconds: {result.metadata['sampling_seconds']}",
                    f"total_seconds: {elapsed}",
                    f"transition_index: {int(dust_result['transition_index'][0])}",
                    f"plot_path: {plot_path.resolve()}",
                    f"progress_path: {progress_path.resolve()}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        print(plot_path.resolve())
        print(txt_path.resolve())


if __name__ == "__main__":
    main()
