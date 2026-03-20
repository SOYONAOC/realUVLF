#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from massfunc import Mass_func, SFRD
from scipy.interpolate import interp1d

from uvlf import compute_dust_attenuated_uvlf, run_halo_uv_pipeline, uv_luminosity_to_muv


MUV_MIN = -28.0
MUV_MAX = -10.0
LOGM_MIN = 9.0
LOGM_MAX = 13.0
KUV_SSP_LONG = 6.1e-29


class Zhang25DustBaseline:
    """Analytic Zhang+25-style baseline with the same fstar parameters."""

    def __init__(self, z: float, sfr_params: dict[str, float]) -> None:
        self.z = float(z)
        self.sfr_params = dict(sfr_params)
        self.mass_function = Mass_func()
        self.sfrd = SFRD()

    def fstar(self, mh: np.ndarray) -> np.ndarray:
        eps0 = self.sfr_params["epsilon_0"]
        mc = self.sfr_params["characteristic_mass"]
        beta = self.sfr_params["beta_star"]
        gamma = self.sfr_params["gamma_star"]
        mh = np.asarray(mh, dtype=float)
        return 2.0 * eps0 / ((mh / mc) ** (-beta) + (mh / mc) ** gamma)

    def fduty(self, mh: np.ndarray) -> np.ndarray:
        mturn = 3.3e7 * ((1.0 + self.z) / 21.0) ** -1.5
        return np.exp(-mturn / np.asarray(mh, dtype=float))

    def mdot(self, mh: np.ndarray, z: float) -> np.ndarray:
        mh = np.asarray(mh, dtype=float)
        return 24.1 * (mh / 1.0e12) ** 1.094 * (1.0 + 1.75 * z) * np.sqrt(
            0.315 * (1.0 + z) ** 3 + 0.685
        )

    def sfr(self, mh: np.ndarray, z: float) -> np.ndarray:
        fb = 0.16
        return self.fstar(mh) * fb * self.mdot(mh, z)

    def luminosity_hat(self, mh: np.ndarray, z: float) -> np.ndarray:
        ka = 1.17e-28
        return self.sfr(mh, z) / ka

    def auv(self, muvobs: np.ndarray, z: float) -> np.ndarray:
        beta0 = -0.09 * z - 1.49
        dbeta = -0.007 * z - 0.09
        m0 = -19.5
        muvobs = np.asarray(muvobs, dtype=float)
        beta = beta0 + dbeta * (muvobs - m0)
        return np.maximum(4.85 + 2.10 * beta, 0.0)

    def _build_noscatter_interpolator(self, z: float, n_grid: int = 2000) -> interp1d:
        m_min = self.sfrd.M_vir(0.61, 1.0e4, z)
        m_max = self.sfrd.M_vir(0.61, 1.0e9, z)
        mh = np.logspace(np.log10(m_min), np.log10(m_max), n_grid)
        muv = np.asarray(uv_luminosity_to_muv(self.luminosity_hat(mh, z)), dtype=float)
        dndm = np.asarray(self.mass_function.dndmst(mh, z), dtype=float)
        dmuv_dmh = np.gradient(muv, mh)
        phi = self.fduty(mh) * dndm * np.abs(1.0 / dmuv_dmh)

        order = np.argsort(muv)
        muv_sorted = muv[order]
        phi_sorted = np.clip(phi[order], 1e-300, None)

        unique_mask = np.concatenate(([True], np.diff(muv_sorted) > 0.0))
        return interp1d(
            muv_sorted[unique_mask],
            np.log10(phi_sorted[unique_mask]),
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=True,
        )

    def uvlf_dust(self, muvobs: np.ndarray, z: float, n_grid: int = 2000) -> np.ndarray:
        muv = np.asarray(muvobs, dtype=float) - self.auv(muvobs, z)
        dmuvdmobs = 1.09 + 0.007 * z
        interp = self._build_noscatter_interpolator(z, n_grid=n_grid)
        logphi = interp(muv)
        phi = np.full_like(muv, np.nan, dtype=float)
        finite = np.isfinite(logphi)
        phi[finite] = 10.0 ** logphi[finite] * dmuvdmobs
        return phi

    def uvlf_intrinsic(self, muv: np.ndarray, z: float, n_grid: int = 2000) -> np.ndarray:
        interp = self._build_noscatter_interpolator(z, n_grid=n_grid)
        logphi = interp(np.asarray(muv, dtype=float))
        phi = np.full_like(np.asarray(muv, dtype=float), np.nan, dtype=float)
        finite = np.isfinite(logphi)
        phi[finite] = 10.0 ** logphi[finite]
        return phi


def format_redshift_tag(z_value: float) -> str:
    return f"{z_value:g}".replace(".", "p")


def load_observational_uvlf(z_value: float) -> list[dict[str, np.ndarray | str]]:
    obs_dir = Path("obsdata") / f"redshift_{format_redshift_tag(z_value)}"
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


def write_progress(progress_path: Path, completed: int, total: int, elapsed_seconds: float) -> None:
    fraction = completed / total
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
    progress_path.write_text(text, encoding="utf-8")


def run_single_mass_compare(
    args: tuple[int, float, float, float, float, int, float, int, str, bool, str, int | None]
) -> tuple[int, float, np.ndarray, np.ndarray, float, dict[str, float]]:
    (
        mass_index,
        log_mass,
        mass,
        z_obs,
        n_tracks,
        z_start_max,
        n_grid,
        sampler,
        enable_time_delay,
        ssp_file,
        random_seed,
    ) = args

    t0 = time.perf_counter()
    pipeline_result = run_halo_uv_pipeline(
        n_tracks=n_tracks,
        z_final=z_obs,
        Mh_final=mass,
        z_start_max=z_start_max,
        n_grid=n_grid,
        random_seed=random_seed,
        sampler=sampler,
        enable_time_delay=enable_time_delay,
        workers=1,
        ssp_file=ssp_file,
    )
    duration = time.perf_counter() - t0

    ssp_luminosity = np.asarray(pipeline_result.uv_luminosities, dtype=float)
    n_realizations = ssp_luminosity.size
    steps_per_halo = pipeline_result.redshift_grid.size
    sfr_grid = np.asarray(pipeline_result.sfr_tracks["SFR"], dtype=float).reshape(n_realizations, steps_per_halo)
    instant_sfr = sfr_grid[:, -1]
    instant_luminosity = instant_sfr / KUV_SSP_LONG

    sfr_params = {
        "epsilon_0": float(pipeline_result.metadata["sfr_model_parameters"]["epsilon_0"]),
        "characteristic_mass": float(pipeline_result.metadata["sfr_model_parameters"]["characteristic_mass"]),
        "beta_star": float(pipeline_result.metadata["sfr_model_parameters"]["beta_star"]),
        "gamma_star": float(pipeline_result.metadata["sfr_model_parameters"]["gamma_star"]),
    }
    return mass_index, log_mass, ssp_luminosity, instant_luminosity, duration, sfr_params


def build_intrinsic_uvlf(
    luminosity: np.ndarray,
    sample_weight: np.ndarray,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    muv = np.asarray(uv_luminosity_to_muv(luminosity), dtype=float)
    valid_mask = np.isfinite(muv) & np.isfinite(sample_weight)
    weighted_counts, used_edges = np.histogram(muv[valid_mask], bins=bins, weights=sample_weight[valid_mask])
    bin_width = np.diff(used_edges)
    phi = weighted_counts / bin_width
    bin_centers = 0.5 * (used_edges[:-1] + used_edges[1:])
    return bin_centers, phi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the current dusty UVLF against the same-model instantaneous SFR/K_UV baseline."
    )
    parser.add_argument("--z-obs", type=float, default=6.0)
    parser.add_argument("--N-mass", type=int, default=3000)
    parser.add_argument("--n-tracks", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--z-start-max", type=float, default=50.0)
    parser.add_argument("--n-grid", type=int, default=240)
    parser.add_argument("--sampler", type=str, default="mcbride")
    parser.add_argument("--enable-time-delay", action="store_true")
    parser.add_argument("--dust-only", action="store_true")
    args = parser.parse_args()

    z_obs = float(args.z_obs)
    z_tag = format_redshift_tag(z_obs)
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    plot_path = outputs_dir / f"uvlf_compare_no_puv_z{z_tag}.png"
    plot_pdf_path = outputs_dir / f"uvlf_compare_no_puv_z{z_tag}.pdf"
    txt_path = outputs_dir / f"uvlf_compare_no_puv_z{z_tag}.txt"
    progress_path = outputs_dir / f"uvlf_compare_no_puv_z{z_tag}_progress.txt"

    bins = np.linspace(MUV_MIN, MUV_MAX, args.bins + 1, dtype=float)
    rng = np.random.default_rng(args.random_seed)
    hmf = Mass_func()
    hmf.sigma2_interpolation_set()
    hmf.dsig2dm_interpolation_set()

    t0 = time.perf_counter()
    log_mh = rng.uniform(LOGM_MIN, LOGM_MAX, size=args.N_mass)
    mh = np.power(10.0, log_mh)
    dndm = np.asarray(hmf.dndmst(mh, z_obs), dtype=float)
    dndlogm = mh * np.log(10.0) * dndm
    mass_weight = (LOGM_MAX - LOGM_MIN) * dndlogm / args.N_mass

    total_samples = args.N_mass * args.n_tracks
    sample_weight = np.empty(total_samples, dtype=float)
    ssp_luminosity = np.empty(total_samples, dtype=float)
    instant_luminosity = np.empty(total_samples, dtype=float)
    per_mass_seconds = np.empty(args.N_mass, dtype=float)
    sfr_params: dict[str, float] | None = None

    tasks = [
        (
            mass_index,
            float(log_mass),
            float(mass),
            float(z_obs),
            int(args.n_tracks),
            float(args.z_start_max),
            int(args.n_grid),
            args.sampler,
            bool(args.enable_time_delay),
            "spectra-bin_byrne23/spectra-bin-imf135_300.BASEL.z001.a+00.dat",
            int(args.random_seed + mass_index),
        )
        for mass_index, (log_mass, mass) in enumerate(zip(log_mh, mh, strict=True))
    ]

    write_progress(progress_path, completed=0, total=args.N_mass, elapsed_seconds=0.0)
    progress_stride = max(1, args.N_mass // 100)
    completed = 0
    max_workers = max(1, int(args.workers))

    if max_workers == 1:
        iterator = (run_single_mass_compare(task) for task in tasks)
        for mass_index, _, ssp_chunk, inst_chunk, duration, params in iterator:
            start = mass_index * args.n_tracks
            stop = start + args.n_tracks
            sample_weight[start:stop] = mass_weight[mass_index] / args.n_tracks
            ssp_luminosity[start:stop] = ssp_chunk
            instant_luminosity[start:stop] = inst_chunk
            per_mass_seconds[mass_index] = duration
            if sfr_params is None:
                sfr_params = params
            completed += 1
            if completed == args.N_mass or completed % progress_stride == 0:
                write_progress(progress_path, completed=completed, total=args.N_mass, elapsed_seconds=time.perf_counter() - t0)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(run_single_mass_compare, task): task[0] for task in tasks}
            for future in as_completed(future_map):
                mass_index, _, ssp_chunk, inst_chunk, duration, params = future.result()
                start = mass_index * args.n_tracks
                stop = start + args.n_tracks
                sample_weight[start:stop] = mass_weight[mass_index] / args.n_tracks
                ssp_luminosity[start:stop] = ssp_chunk
                instant_luminosity[start:stop] = inst_chunk
                per_mass_seconds[mass_index] = duration
                if sfr_params is None:
                    sfr_params = params
                completed += 1
                if completed == args.N_mass or completed % progress_stride == 0:
                    write_progress(progress_path, completed=completed, total=args.N_mass, elapsed_seconds=time.perf_counter() - t0)

    if sfr_params is None:
        raise RuntimeError("failed to recover SFR model parameters from pipeline output")

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
    dust_muv = np.asarray(dust_result["Muv_obs"], dtype=float)
    dust_phi = np.asarray(dust_result["phi_obs"], dtype=float)
    inst_phi = np.asarray(inst_dust_result["phi_obs"], dtype=float)

    if args.dust_only:
        fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.9), constrained_layout=True)
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)
        ax = axes[0]
    finite_dust = np.isfinite(dust_phi) & (dust_phi > 0.0)
    finite_inst = np.isfinite(inst_phi) & (inst_phi > 0.0)
    ax.plot(
        dust_muv[finite_dust],
        dust_phi[finite_dust],
        color="#C0392B",
        linewidth=2.1,
        label="Our model",
        zorder=3,
    )
    ax.plot(
        dust_muv[finite_inst],
        inst_phi[finite_inst],
        color="#2E86DE",
        linestyle="--",
        linewidth=2.0,
        label="Instant",
        zorder=1,
    )

    for obs in load_observational_uvlf(z_obs):
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
    ax.set_title(f"Dusty UVLF at z = {z_obs:g}")
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
    ax.legend(frameon=False, fontsize=11.2, loc="lower left")

    if not args.dust_only:
        ax_intr = axes[1]
        finite_ssp_intrinsic = np.isfinite(ssp_intrinsic_phi) & (ssp_intrinsic_phi > 0.0)
        finite_inst_intrinsic = np.isfinite(inst_intrinsic_phi) & (inst_intrinsic_phi > 0.0)
        ax_intr.plot(
            ssp_intrinsic_muv[finite_ssp_intrinsic],
            ssp_intrinsic_phi[finite_ssp_intrinsic],
            color="#C0392B",
            linewidth=2.1,
            label="Our model",
            zorder=3,
        )
        ax_intr.plot(
            inst_intrinsic_muv[finite_inst_intrinsic],
            inst_intrinsic_phi[finite_inst_intrinsic],
            color="#2E86DE",
            linestyle="--",
            linewidth=2.0,
            label="Instant",
            zorder=1,
        )
        ax_intr.set_yscale("log")
        ax_intr.set_xlim(MUV_MIN, MUV_MAX)
        ax_intr.set_ylim(1.0e-8, 1.0)
        ax_intr.set_xlabel(r"$M_{\rm UV}^{\rm int}$")
        ax_intr.set_ylabel(r"$\phi(M_{\rm UV})$")
        ax_intr.set_title(f"Intrinsic UVLF at z = {z_obs:g}")
        ax_intr.tick_params(direction="in", top=True, right=True)
        ax_intr.minorticks_on()
        ax_intr.text(
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
            transform=ax_intr.transAxes,
            va="top",
            ha="left",
            fontsize=8.0,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
        )
        ax_intr.legend(frameon=False, fontsize=11.2, loc="lower left")

    fig.savefig(plot_pdf_path, bbox_inches="tight")
    fig.savefig(plot_path, dpi=500, bbox_inches="tight")
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
                f"KUV_SSP_LONG: {KUV_SSP_LONG}",
                "legend_lines: Our model / Instant",
                f"dust_only: {args.dust_only}",
                "panels: left=dusty, right=intrinsic" if not args.dust_only else "panels: dusty only",
                f"epsilon_0: {sfr_params['epsilon_0']}",
                f"characteristic_mass: {sfr_params['characteristic_mass']}",
                f"beta_star: {sfr_params['beta_star']}",
                f"gamma_star: {sfr_params['gamma_star']}",
                f"sampling_seconds: {elapsed}",
                f"mean_per_mass_seconds: {float(np.mean(per_mass_seconds))}",
                f"plot_path: {plot_path.resolve()}",
                f"plot_pdf_path: {plot_pdf_path.resolve()}",
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
