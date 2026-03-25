#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from sfr import DEFAULT_SFR_MODEL_PARAMETERS, SFRModelParameters
from uvlf import compute_dust_attenuated_uvlf, sample_uvlf_from_hmf
from uvlf.pipeline import DEFAULT_SSP_FILE


DEFAULT_OLD_SSP_FILE = "spectra-bin_byrne23/spectra-bin-imf135_300.BASEL.z001.a+00.dat"
DEFAULT_Z_VALUES = (6.0, 8.0, 10.0, 12.5)
DEFAULT_LOGM_MIN = 9.0
DEFAULT_LOGM_MAX = 13.0
DEFAULT_MUV_MIN = -24.5
DEFAULT_MUV_MAX = -15.0


def _tag_from_z(z_value: float) -> str:
    return f"z{str(float(z_value)).replace('.', 'p')}"


def _default_output_prefix(project_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "data_save" / f"uvlf_imf_no_delay_compare_allz_{timestamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare no-delay UVLFs for the legacy IMF SSP and the current top-heavy SSP across redshifts."
    )
    parser.add_argument("--z-values", nargs="+", type=float, default=list(DEFAULT_Z_VALUES))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--N-mass", type=int, default=3000)
    parser.add_argument("--n-tracks", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--muv-min", type=float, default=DEFAULT_MUV_MIN)
    parser.add_argument("--muv-max", type=float, default=DEFAULT_MUV_MAX)
    parser.add_argument("--logM-min", type=float, default=DEFAULT_LOGM_MIN)
    parser.add_argument("--logM-max", type=float, default=DEFAULT_LOGM_MAX)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--z-start-max", type=float, default=50.0)
    parser.add_argument("--n-grid", type=int, default=240)
    parser.add_argument("--sampler", type=str, default="mcbride")
    parser.add_argument("--epsilon-0", type=float, default=DEFAULT_SFR_MODEL_PARAMETERS.epsilon_0)
    parser.add_argument("--old-ssp-file", type=str, default=DEFAULT_OLD_SSP_FILE)
    parser.add_argument("--topheavy-ssp-file", type=str, default=DEFAULT_SSP_FILE)
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--apply-dust", action="store_true")
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def _resolve_prefix(project_root: Path, output_prefix: str | None) -> Path:
    if output_prefix is None:
        return _default_output_prefix(project_root)
    prefix = Path(output_prefix).expanduser()
    if not prefix.is_absolute():
        prefix = (project_root / prefix).resolve()
    else:
        prefix = prefix.resolve()
    return prefix.with_suffix("") if prefix.suffix else prefix


def _run_single_imf_uvlf(
    *,
    z_obs: float,
    n_mass: int,
    n_tracks: int,
    bins: np.ndarray,
    logm_min: float,
    logm_max: float,
    random_seed: int,
    z_start_max: float,
    n_grid: int,
    sampler: str,
    workers: int,
    ssp_file: Path,
    progress_path: Path,
    print_progress: bool,
    sfr_model_parameters: SFRModelParameters,
) -> dict[str, np.ndarray | dict[str, object]]:
    result = sample_uvlf_from_hmf(
        z_obs=z_obs,
        N_mass=n_mass,
        n_tracks=n_tracks,
        random_seed=random_seed,
        quantity="Muv",
        bins=bins,
        logM_min=logm_min,
        logM_max=logm_max,
        z_start_max=z_start_max,
        n_grid=n_grid,
        sampler=sampler,
        enable_time_delay=False,
        pipeline_workers=workers,
        ssp_file=str(ssp_file),
        progress_path=progress_path,
        print_progress=print_progress,
        sfr_model_parameters=sfr_model_parameters,
    )
    return {
        "bin_edges": np.asarray(result.uvlf["bin_edges"], dtype=float),
        "bin_centers": np.asarray(result.uvlf["bin_centers"], dtype=float),
        "bin_width": np.asarray(result.uvlf["bin_width"], dtype=float),
        "weighted_counts": np.asarray(result.uvlf["weighted_counts"], dtype=float),
        "phi": np.asarray(result.uvlf["phi"], dtype=float),
        "metadata": result.metadata,
    }


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parent
    data_save_dir = project_root / "data_save"
    outputs_dir = project_root / "outputs"
    data_save_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = _resolve_prefix(project_root, args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    stem = output_prefix.name
    npz_path = output_prefix.with_suffix(".npz")
    summary_path = outputs_dir / f"{stem}.txt"

    old_ssp_file = (project_root / args.old_ssp_file).resolve()
    topheavy_ssp_file = (project_root / args.topheavy_ssp_file).resolve()
    if not old_ssp_file.exists():
        raise FileNotFoundError(f"Legacy SSP file not found: {old_ssp_file}")
    if not topheavy_ssp_file.exists():
        raise FileNotFoundError(f"Top-heavy SSP file not found: {topheavy_ssp_file}")

    z_values = [float(z) for z in args.z_values]
    if len(z_values) == 0:
        raise ValueError("at least one redshift must be provided")
    if args.N_mass < 1 or args.n_tracks < 1 or args.bins < 1:
        raise ValueError("N-mass, n-tracks, and bins must all be positive")
    if args.muv_max <= args.muv_min:
        raise ValueError("muv-max must be larger than muv-min")
    if args.logM_max <= args.logM_min:
        raise ValueError("logM-max must be larger than logM-min")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if not 0.0 <= float(args.epsilon_0) <= 1.0:
        raise ValueError("epsilon-0 must lie in [0, 1]")

    sfr_model_parameters = SFRModelParameters(
        epsilon_0=float(args.epsilon_0),
        characteristic_mass=DEFAULT_SFR_MODEL_PARAMETERS.characteristic_mass,
        beta_star=DEFAULT_SFR_MODEL_PARAMETERS.beta_star,
        gamma_star=DEFAULT_SFR_MODEL_PARAMETERS.gamma_star,
    )

    bins = np.linspace(float(args.muv_min), float(args.muv_max), int(args.bins) + 1, dtype=float)
    payload: dict[str, np.ndarray] = {
        "z_values": np.asarray(z_values, dtype=float),
        "shared_bin_edges": bins,
        "workers": np.asarray([int(args.workers)], dtype=int),
        "N_mass": np.asarray([int(args.N_mass)], dtype=int),
        "n_tracks": np.asarray([int(args.n_tracks)], dtype=int),
        "random_seed": np.asarray([int(args.random_seed)], dtype=int),
        "z_start_max": np.asarray([float(args.z_start_max)], dtype=float),
        "n_grid": np.asarray([int(args.n_grid)], dtype=int),
        "bins_count": np.asarray([int(args.bins)], dtype=int),
        "muv_min": np.asarray([float(args.muv_min)], dtype=float),
        "muv_max": np.asarray([float(args.muv_max)], dtype=float),
        "logM_min": np.asarray([float(args.logM_min)], dtype=float),
        "logM_max": np.asarray([float(args.logM_max)], dtype=float),
        "apply_dust": np.asarray([bool(args.apply_dust)]),
        "epsilon_0": np.asarray([float(args.epsilon_0)], dtype=float),
        "old_ssp_file": np.asarray([str(old_ssp_file)]),
        "topheavy_ssp_file": np.asarray([str(topheavy_ssp_file)]),
    }

    summary_lines = [
        f"python: {sys.executable}",
        f"npz_path: {npz_path}",
        f"summary_path: {summary_path}",
        f"old_ssp_file: {old_ssp_file}",
        f"topheavy_ssp_file: {topheavy_ssp_file}",
        f"workers: {args.workers}",
        f"N_mass: {args.N_mass}",
        f"n_tracks: {args.n_tracks}",
        f"bins: {args.bins}",
        f"muv_range: [{args.muv_min}, {args.muv_max}]",
        f"logM_range: [{args.logM_min}, {args.logM_max}]",
        f"z_values: {' '.join(str(z) for z in z_values)}",
        f"apply_dust: {bool(args.apply_dust)}",
        f"epsilon_0: {float(args.epsilon_0)}",
        "enable_time_delay: False",
        "",
    ]

    t0 = time.perf_counter()
    for z_index, z_obs in enumerate(z_values):
        z_tag = _tag_from_z(z_obs)
        seed = int(args.random_seed + 1000 * z_index)
        old_progress = outputs_dir / f"{stem}_{z_tag}_old_progress.txt"
        top_progress = outputs_dir / f"{stem}_{z_tag}_topheavy_progress.txt"

        print(
            f"Computing z={z_obs:g} with shared seed={seed}, workers={args.workers}, "
            f"old_ssp={old_ssp_file.name}, topheavy_ssp={topheavy_ssp_file.name}",
            flush=True,
        )

        old_result = _run_single_imf_uvlf(
            z_obs=z_obs,
            n_mass=int(args.N_mass),
            n_tracks=int(args.n_tracks),
            bins=bins,
            logm_min=float(args.logM_min),
            logm_max=float(args.logM_max),
            random_seed=seed,
            z_start_max=float(args.z_start_max),
            n_grid=int(args.n_grid),
            sampler=str(args.sampler),
            workers=int(args.workers),
            ssp_file=old_ssp_file,
            progress_path=old_progress,
            print_progress=bool(args.print_progress),
            sfr_model_parameters=sfr_model_parameters,
        )
        top_result = _run_single_imf_uvlf(
            z_obs=z_obs,
            n_mass=int(args.N_mass),
            n_tracks=int(args.n_tracks),
            bins=bins,
            logm_min=float(args.logM_min),
            logm_max=float(args.logM_max),
            random_seed=seed,
            z_start_max=float(args.z_start_max),
            n_grid=int(args.n_grid),
            sampler=str(args.sampler),
            workers=int(args.workers),
            ssp_file=topheavy_ssp_file,
            progress_path=top_progress,
            print_progress=bool(args.print_progress),
            sfr_model_parameters=sfr_model_parameters,
        )

        old_phi_intrinsic = np.asarray(old_result["phi"], dtype=float)
        top_phi_intrinsic = np.asarray(top_result["phi"], dtype=float)
        old_centers = np.asarray(old_result["bin_centers"], dtype=float)
        top_centers = np.asarray(top_result["bin_centers"], dtype=float)
        if not np.allclose(old_centers, top_centers, rtol=0.0, atol=0.0):
            raise RuntimeError(f"intrinsic bin centers differ between IMF runs at z={z_obs:g}")

        if args.apply_dust:
            old_dust = compute_dust_attenuated_uvlf(
                intrinsic_muv=old_centers,
                intrinsic_phi=old_phi_intrinsic,
                z=z_obs,
                muv_obs=old_centers,
            )
            top_dust = compute_dust_attenuated_uvlf(
                intrinsic_muv=top_centers,
                intrinsic_phi=top_phi_intrinsic,
                z=z_obs,
                muv_obs=top_centers,
            )
            old_phi_final = np.asarray(old_dust["phi_obs"], dtype=float)
            top_phi_final = np.asarray(top_dust["phi_obs"], dtype=float)
        else:
            old_phi_final = old_phi_intrinsic
            top_phi_final = top_phi_intrinsic

        phi_ratio = np.divide(
            top_phi_final,
            old_phi_final,
            out=np.full_like(top_phi_final, np.nan),
            where=old_phi_final > 0.0,
        )

        payload[f"{z_tag}_bin_edges"] = np.asarray(old_result["bin_edges"], dtype=float)
        payload[f"{z_tag}_bin_centers"] = np.asarray(old_result["bin_centers"], dtype=float)
        payload[f"{z_tag}_bin_width"] = np.asarray(old_result["bin_width"], dtype=float)
        payload[f"{z_tag}_old_intrinsic_weighted_counts"] = np.asarray(old_result["weighted_counts"], dtype=float)
        payload[f"{z_tag}_topheavy_intrinsic_weighted_counts"] = np.asarray(top_result["weighted_counts"], dtype=float)
        payload[f"{z_tag}_old_intrinsic_phi"] = old_phi_intrinsic
        payload[f"{z_tag}_topheavy_intrinsic_phi"] = top_phi_intrinsic
        payload[f"{z_tag}_old_weighted_counts"] = old_phi_final * np.asarray(old_result["bin_width"], dtype=float)
        payload[f"{z_tag}_topheavy_weighted_counts"] = top_phi_final * np.asarray(top_result["bin_width"], dtype=float)
        payload[f"{z_tag}_old_phi"] = old_phi_final
        payload[f"{z_tag}_topheavy_phi"] = top_phi_final
        payload[f"{z_tag}_phi_ratio_topheavy_over_old"] = np.asarray(phi_ratio, dtype=float)
        payload[f"{z_tag}_seed"] = np.asarray([seed], dtype=int)
        payload[f"{z_tag}_old_sampling_seconds"] = np.asarray(
            [float(old_result["metadata"]["sampling_seconds"])],
            dtype=float,
        )
        payload[f"{z_tag}_topheavy_sampling_seconds"] = np.asarray(
            [float(top_result["metadata"]["sampling_seconds"])],
            dtype=float,
        )

        overlap = np.isfinite(phi_ratio) & np.isfinite(old_phi_final) & np.isfinite(top_phi_final)
        summary_lines.append(f"z={z_obs:g}")
        summary_lines.append(f"  seed={seed}")
        summary_lines.append(f"  old_progress={old_progress}")
        summary_lines.append(f"  topheavy_progress={top_progress}")
        summary_lines.append(f"  old_sampling_seconds={float(old_result['metadata']['sampling_seconds']):.3f}")
        summary_lines.append(f"  topheavy_sampling_seconds={float(top_result['metadata']['sampling_seconds']):.3f}")
        summary_lines.append(f"  old_intrinsic_phi_median={float(np.nanmedian(old_phi_intrinsic[np.isfinite(old_phi_intrinsic)])):.6e}")
        summary_lines.append(f"  topheavy_intrinsic_phi_median={float(np.nanmedian(top_phi_intrinsic[np.isfinite(top_phi_intrinsic)])):.6e}")
        if np.any(overlap):
            summary_lines.append(f"  phi_ratio_median={float(np.nanmedian(phi_ratio[overlap])):.6f}")
            summary_lines.append(f"  phi_ratio_min={float(np.nanmin(phi_ratio[overlap])):.6f}")
            summary_lines.append(f"  phi_ratio_max={float(np.nanmax(phi_ratio[overlap])):.6f}")
        else:
            summary_lines.append("  phi_ratio_median=nan")
        summary_lines.append("")

        np.savez(npz_path, **payload)
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"Finished z={z_obs:g}; partial results saved to {npz_path}", flush=True)

    total_seconds = time.perf_counter() - t0
    payload["total_seconds"] = np.asarray([total_seconds], dtype=float)
    np.savez(npz_path, **payload)
    summary_lines.append(f"total_seconds={total_seconds:.3f}")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"saved_npz={npz_path}", flush=True)
    print(f"saved_summary={summary_path}", flush=True)
    print(f"total_seconds={total_seconds:.3f}", flush=True)


if __name__ == "__main__":
    main()
