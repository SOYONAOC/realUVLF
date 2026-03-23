#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uvlf import run_halo_uv_pipeline


DEFAULT_Z_VALUES = [6.0, 8.0, 10.0, 12.5]
DEFAULT_MH_FINAL = 1.0e11
DEFAULT_N_TRACKS = 1000
DEFAULT_N_GRID = 480
DEFAULT_LOOKBACK_MAX_MYR = 100.0
DEFAULT_Z_START_MAX = 50.0


def _try_use_apj_style() -> None:
    try:
        plt.style.use("apj")
    except OSError:
        pass


def _tag_from_z(z_value: float) -> str:
    return f"z{str(float(z_value)).replace('.', 'p')}"


def _active_nanpercentile(values: np.ndarray, active: np.ndarray, percentile: float) -> np.ndarray:
    masked = np.where(active, values, np.nan)
    return np.nanpercentile(masked, percentile, axis=0)


def _summarize_case(
    z_obs: float,
    *,
    enable_time_delay: bool,
    mh_final: float,
    n_tracks: int,
    n_grid: int,
    lookback_max_myr: float,
    z_start_max: float,
    seed: int,
) -> dict[str, np.ndarray | float | bool]:
    t0 = time.perf_counter()
    result = run_halo_uv_pipeline(
        n_tracks=n_tracks,
        z_final=z_obs,
        Mh_final=mh_final,
        z_start_max=z_start_max,
        n_grid=n_grid,
        random_seed=seed,
        enable_time_delay=enable_time_delay,
        workers=1,
    )
    elapsed = time.perf_counter() - t0

    steps = int(result.redshift_grid.size)
    t_grid = np.asarray(result.sfr_tracks["t_gyr"], dtype=float).reshape(n_tracks, steps)
    mh_grid = np.asarray(result.sfr_tracks["Mh"], dtype=float).reshape(n_tracks, steps)
    sfr_grid = np.asarray(result.sfr_tracks["SFR"], dtype=float).reshape(n_tracks, steps)
    active_grid = np.asarray(result.sfr_tracks["active_flag"], dtype=bool).reshape(n_tracks, steps)

    lookback_myr = (t_grid[:, -1][:, None] - t_grid) * 1.0e3
    recent_mask = np.asarray(lookback_myr[0] <= lookback_max_myr, dtype=bool)

    return {
        "z_obs": float(z_obs),
        "enable_time_delay": bool(enable_time_delay),
        "lookback_myr": np.asarray(lookback_myr[0], dtype=float),
        "recent_mask": recent_mask,
        "mh_p16": np.nanpercentile(mh_grid, 16.0, axis=0),
        "mh_p50": np.nanpercentile(mh_grid, 50.0, axis=0),
        "mh_p84": np.nanpercentile(mh_grid, 84.0, axis=0),
        "sfr_p16": _active_nanpercentile(sfr_grid, active_grid, 16.0),
        "sfr_p50": _active_nanpercentile(sfr_grid, active_grid, 50.0),
        "sfr_p84": _active_nanpercentile(sfr_grid, active_grid, 84.0),
        "elapsed_seconds": float(elapsed),
    }


def _worker(task: tuple[float, bool, float, int, int, float, float, int]) -> dict[str, np.ndarray | float | bool]:
    z_obs, enable_time_delay, mh_final, n_tracks, n_grid, lookback_max_myr, z_start_max, seed = task
    return _summarize_case(
        z_obs,
        enable_time_delay=enable_time_delay,
        mh_final=mh_final,
        n_tracks=n_tracks,
        n_grid=n_grid,
        lookback_max_myr=lookback_max_myr,
        z_start_max=z_start_max,
        seed=seed,
    )


def _save_data(
    data_path: Path,
    no_delay_results: dict[float, dict[str, np.ndarray | float | bool]],
    delay_results: dict[float, dict[str, np.ndarray | float | bool]],
    *,
    mh_final: float,
    n_tracks: int,
    n_grid: int,
    lookback_max_myr: float,
) -> None:
    payload: dict[str, np.ndarray] = {
        "z_values": np.asarray(sorted(no_delay_results.keys()), dtype=float),
        "mh_final": np.array([float(mh_final)], dtype=float),
        "n_tracks": np.array([int(n_tracks)], dtype=int),
        "n_grid": np.array([int(n_grid)], dtype=int),
        "lookback_max_myr": np.array([float(lookback_max_myr)], dtype=float),
    }
    for z_obs in sorted(no_delay_results):
        tag = _tag_from_z(z_obs)
        for prefix, result in (("no_delay", no_delay_results[z_obs]), ("delay", delay_results[z_obs])):
            payload[f"{tag}_{prefix}_lookback_myr"] = np.asarray(result["lookback_myr"], dtype=float)
            payload[f"{tag}_{prefix}_recent_mask"] = np.asarray(result["recent_mask"], dtype=bool)
            payload[f"{tag}_{prefix}_mh_p16"] = np.asarray(result["mh_p16"], dtype=float)
            payload[f"{tag}_{prefix}_mh_p50"] = np.asarray(result["mh_p50"], dtype=float)
            payload[f"{tag}_{prefix}_mh_p84"] = np.asarray(result["mh_p84"], dtype=float)
            payload[f"{tag}_{prefix}_sfr_p16"] = np.asarray(result["sfr_p16"], dtype=float)
            payload[f"{tag}_{prefix}_sfr_p50"] = np.asarray(result["sfr_p50"], dtype=float)
            payload[f"{tag}_{prefix}_sfr_p84"] = np.asarray(result["sfr_p84"], dtype=float)
            payload[f"{tag}_{prefix}_elapsed_seconds"] = np.array([float(result["elapsed_seconds"])], dtype=float)
    np.savez_compressed(data_path, **payload)


def _write_summary(
    txt_path: Path,
    no_delay_results: dict[float, dict[str, np.ndarray | float | bool]],
    delay_results: dict[float, dict[str, np.ndarray | float | bool]],
) -> None:
    lines: list[str] = []
    for z_obs in sorted(no_delay_results):
        no_delay = no_delay_results[z_obs]
        delay = delay_results[z_obs]
        ratio = np.divide(
            np.asarray(delay["sfr_p50"], dtype=float),
            np.asarray(no_delay["sfr_p50"], dtype=float),
            out=np.full_like(np.asarray(no_delay["sfr_p50"], dtype=float), np.nan),
            where=np.asarray(no_delay["sfr_p50"], dtype=float) > 0.0,
        )
        recent_mask = np.asarray(no_delay["recent_mask"], dtype=bool)
        lookback = np.asarray(no_delay["lookback_myr"], dtype=float)
        valid = recent_mask & np.isfinite(ratio)
        lines.append(f"z={z_obs:g}")
        if np.any(valid):
            lines.append(f"  median_recent_sfr_ratio={float(np.nanmedian(ratio[valid])):.6f}")
            lines.append(f"  final_sfr_ratio={float(ratio[np.where(valid)[0][-1]]):.6f}")
            mh_p50 = np.asarray(no_delay["mh_p50"], dtype=float)
            lines.append(
                "  ratio_at_lookback_25_50_75_100_myr="
                + ", ".join(
                    f"{target:g}:{float(ratio[np.argmin(np.abs(lookback - target))]):.6f}"
                    for target in (25.0, 50.0, 75.0, 100.0)
                )
            )
            lines.append(
                "  median_Mh_at_lookback_25_50_75_100_myr="
                + ", ".join(
                    f"{target:g}:{float(mh_p50[np.argmin(np.abs(lookback - target))]):.6e}"
                    for target in (25.0, 50.0, 75.0, 100.0)
                )
            )
        lines.append(
            f"  timing_no_delay={float(no_delay['elapsed_seconds']):.2f}s "
            f"timing_delay={float(delay['elapsed_seconds']):.2f}s"
        )
        lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare delay and no-delay SFR histories at four redshifts.")
    parser.add_argument("--z-values", nargs="+", type=float, default=DEFAULT_Z_VALUES)
    parser.add_argument("--mh-final", type=float, default=DEFAULT_MH_FINAL)
    parser.add_argument("--n-tracks", type=int, default=DEFAULT_N_TRACKS)
    parser.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    parser.add_argument("--lookback-max-myr", type=float, default=DEFAULT_LOOKBACK_MAX_MYR)
    parser.add_argument("--z-start-max", type=float, default=DEFAULT_Z_START_MAX)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outputs/delay_sfr_four_z_compare",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data_save/delay_sfr_four_z_compare.npz",
    )
    args = parser.parse_args()

    _try_use_apj_style()

    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path).expanduser().resolve()
    data_path.parent.mkdir(parents=True, exist_ok=True)

    z_values = [float(z) for z in args.z_values]
    tasks = [
        (
            z_obs,
            enable_time_delay,
            float(args.mh_final),
            int(args.n_tracks),
            int(args.n_grid),
            float(args.lookback_max_myr),
            float(args.z_start_max),
            int(args.seed),
        )
        for z_obs in z_values
        for enable_time_delay in (False, True)
    ]

    results: list[dict[str, np.ndarray | float | bool]] = []
    max_workers = max(1, min(int(args.workers), len(tasks)))
    if max_workers == 1:
        for task in tasks:
            result = _worker(task)
            print(f"completed z={task[0]:g} delay={task[1]}", flush=True)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_worker, task): (task[0], task[1]) for task in tasks}
            for future in as_completed(future_map):
                z_obs, enable_time_delay = future_map[future]
                print(f"completed z={z_obs:g} delay={enable_time_delay}", flush=True)
                results.append(future.result())

    no_delay_results = {
        float(result["z_obs"]): result for result in results if not bool(result["enable_time_delay"])
    }
    delay_results = {
        float(result["z_obs"]): result for result in results if bool(result["enable_time_delay"])
    }

    fig, axes = plt.subplots(
        3,
        len(z_values),
        figsize=(5.0 * len(z_values), 10.0),
        constrained_layout=True,
        sharex="col",
    )
    if len(z_values) == 1:
        axes = np.asarray(axes).reshape(3, 1)

    for column, z_obs in enumerate(sorted(z_values)):
        no_delay = no_delay_results[z_obs]
        delay = delay_results[z_obs]

        lookback = np.asarray(no_delay["lookback_myr"], dtype=float)
        recent_mask = np.asarray(no_delay["recent_mask"], dtype=bool)
        mh_p16 = np.asarray(no_delay["mh_p16"], dtype=float)
        mh_p50 = np.asarray(no_delay["mh_p50"], dtype=float)
        mh_p84 = np.asarray(no_delay["mh_p84"], dtype=float)

        no_p16 = np.asarray(no_delay["sfr_p16"], dtype=float)
        no_p50 = np.asarray(no_delay["sfr_p50"], dtype=float)
        no_p84 = np.asarray(no_delay["sfr_p84"], dtype=float)
        de_p16 = np.asarray(delay["sfr_p16"], dtype=float)
        de_p50 = np.asarray(delay["sfr_p50"], dtype=float)
        de_p84 = np.asarray(delay["sfr_p84"], dtype=float)

        ratio = np.divide(de_p50, no_p50, out=np.full_like(no_p50, np.nan), where=no_p50 > 0.0)

        ax_mass = axes[0, column]
        mass_mask = recent_mask & np.isfinite(mh_p16) & np.isfinite(mh_p50) & np.isfinite(mh_p84)
        ax_mass.fill_between(lookback[mass_mask], mh_p16[mass_mask], mh_p84[mass_mask], color="0.5", alpha=0.18)
        ax_mass.plot(lookback[mass_mask], mh_p50[mass_mask], color="black", lw=2.0, label=r"$M_{\rm h}(t)$")
        ax_mass.set_yscale("log")
        ax_mass.set_xlim(float(args.lookback_max_myr), 0.0)
        ax_mass.grid(alpha=0.22)
        ax_mass.set_title(f"z = {z_obs:g}")
        if column == 0:
            ax_mass.set_ylabel(r"$M_{\rm h}\ [{\rm M_\odot}]$")
            ax_mass.legend(frameon=False, fontsize=10, loc="lower left")

        ax_top = axes[1, column]
        top_mask = recent_mask & np.isfinite(no_p16) & np.isfinite(no_p50) & np.isfinite(no_p84)
        ax_top.fill_between(lookback[top_mask], no_p16[top_mask], no_p84[top_mask], color="black", alpha=0.12)
        ax_top.plot(lookback[top_mask], no_p50[top_mask], color="black", lw=2.0, label="no delay")
        top_mask_delay = recent_mask & np.isfinite(de_p16) & np.isfinite(de_p50) & np.isfinite(de_p84)
        ax_top.fill_between(lookback[top_mask_delay], de_p16[top_mask_delay], de_p84[top_mask_delay], color="#c44e52", alpha=0.16)
        ax_top.plot(lookback[top_mask_delay], de_p50[top_mask_delay], color="#c44e52", lw=2.0, label="delay")
        ax_top.set_yscale("log")
        ax_top.set_xlim(float(args.lookback_max_myr), 0.0)
        ax_top.grid(alpha=0.22)
        if column == 0:
            ax_top.set_ylabel(r"${\rm SFR}\ [{\rm M_\odot\,yr^{-1}}]$")
            ax_top.legend(frameon=False, fontsize=10, loc="lower left")

        ax_bottom = axes[2, column]
        bottom_mask = recent_mask & np.isfinite(ratio)
        ax_bottom.plot(lookback[bottom_mask], ratio[bottom_mask], color="#1f77b4", lw=2.0)
        ax_bottom.axhline(1.0, color="0.4", ls="--", lw=1.0)
        ax_bottom.set_xlim(float(args.lookback_max_myr), 0.0)
        ax_bottom.set_ylim(0.2, 1.15)
        ax_bottom.grid(alpha=0.22)
        ax_bottom.set_xlabel("Lookback time before observation [Myr]")
        if column == 0:
            ax_bottom.set_ylabel("delay / no delay")

    fig.suptitle(
        rf"Halo-mass and SFR history comparison at fixed $M_{{\rm h,final}}={float(args.mh_final):.0e}\,{{\rm M_\odot}}$",
        fontsize=16,
    )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    txt_path = output_prefix.with_suffix(".txt")
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)

    _save_data(
        data_path,
        no_delay_results,
        delay_results,
        mh_final=float(args.mh_final),
        n_tracks=int(args.n_tracks),
        n_grid=int(args.n_grid),
        lookback_max_myr=float(args.lookback_max_myr),
    )
    _write_summary(txt_path, no_delay_results, delay_results)

    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")
    print(f"saved_txt={txt_path}")
    print(f"saved_npz={data_path}")


if __name__ == "__main__":
    main()
