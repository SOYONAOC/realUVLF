#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uvlf import run_halo_uv_pipeline, uv_luminosity_to_muv


MASS_COLORS = {
    1.0e8: "#6C5CE7",
    1.0e9: "#00A8B5",
    1.0e10: "#E67E22",
    1.0e11: "#C0392B",
    1.0e12: "#1F618D",
}


def format_redshift_tag(z_value: float) -> str:
    return f"{z_value:g}".replace(".", "p")


def _run_chunk(args: tuple[float, float, int, int]) -> tuple[float, float, np.ndarray]:
    z_final, mh_final, n_tracks, seed = args
    result = run_halo_uv_pipeline(
        n_tracks=n_tracks,
        z_final=z_final,
        Mh_final=mh_final,
        workers=1,
        random_seed=seed,
    )
    muv = np.asarray(uv_luminosity_to_muv(result.uv_luminosities), dtype=float)
    return z_final, mh_final, muv[np.isfinite(muv)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot fixed-Mh Muv PDFs and bright-end tail probabilities for multiple redshifts."
    )
    parser.add_argument("--z-values", type=float, nargs="+", default=[6.0, 8.0, 10.0, 12.5])
    parser.add_argument("--mh-values", type=float, nargs="+", default=[1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e12])
    parser.add_argument("--n-tracks", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=180)
    parser.add_argument("--tail-threshold", type=float, default=-19.0)
    parser.add_argument("--ymin", type=float, default=1.0e-6)
    args = parser.parse_args()

    if args.n_tracks <= 0 or args.chunk_size <= 0 or args.n_bins <= 1:
        raise ValueError("n-tracks, chunk-size, and n-bins must be positive")

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data_save")
    data_dir.mkdir(parents=True, exist_ok=True)

    z_values = [float(x) for x in args.z_values]
    mh_values = [float(x) for x in args.mh_values]
    n_chunks = math.ceil(args.n_tracks / args.chunk_size)
    chunk_sizes = [args.chunk_size] * n_chunks
    chunk_sizes[-1] = args.n_tracks - args.chunk_size * (n_chunks - 1)

    tasks: list[tuple[float, float, int, int]] = []
    task_index = 0
    for z_final in z_values:
        for mh_final in mh_values:
            for chunk_n in chunk_sizes:
                tasks.append((z_final, mh_final, int(chunk_n), int(args.random_seed + task_index)))
                task_index += 1

    samples_by_key: dict[tuple[float, float], list[np.ndarray]] = {(z, mh): [] for z in z_values for mh in mh_values}
    completed = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [executor.submit(_run_chunk, task) for task in tasks]
        for future in as_completed(futures):
            z_final, mh_final, muv_chunk = future.result()
            samples_by_key[(z_final, mh_final)].append(muv_chunk)
            completed += 1
            if completed == 1 or completed % max(1, len(tasks) // 40) == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - t0
                print(f"chunks {completed}/{len(tasks)} elapsed={elapsed:.1f}s", flush=True)

    pdf_rows = ["z_final\tmh_final\tbin_left\tbin_right\tbin_center\tpdf"]
    stat_rows = ["z_final\tmh_final\tmean_muv\tstd_muv\tp16\tp50\tp84\tp99\tp_lt_tail"]
    summary_lines = [
        "Fixed-Mh Muv PDF comparison across redshift",
        "z_values: " + ", ".join(f"{z:g}" for z in z_values),
        "mh_values: " + ", ".join(f"{mh:.0e}" for mh in mh_values),
        f"n_tracks_each: {args.n_tracks}",
        f"chunk_size: {args.chunk_size}",
        f"workers: {args.workers}",
        f"random_seed: {args.random_seed}",
        f"n_bins: {args.n_bins}",
        f"tail_threshold: {args.tail_threshold}",
        f"ymin: {args.ymin}",
    ]

    for z_final in z_values:
        concatenated: dict[float, np.ndarray] = {}
        z_min = np.inf
        z_max = -np.inf
        for mh_final in mh_values:
            samples = samples_by_key[(z_final, mh_final)]
            muv = np.concatenate(samples) if samples else np.empty(0, dtype=float)
            if muv.size == 0:
                raise RuntimeError(f"no finite Muv samples for z={z_final:g}, Mh={mh_final:.3e}")
            concatenated[mh_final] = muv
            z_min = min(z_min, float(np.min(muv)))
            z_max = max(z_max, float(np.max(muv)))

        edges = np.linspace(z_min, z_max, int(args.n_bins) + 1, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])

        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)
        ax_pdf, ax_tail = axes

        for mh_final in mh_values:
            muv = concatenated[mh_final]
            density, hist_edges = np.histogram(muv, bins=edges, density=True)
            positive = density > 0.0
            color = MASS_COLORS.get(mh_final, None)
            label = rf"$M_h=10^{{{int(np.log10(mh_final))}}}\,M_\odot$"
            ax_pdf.step(centers[positive], density[positive], where="mid", linewidth=2.0, color=color, label=label)

            mean = float(np.mean(muv))
            std = float(np.std(muv))
            p16, p50, p84, p99 = np.percentile(muv, [16.0, 50.0, 84.0, 99.0])
            p_lt_tail = float(np.mean(muv < args.tail_threshold))

            stat_rows.append(
                "\t".join(
                    [
                        f"{z_final:.6f}",
                        f"{mh_final:.6e}",
                        f"{mean:.8f}",
                        f"{std:.8f}",
                        f"{float(p16):.8f}",
                        f"{float(p50):.8f}",
                        f"{float(p84):.8f}",
                        f"{float(p99):.8f}",
                        f"{p_lt_tail:.12e}",
                    ]
                )
            )
            summary_lines.extend(
                [
                    f"z={z_final:g}, Mh={mh_final:.0e}: mean={mean:.4f}, std={std:.4f}, p50={float(p50):.4f}, P(Muv<{args.tail_threshold:g})={p_lt_tail:.4e}",
                ]
            )

            for left, right, center, pdf in zip(hist_edges[:-1], hist_edges[1:], centers, density, strict=True):
                pdf_rows.append(
                    "\t".join(
                        [
                            f"{z_final:.6f}",
                            f"{mh_final:.6e}",
                            f"{left:.8f}",
                            f"{right:.8f}",
                            f"{center:.8f}",
                            f"{float(pdf):.12e}",
                        ]
                    )
                )

        tail_probs = [float(np.mean(concatenated[mh] < args.tail_threshold)) for mh in mh_values]
        ax_tail.plot(mh_values, tail_probs, marker="o", linewidth=2.0, color="#1F3A5F")
        for mh_final, p_tail in zip(mh_values, tail_probs, strict=True):
            color = MASS_COLORS.get(mh_final, "#1F3A5F")
            ax_tail.scatter([mh_final], [p_tail], s=54, color=color, zorder=3)

        ax_pdf.axvline(args.tail_threshold, color="0.3", linestyle="--", linewidth=1.3)
        ax_pdf.set_yscale("log")
        ax_pdf.set_ylim(float(args.ymin), 1.0)
        ax_pdf.set_xlabel(r"$M_{\rm UV}$")
        ax_pdf.set_ylabel("PDF")
        ax_pdf.set_title(rf"Fixed-$M_h$ dispersion at $z={z_final:g}$")
        ax_pdf.grid(True, which="both", alpha=0.22)
        ax_pdf.legend(frameon=False, fontsize=10)

        ax_tail.set_xscale("log")
        ax_tail.set_yscale("log")
        nonzero_tail = [p for p in tail_probs if p > 0.0]
        tail_ymin = max(1.0e-8, min(nonzero_tail) * 0.4) if nonzero_tail else 1.0e-8
        ax_tail.set_ylim(tail_ymin, 1.0)
        ax_tail.set_xlabel(r"$M_h\,[M_\odot]$")
        ax_tail.set_ylabel(rf"$P(M_{{\rm UV}}<{args.tail_threshold:g}\mid M_h)$")
        ax_tail.set_title(rf"Bright-end tail at $z={z_final:g}$")
        ax_tail.grid(True, which="both", alpha=0.22)

        png_path = outputs_dir / f"uvlf_fixed_mass_pdf_multi_z{format_redshift_tag(z_final)}.png"
        pdf_path = outputs_dir / f"uvlf_fixed_mass_pdf_multi_z{format_redshift_tag(z_final)}.pdf"
        fig.savefig(png_path, dpi=240, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        summary_lines.extend(
            [
                f"z={z_final:g}: png_path={png_path.resolve()}",
                f"z={z_final:g}: pdf_path={pdf_path.resolve()}",
                "",
            ]
        )

    elapsed = time.perf_counter() - t0
    summary_lines.append(f"elapsed_seconds: {elapsed:.3f}")

    stats_path = data_dir / "uvlf_fixed_mass_pdf_fourz_stats.tsv"
    hist_path = data_dir / "uvlf_fixed_mass_pdf_fourz_hist.tsv"
    summary_path = data_dir / "uvlf_fixed_mass_pdf_fourz.txt"
    stats_path.write_text("\n".join(stat_rows) + "\n", encoding="utf-8")
    hist_path.write_text("\n".join(pdf_rows) + "\n", encoding="utf-8")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    for z_final in z_values:
        print((outputs_dir / f"uvlf_fixed_mass_pdf_multi_z{format_redshift_tag(z_final)}.png").resolve())
    print(stats_path.resolve())
    print(hist_path.resolve())
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
