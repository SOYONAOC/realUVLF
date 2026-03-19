#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from massfunc import Mass_func


MASS_COLORS = {
    1.0e8: "#6C5CE7",
    1.0e9: "#00A8B5",
    1.0e10: "#E67E22",
    1.0e11: "#C0392B",
    1.0e12: "#1F618D",
}


def format_redshift_tag(z_value: float) -> str:
    return f"{z_value:g}".replace(".", "p")


def _read_hist(path: Path) -> dict[tuple[float, float], dict[str, np.ndarray]]:
    rows: dict[tuple[float, float], list[tuple[float, float, float, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            z = float(row["z_final"])
            mh = float(row["mh_final"])
            rows[(z, mh)].append(
                (
                    float(row["bin_left"]),
                    float(row["bin_right"]),
                    float(row["bin_center"]),
                    float(row["pdf"]),
                )
            )

    out: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    for key, values in rows.items():
        arr = np.asarray(values, dtype=float)
        out[key] = {
            "bin_left": arr[:, 0],
            "bin_right": arr[:, 1],
            "bin_center": arr[:, 2],
            "pdf": arr[:, 3],
        }
    return out


def _read_stats(path: Path) -> dict[tuple[float, float], dict[str, float]]:
    out: dict[tuple[float, float], dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            z = float(row["z_final"])
            mh = float(row["mh_final"])
            out[(z, mh)] = {k: float(v) for k, v in row.items() if k not in {"z_final", "mh_final"}}
    return out


def _log_mass_edges(masses: np.ndarray) -> np.ndarray:
    logm = np.log10(masses)
    mids = 0.5 * (logm[1:] + logm[:-1])
    left = logm[0] - 0.5 * (logm[1] - logm[0])
    right = logm[-1] + 0.5 * (logm[-1] - logm[-2])
    return np.concatenate([[left], mids, [right]])


def _nice_log_lower(values: np.ndarray, floor: float = 1.0e-14) -> float:
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return floor
    exponent = math.floor(np.log10(float(np.min(positive))))
    return max(floor, 10.0 ** exponent)


def main() -> None:
    base = Path(".")
    outputs_dir = base / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir = base / "data_save"
    data_dir.mkdir(parents=True, exist_ok=True)

    hist = _read_hist(data_dir / "uvlf_fixed_mass_pdf_fourz_hist.tsv")
    stats = _read_stats(data_dir / "uvlf_fixed_mass_pdf_fourz_stats.tsv")

    z_values = sorted({z for z, _ in hist})
    mh_values = np.asarray(sorted({mh for _, mh in hist}), dtype=float)
    mass_edges = _log_mass_edges(mh_values)
    dlogm = np.diff(mass_edges)

    mf = Mass_func()

    summary_lines = [
        "HMF-weighted fixed-Mh UVLF contribution diagnostic",
        "Contribution approximation:",
        "phi_k(Muv) ~= P(Muv|Mh_k,z) * [dn/dlogMh]_k * Delta(logMh)_k",
        "using the saved conditional PDF from uvlf_fixed_mass_pdf_fourz.",
        "",
    ]
    stat_rows = [
        "z_final\tmh_final\tdlogm\tdndm\tdndlogm\tmass_interval_density\tp_lt_tail\ttail_contribution\ttail_fraction"
    ]

    for z_final in z_values:
        dndm = np.asarray(mf.dndmst(mh_values, z_final), dtype=float)
        dndlogm = np.log(10.0) * mh_values * dndm
        interval_density = dndlogm * dlogm

        first_key = (z_final, float(mh_values[0]))
        bin_center = hist[first_key]["bin_center"]
        weighted_curves: dict[float, np.ndarray] = {}
        total_curve = np.zeros_like(bin_center)

        tail_abs = []
        for mh, density_weight in zip(mh_values, interval_density, strict=True):
            key = (z_final, float(mh))
            pdf = hist[key]["pdf"]
            curve = pdf * density_weight
            weighted_curves[float(mh)] = curve
            total_curve += curve

            p_tail = stats[key]["p_lt_tail"]
            tail_abs.append(density_weight * p_tail)

        tail_abs_arr = np.asarray(tail_abs, dtype=float)
        tail_total = float(np.sum(tail_abs_arr))
        tail_frac_arr = tail_abs_arr / tail_total if tail_total > 0.0 else np.zeros_like(tail_abs_arr)

        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)
        ax_uvlf, ax_tail = axes

        for mh in mh_values:
            curve = weighted_curves[float(mh)]
            color = MASS_COLORS.get(float(mh), None)
            label = rf"$M_h=10^{{{int(np.log10(mh))}}}\,M_\odot$"
            positive = curve > 0.0
            if np.any(positive):
                ax_uvlf.plot(bin_center[positive], curve[positive], color=color, linewidth=2.0, label=label)

        total_positive = total_curve > 0.0
        if np.any(total_positive):
            ax_uvlf.plot(
                bin_center[total_positive],
                total_curve[total_positive],
                color="black",
                linewidth=2.6,
                label="Total",
            )

        ax_uvlf.axvline(-19.0, color="0.35", linestyle="--", linewidth=1.2)
        ax_uvlf.set_yscale("log")
        ax_uvlf.set_ylim(_nice_log_lower(total_curve), 1.0e1)
        ax_uvlf.set_xlabel(r"$M_{\rm UV}$")
        ax_uvlf.set_ylabel(r"$\phi(M_{\rm UV})\ [{\rm Mpc}^{-3}\ {\rm mag}^{-1}]$")
        ax_uvlf.set_title(rf"HMF-weighted UVLF contributions at $z={z_final:g}$")
        ax_uvlf.grid(True, which="both", alpha=0.22)
        ax_uvlf.legend(frameon=False, fontsize=10, ncol=2)

        ax_tail.plot(mh_values, tail_abs_arr, color="#1F3A5F", marker="o", linewidth=2.0)
        for mh, y, frac in zip(mh_values, tail_abs_arr, tail_frac_arr, strict=True):
            color = MASS_COLORS.get(float(mh), "#1F3A5F")
            ax_tail.scatter([mh], [y], s=60, color=color, zorder=3)
            if y > 0.0:
                ax_tail.annotate(
                    f"{frac:.2f}",
                    (mh, y),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color=color,
                )

        ax_tail.set_xscale("log")
        ax_tail.set_yscale("log")
        ax_tail.set_ylim(_nice_log_lower(tail_abs_arr, floor=1.0e-20), max(1.0, np.max(tail_abs_arr) * 1.3))
        ax_tail.set_xlabel(r"$M_h\ [M_\odot]$")
        ax_tail.set_ylabel(r"Contribution to $n(M_{\rm UV}<-19)$")
        ax_tail.set_title(rf"Bright-end contribution at $z={z_final:g}$")
        ax_tail.grid(True, which="both", alpha=0.22)

        png_path = outputs_dir / f"uvlf_fixed_mass_pdf_hmf_weighted_z{format_redshift_tag(z_final)}.png"
        pdf_path = outputs_dir / f"uvlf_fixed_mass_pdf_hmf_weighted_z{format_redshift_tag(z_final)}.pdf"
        fig.savefig(png_path, dpi=240, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=240, bbox_inches="tight")
        plt.close(fig)

        summary_lines.extend(
            [
                f"z={z_final:g}",
                f"  png_path={png_path.resolve()}",
                f"  pdf_path={pdf_path.resolve()}",
            ]
        )
        for mh, dlm, dndm_i, dndlogm_i, dens, p_tail, tail_abs_i, frac_i in zip(
            mh_values, dlogm, dndm, dndlogm, interval_density, [stats[(z_final, float(mh))]["p_lt_tail"] for mh in mh_values], tail_abs_arr, tail_frac_arr, strict=True
        ):
            stat_rows.append(
                "\t".join(
                    [
                        f"{z_final:.6f}",
                        f"{float(mh):.6e}",
                        f"{float(dlm):.6f}",
                        f"{float(dndm_i):.12e}",
                        f"{float(dndlogm_i):.12e}",
                        f"{float(dens):.12e}",
                        f"{float(p_tail):.12e}",
                        f"{float(tail_abs_i):.12e}",
                        f"{float(frac_i):.12e}",
                    ]
                )
            )
            summary_lines.append(
                f"  Mh={mh:.0e}: dn/dlogMh={float(dndlogm_i):.3e}, interval_density={float(dens):.3e}, tail_frac={float(frac_i):.3f}"
            )
        summary_lines.append("")

    summary_path = data_dir / "uvlf_fixed_mass_pdf_hmf_weighted.txt"
    stats_path = data_dir / "uvlf_fixed_mass_pdf_hmf_weighted_stats.tsv"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    stats_path.write_text("\n".join(stat_rows) + "\n", encoding="utf-8")

    print(summary_path.resolve())
    print(stats_path.resolve())


if __name__ == "__main__":
    main()
