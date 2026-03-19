#!/usr/bin/env python3
from __future__ import annotations

import time
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

OUTPUT_DIR = Path("outputs")
TSV_PATH = OUTPUT_DIR / "ssp_vs_instant_ssplong_same_sfh.tsv"
TXT_PATH = OUTPUT_DIR / "ssp_vs_instant_ssplong_same_sfh.txt"
PNG_PATH = OUTPUT_DIR / "ssp_vs_instant_ssplong_same_sfh.png"

Z_VALUES = [6.0, 8.0, 10.0, 12.5]
MASS_VALUES = [2.0e10, 3.0e10, 5.0e10, 1.0e11, 2.0e11]
N_TRACKS = 1000
Z_START_MAX = 50.0
N_GRID = 240
KA_UV = 6.1e-29


def reserve_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def summarize_same_sfh(z_final: float, mh_final: float) -> dict[str, float]:
    redshift_grid = np.linspace(Z_START_MAX, z_final, N_GRID, dtype=float)
    cosmology = Cosmology()
    histories = generate_halo_histories(
        n_tracks=N_TRACKS,
        z_final=z_final,
        Mh_final=mh_final,
        z_start_max=Z_START_MAX,
        cosmology=cosmology,
        random_seed=42,
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
        instant_luv = float(sfr_used[-1] / KA_UV)

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
        "mean_delta_mag": float(np.mean(delta_mag)),
        "median_delta_mag": float(np.median(delta_mag)),
        "mean_muv_ssp": float(np.mean(np.asarray(uv_luminosity_to_muv(ssp_luv_array), dtype=float))),
        "mean_muv_instant": float(np.mean(np.asarray(uv_luminosity_to_muv(instant_luv_array), dtype=float))),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tsv_path = TSV_PATH
    txt_path = TXT_PATH
    png_path = PNG_PATH

    t0 = time.perf_counter()
    rows: list[dict[str, float]] = []
    for z_value in Z_VALUES:
        for mh_value in MASS_VALUES:
            rows.append(summarize_same_sfh(z_value, mh_value))

    header = [
        "z",
        "Mh_final",
        "n_valid_tracks",
        "mean_luv_ratio",
        "median_luv_ratio",
        "mean_delta_mag",
        "median_delta_mag",
        "mean_muv_ssp",
        "mean_muv_instant",
    ]
    with tsv_path.open("w", encoding="utf-8") as handle:
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
                        f"{row['mean_delta_mag']:.6f}",
                        f"{row['median_delta_mag']:.6f}",
                        f"{row['mean_muv_ssp']:.6f}",
                        f"{row['mean_muv_instant']:.6f}",
                    ]
                )
                + "\n"
            )

    fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    colors = {
        2.0e10: "#1f77b4",
        3.0e10: "#ff7f0e",
        5.0e10: "#2ca02c",
        1.0e11: "#d62728",
        2.0e11: "#9467bd",
    }
    for mh_value in MASS_VALUES:
        subset = [row for row in rows if np.isclose(row["Mh_final"], mh_value)]
        subset.sort(key=lambda item: item["z"])
        ax.plot(
            [row["z"] for row in subset],
            [row["mean_luv_ratio"] for row in subset],
            marker="o",
            lw=2,
            color=colors[mh_value],
            label=rf"$M_h={mh_value:.0e}\,M_\odot$",
        )

    ax.set_xlabel("红移 z")
    ax.set_ylabel("相对瞬时 UV 基准的亮度比")
    ax.set_title("SSP 相对长时稳态瞬时 UV 基准的偏离")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    fig.savefig(png_path, dpi=220)

    elapsed = time.perf_counter() - t0
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write(f"n_tracks: {N_TRACKS}\n")
        handle.write(f"z_values: {Z_VALUES}\n")
        handle.write(f"mass_values: {MASS_VALUES}\n")
        handle.write(f"ssp_file: {DEFAULT_SSP_FILE}\n")
        handle.write(f"ka_uv_ssp_long: {KA_UV:.6e}\n")
        handle.write(f"elapsed_seconds: {elapsed:.6f}\n")
        handle.write(f"tsv_path: {tsv_path.resolve()}\n")
        handle.write(f"png_path: {png_path.resolve()}\n")

    print(f"saved_tsv={tsv_path.resolve()}")
    print(f"saved_txt={txt_path.resolve()}")
    print(f"saved_png={png_path.resolve()}")


if __name__ == "__main__":
    main()
