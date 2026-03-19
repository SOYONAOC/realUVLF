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
from uvlf.pipeline import DEFAULT_SSP_FILE


FONT_PATH = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path("outputs")
PNG_PATH = OUTPUT_DIR / "ssp_convolution_explanation_ssplong.png"
TXT_PATH = OUTPUT_DIR / "ssp_convolution_explanation_ssplong.txt"

Z_VALUES = [6.0, 12.5]
Z_START_MAX = 50.0
N_GRID = 240
N_TRACKS = 1000
MH_FINAL = 1.0e11
LOOKBACK_MAX_MYR = 300.0
KA_UV_SSP_LONG = 6.1e-29


def reserve_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def cumulative_recent_uv_fraction(lookback_myr: np.ndarray, contribution_density: np.ndarray) -> np.ndarray:
    order = np.argsort(lookback_myr)
    lookback_sorted = lookback_myr[order]
    contrib_sorted = contribution_density[order]
    cumulative = np.zeros_like(contrib_sorted, dtype=float)
    if contrib_sorted.size > 1:
        cumulative[1:] = np.cumsum(
            0.5 * (contrib_sorted[1:] + contrib_sorted[:-1]) * np.diff(lookback_sorted)
        )
    total = cumulative[-1] if cumulative.size else 0.0
    if total > 0.0:
        cumulative /= total
    result = np.empty_like(cumulative)
    result[order] = cumulative
    return result


def build_case(z_final: float) -> dict[str, np.ndarray | float]:
    cosmology = Cosmology()
    redshift_grid = np.linspace(Z_START_MAX, z_final, N_GRID, dtype=float)
    histories = generate_halo_histories(
        n_tracks=N_TRACKS,
        z_final=z_final,
        Mh_final=MH_FINAL,
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

    final_sfr_values: list[float] = []
    halo_order: list[int] = []
    for halo_id in np.unique(halo_ids):
        mask = (halo_ids == halo_id) & active
        if not np.any(mask):
            continue
        halo_order.append(int(halo_id))
        final_sfr_values.append(float(sfr[mask][-1]))

    final_sfr_array = np.asarray(final_sfr_values, dtype=float)
    target_sfr = float(np.median(final_sfr_array))
    chosen_index = int(np.argmin(np.abs(final_sfr_array - target_sfr)))
    chosen_halo_id = halo_order[chosen_index]

    chosen_mask = (halo_ids == chosen_halo_id) & active
    t_used = t_gyr[chosen_mask]
    mh_used = mh[chosen_mask]
    sfr_used = sfr[chosen_mask]
    t_obs = float(t_used[-1])

    details = compute_halo_uv_luminosity(
        t_obs=t_obs,
        t_history=t_used,
        mh_history=mh_used,
        sfr_history=sfr_used,
        ssp_age_grid=ssp_age_grid_gyr,
        ssp_luv_grid=luv_per_msun,
        M_min=0.0,
        t_z50=float(t_used[0]),
        time_unit_in_years=1.0e9,
        return_details=True,
    )

    lookback_myr = np.asarray(details["age_used"], dtype=float) * 1.0e3
    kernel = np.asarray(details["kernel_used"], dtype=float)
    sfr_interp = np.asarray(np.interp(np.asarray(details["t_used"], dtype=float), t_used, sfr_used), dtype=float)
    contribution_density = np.asarray(details["integrand_used"], dtype=float)

    within_window = lookback_myr <= LOOKBACK_MAX_MYR
    lookback_myr = lookback_myr[within_window]
    kernel = kernel[within_window]
    sfr_interp = sfr_interp[within_window]
    contribution_density = contribution_density[within_window]

    cumulative_fraction = cumulative_recent_uv_fraction(lookback_myr, contribution_density)

    current_sfr = float(sfr_interp[np.argmin(lookback_myr)])
    current_kernel = float(kernel[np.argmin(lookback_myr)])
    current_luv_instant = current_sfr / KA_UV_SSP_LONG
    total_luv_ssp = float(details["L_uv_halo"])
    kuv_eff = current_sfr / total_luv_ssp

    return {
        "z_final": z_final,
        "halo_id": chosen_halo_id,
        "target_sfr": target_sfr,
        "current_sfr": current_sfr,
        "instant_luv_ssplong": current_luv_instant,
        "ssp_luv": total_luv_ssp,
        "ssp_to_instant": total_luv_ssp / current_luv_instant,
        "kuv_eff": kuv_eff,
        "lookback_myr": lookback_myr,
        "sfr_norm": sfr_interp / current_sfr,
        "kernel_norm": kernel / current_kernel,
        "contribution_norm": contribution_density / np.max(contribution_density),
        "cumulative_fraction": cumulative_fraction,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = reserve_output_path(PNG_PATH)
    txt_path = reserve_output_path(TXT_PATH)

    t0 = time.perf_counter()
    cases = [build_case(z_value) for z_value in Z_VALUES]

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(13.5, 11.5),
        sharex="col",
        constrained_layout=True,
    )
    colors = {6.0: "#1f77b4", 12.5: "#d62728"}

    row_titles = [
        "归一化 SFR 历史  SFR(t) / SFR(t_obs)",
        "归一化 SSP 核函数  Psi_UV(age) / Psi_UV(0)",
        "归一化积分贡献  [SFR × Psi_UV] / max",
        "累计 UV 占比  L_UV(< Δt) / L_UV(full)",
    ]

    for col_index, case in enumerate(cases):
        z_final = float(case["z_final"])
        color = colors[z_final]
        lookback_myr = np.asarray(case["lookback_myr"], dtype=float)

        axes[0, col_index].plot(lookback_myr, np.asarray(case["sfr_norm"], dtype=float), color=color, lw=2.4)
        axes[1, col_index].plot(lookback_myr, np.asarray(case["kernel_norm"], dtype=float), color=color, lw=2.4)
        axes[2, col_index].fill_between(
            lookback_myr,
            0.0,
            np.asarray(case["contribution_norm"], dtype=float),
            color=color,
            alpha=0.22,
        )
        axes[2, col_index].plot(
            lookback_myr,
            np.asarray(case["contribution_norm"], dtype=float),
            color=color,
            lw=2.4,
        )
        axes[3, col_index].plot(
            lookback_myr,
            np.asarray(case["cumulative_fraction"], dtype=float),
            color=color,
            lw=2.4,
        )

        axes[0, col_index].set_title(
            f"z = {z_final:g}\n"
            f"代表性轨道 halo_id={int(case['halo_id'])},  SSP / inst,long = {float(case['ssp_to_instant']):.2f}x",
            fontsize=13,
        )
        text = (
            rf"$K_{{\rm UV,SSP,long}}\approx{KA_UV_SSP_LONG:.2e}$" "\n"
            rf"$K_{{\rm UV,eff}}={float(case['kuv_eff']):.2e}$" "\n"
            rf"$L_{{\rm SSP}}/L_{{\rm inst,long}}={float(case['ssp_to_instant']):.2f}$"
        )
        axes[0, col_index].text(
            0.03,
            0.08,
            text,
            transform=axes[0, col_index].transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9),
        )

        for row_index in range(4):
            axes[row_index, col_index].grid(True, which="both", alpha=0.25)
            if row_index < 3:
                axes[row_index, col_index].set_yscale("log")
            axes[row_index, col_index].set_xlim(0.0, LOOKBACK_MAX_MYR)

    for row_index, label in enumerate(row_titles):
        axes[row_index, 0].set_ylabel(label)

    axes[3, 0].set_xlabel("距观测时刻的回望时间 Δt [Myr]")
    axes[3, 1].set_xlabel("距观测时刻的回望时间 Δt [Myr]")

    fig.suptitle(
        "SSP 卷积图解：相对于长期极限 $K_{\\rm UV,SSP,long}$ 的瞬时 UV 基准\n"
        r"$L_{\rm UV}(t_{\rm obs})=\int {\rm SFR}(t)\,\Psi_{\rm UV}(t_{\rm obs}-t)\,dt,\ "
        r"K_{\rm UV,eff}={\rm SFR}(t_{\rm obs})/L_{\rm UV}(t_{\rm obs})$"
        "\n"
        rf"固定代表性终质量 $M_{{h,final}}={MH_FINAL:.0e}\,M_\odot$",
        fontsize=16,
    )
    fig.savefig(png_path, dpi=220)

    elapsed = time.perf_counter() - t0
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write(f"MH_FINAL: {MH_FINAL}\n")
        handle.write(f"Z_VALUES: {Z_VALUES}\n")
        handle.write(f"N_TRACKS: {N_TRACKS}\n")
        handle.write(f"LOOKBACK_MAX_MYR: {LOOKBACK_MAX_MYR}\n")
        handle.write(f"elapsed_seconds: {elapsed:.6f}\n")
        handle.write(f"png_path: {png_path.resolve()}\n")
        for case in cases:
            handle.write(
                f"z={float(case['z_final']):g}, "
                f"halo_id={int(case['halo_id'])}, "
                f"target_sfr={float(case['target_sfr']):.6f}, "
                f"current_sfr={float(case['current_sfr']):.6f}, "
                f"instant_luv_ssplong={float(case['instant_luv_ssplong']):.6e}, "
                f"ssp_luv={float(case['ssp_luv']):.6e}, "
                f"ssp_to_instant={float(case['ssp_to_instant']):.6f}, "
                f"kuv_eff={float(case['kuv_eff']):.6e}\n"
            )

    print(f"saved_png={png_path.resolve()}")
    print(f"saved_txt={txt_path.resolve()}")


if __name__ == "__main__":
    main()
