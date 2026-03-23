#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sfr.calculator import EXTENDED_BURST_KAPPA
from uvlf import run_halo_uv_pipeline, uv_luminosity_to_muv
from uvlf.hmf_sampling import UVLFSamplingResult, sample_uvlf_from_hmf


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return np.full_like(quantiles, np.nan, dtype=float)

    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    if total <= 0.0:
        return np.full_like(quantiles, np.nan, dtype=float)
    cdf = (cumulative - 0.5 * weights) / total
    return np.interp(quantiles, cdf, values, left=values[0], right=values[-1])


def _weighted_histogram_pdf(
    values: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts, used_edges = np.histogram(values, bins=edges, weights=weights, density=False)
    width = np.diff(used_edges)
    total_weight = np.sum(weights[np.isfinite(values) & np.isfinite(weights)])
    pdf = np.divide(counts, total_weight * width, out=np.zeros_like(counts, dtype=float), where=width > 0.0)
    centers = 0.5 * (used_edges[:-1] + used_edges[1:])
    return centers, pdf


def _binned_weighted_quantiles(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    q16 = np.full(edges.size - 1, np.nan, dtype=float)
    q50 = np.full(edges.size - 1, np.nan, dtype=float)
    q84 = np.full(edges.size - 1, np.nan, dtype=float)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if index == edges.size - 2:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)
        if np.count_nonzero(mask) < 8:
            continue
        quantiles = _weighted_quantile(y[mask], weights[mask], np.array([0.16, 0.5, 0.84], dtype=float))
        q16[index], q50[index], q84[index] = quantiles
    return centers, q16, q50, q84


def _reshape_tracks(values: np.ndarray, n_tracks: int, n_grid: int) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(n_tracks, n_grid)


def _sample_delay_comparison(
    z_obs: float,
    n_mass: int,
    n_tracks: int,
    n_grid: int,
    workers: int,
    seed: int,
    output_stem: str,
) -> tuple[UVLFSamplingResult, UVLFSamplingResult]:
    bins = np.arange(-26.0, -9.5, 0.5, dtype=float)
    prefix = Path(output_stem)
    z_tag = str(z_obs).replace(".", "p")
    no_delay = sample_uvlf_from_hmf(
        z_obs=z_obs,
        N_mass=n_mass,
        n_tracks=n_tracks,
        random_seed=seed,
        quantity="Muv",
        bins=bins,
        n_grid=n_grid,
        enable_time_delay=False,
        pipeline_workers=workers,
        progress_path=prefix.parent / f"{prefix.name}_z{z_tag}_no_delay_progress.txt",
        print_progress=True,
    )
    delay = sample_uvlf_from_hmf(
        z_obs=z_obs,
        N_mass=n_mass,
        n_tracks=n_tracks,
        random_seed=seed,
        quantity="Muv",
        bins=bins,
        n_grid=n_grid,
        enable_time_delay=True,
        pipeline_workers=workers,
        progress_path=prefix.parent / f"{prefix.name}_z{z_tag}_delay_progress.txt",
        print_progress=True,
    )
    return no_delay, delay


def _fixed_mass_delay_comparison(
    z_obs: float,
    mh_final: float,
    n_tracks: int,
    n_grid: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    no_delay = run_halo_uv_pipeline(
        n_tracks=n_tracks,
        z_final=z_obs,
        Mh_final=mh_final,
        n_grid=n_grid,
        random_seed=seed,
        enable_time_delay=False,
        workers=1,
    )
    delay = run_halo_uv_pipeline(
        n_tracks=n_tracks,
        z_final=z_obs,
        Mh_final=mh_final,
        n_grid=n_grid,
        random_seed=seed,
        enable_time_delay=True,
        workers=1,
    )

    steps = int(no_delay.redshift_grid.size)
    t_grid = _reshape_tracks(no_delay.sfr_tracks["t_gyr"], n_tracks, steps)
    mdot_grid = _reshape_tracks(no_delay.sfr_tracks["dMh_dt"], n_tracks, steps)
    active_grid = np.asarray(no_delay.sfr_tracks["active_flag"], dtype=bool).reshape(n_tracks, steps)
    td_grid = _reshape_tracks(no_delay.sfr_tracks["td_burst"], n_tracks, steps)
    sfr_no = _reshape_tracks(no_delay.sfr_tracks["SFR"], n_tracks, steps)
    sfr_delay = _reshape_tracks(delay.sfr_tracks["SFR"], n_tracks, steps)

    final_index = np.max(
        np.where(active_grid, np.arange(steps, dtype=int)[None, :], -1),
        axis=1,
    )
    halo_index = np.arange(n_tracks, dtype=int)
    valid_final = final_index >= 0
    sfr_ratio = np.full(n_tracks, np.nan, dtype=float)
    td_final = np.full(n_tracks, np.nan, dtype=float)
    final_mdot = np.full(n_tracks, np.nan, dtype=float)
    for halo_id in np.flatnonzero(valid_final):
        idx = int(final_index[halo_id])
        no_value = float(sfr_no[halo_id, idx])
        delay_value = float(sfr_delay[halo_id, idx])
        if np.isfinite(no_value) and no_value > 0.0 and np.isfinite(delay_value):
            sfr_ratio[halo_id] = delay_value / no_value
        td_final[halo_id] = td_grid[halo_id, idx]
        final_mdot[halo_id] = mdot_grid[halo_id, idx]

    final_mdot_safe = np.where(
        np.isfinite(final_mdot[:, None]) & (final_mdot[:, None] > 0.0),
        final_mdot[:, None],
        np.nan,
    )
    mdot_ratio_grid = np.divide(mdot_grid, final_mdot_safe, out=np.full_like(mdot_grid, np.nan), where=np.isfinite(final_mdot_safe))
    lookback_myr = (t_grid[:, -1][:, None] - t_grid) * 1.0e3

    return (
        {
            "t_grid_gyr": t_grid,
            "lookback_myr": lookback_myr,
            "mdot_ratio_grid": mdot_ratio_grid,
            "active_grid": active_grid,
            "td_final_gyr": td_final,
            "sfr_ratio_final": sfr_ratio,
            "uv_muv_no": np.asarray(uv_luminosity_to_muv(no_delay.uv_luminosities), dtype=float),
            "uv_muv_delay": np.asarray(uv_luminosity_to_muv(delay.uv_luminosities), dtype=float),
        },
        {
            "timing_no_delay": np.array([float(no_delay.metadata["timing_seconds"]["total_without_plotting"])]),
            "timing_delay": np.array([float(delay.metadata["timing_seconds"]["total_without_plotting"])]),
        },
    )


def _choose_delta_muv_edges(delta_sets: list[np.ndarray]) -> np.ndarray:
    finite = [values[np.isfinite(values)] for values in delta_sets if np.any(np.isfinite(values))]
    if not finite:
        return np.linspace(0.0, 2.5, 61, dtype=float)
    all_values = np.concatenate(finite)
    upper = float(np.nanpercentile(all_values, 99.5))
    xmax = max(0.4, min(4.0, upper * 1.1))
    return np.linspace(0.0, xmax, 61, dtype=float)


def _try_use_apj_style() -> None:
    try:
        plt.style.use("apj")
    except OSError:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build diagnostics that explain the UVLF impact of the delay kernel.")
    parser.add_argument("--z-values", nargs="+", type=float, default=[6.0, 12.5])
    parser.add_argument("--n-mass", type=int, default=3000)
    parser.add_argument("--n-tracks", type=int, default=1000)
    parser.add_argument("--n-grid", type=int, default=240)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-mh", type=float, default=1.0e11)
    parser.add_argument("--fixed-tracks", type=int, default=1000)
    parser.add_argument("--fixed-seed", type=int, default=31415)
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outputs/uvlf_delay_understand_diagnostics_20260322",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data_save/uvlf_delay_understand_diagnostics_20260322.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path).expanduser().resolve()
    data_path.parent.mkdir(parents=True, exist_ok=True)

    _try_use_apj_style()

    z_values = [float(z) for z in args.z_values]
    if len(z_values) != 2:
        raise ValueError("this diagnostic script expects exactly two z values so the panels stay readable")

    sample_results: dict[float, tuple[UVLFSamplingResult, UVLFSamplingResult]] = {}
    fixed_results: dict[float, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}

    for z_obs in z_values:
        print(f"running HMF-sampled diagnostic for z={z_obs:g}", flush=True)
        sample_results[z_obs] = _sample_delay_comparison(
            z_obs=z_obs,
            n_mass=int(args.n_mass),
            n_tracks=int(args.n_tracks),
            n_grid=int(args.n_grid),
            workers=int(args.workers),
            seed=int(args.seed),
            output_stem=str(output_prefix),
        )
        print(f"running fixed-mass diagnostic for z={z_obs:g}", flush=True)
        fixed_results[z_obs] = _fixed_mass_delay_comparison(
            z_obs=z_obs,
            mh_final=float(args.fixed_mh),
            n_tracks=int(args.fixed_tracks),
            n_grid=int(args.n_grid),
            seed=int(args.fixed_seed),
        )

    delta_muv_by_z: dict[float, np.ndarray] = {}
    weight_by_z: dict[float, np.ndarray] = {}
    lum_ratio_by_z: dict[float, np.ndarray] = {}
    logmh_by_z: dict[float, np.ndarray] = {}
    summary_lines: list[str] = [
        "Delay-kernel diagnostics",
        f"kappa: {EXTENDED_BURST_KAPPA}",
        f"hmf_n_mass: {args.n_mass}",
        f"hmf_n_tracks: {args.n_tracks}",
        f"hmf_n_grid: {args.n_grid}",
        f"hmf_workers: {args.workers}",
        f"fixed_mh: {args.fixed_mh:.6e}",
        f"fixed_tracks: {args.fixed_tracks}",
        "",
    ]

    for z_obs in z_values:
        no_delay, delay = sample_results[z_obs]
        muv_no = np.asarray(no_delay.samples["Muv"], dtype=float)
        muv_delay = np.asarray(delay.samples["Muv"], dtype=float)
        lum_no = np.asarray(no_delay.samples["luminosity"], dtype=float)
        lum_delay = np.asarray(delay.samples["luminosity"], dtype=float)
        weights = np.asarray(no_delay.samples["sample_weight"], dtype=float)
        logmh = np.asarray(no_delay.samples["logMh"], dtype=float)

        delta_muv = muv_delay - muv_no
        lum_ratio = np.divide(lum_delay, lum_no, out=np.full_like(lum_no, np.nan), where=lum_no > 0.0)

        delta_muv_by_z[z_obs] = delta_muv
        weight_by_z[z_obs] = weights
        lum_ratio_by_z[z_obs] = lum_ratio
        logmh_by_z[z_obs] = logmh

        q16, q50, q84 = _weighted_quantile(delta_muv, weights, np.array([0.16, 0.5, 0.84], dtype=float))
        r16, r50, r84 = _weighted_quantile(lum_ratio, weights, np.array([0.16, 0.5, 0.84], dtype=float))
        summary_lines.extend(
            [
                f"z={z_obs:g}",
                f"  weighted_delta_muv_p16={q16:.6f}",
                f"  weighted_delta_muv_p50={q50:.6f}",
                f"  weighted_delta_muv_p84={q84:.6f}",
                f"  weighted_lum_ratio_p16={r16:.6f}",
                f"  weighted_lum_ratio_p50={r50:.6f}",
                f"  weighted_lum_ratio_p84={r84:.6f}",
                "",
            ]
        )

    delta_edges = _choose_delta_muv_edges([delta_muv_by_z[z] for z in z_values])
    logmh_edges = np.linspace(9.0, 13.0, 17, dtype=float)

    colors = {z_values[0]: "#1f4e79", z_values[1]: "#c44e52"}
    labels = {z: rf"$z={z:g}$" for z in z_values}
    compact_data: dict[str, np.ndarray] = {}

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)
    ax_hist = axes[0, 0]
    ax_mass = axes[0, 1]
    ax_sfr = axes[1, 0]
    ax_mdot = axes[1, 1]

    for z_obs in z_values:
        centers, pdf = _weighted_histogram_pdf(delta_muv_by_z[z_obs], weight_by_z[z_obs], delta_edges)
        ax_hist.plot(centers, pdf, color=colors[z_obs], lw=2.4, label=labels[z_obs])
        q16, q50, q84 = _weighted_quantile(
            delta_muv_by_z[z_obs],
            weight_by_z[z_obs],
            np.array([0.16, 0.5, 0.84], dtype=float),
        )
        ax_hist.axvline(q50, color=colors[z_obs], ls="--", lw=1.4, alpha=0.9)
        ax_hist.axvspan(q16, q84, color=colors[z_obs], alpha=0.12)
        z_key = f"z{str(z_obs).replace('.', 'p')}"
        compact_data[f"{z_key}_delta_hist_centers"] = centers
        compact_data[f"{z_key}_delta_hist_pdf"] = pdf
        compact_data[f"{z_key}_delta_q16_q50_q84"] = np.array([q16, q50, q84], dtype=float)

        centers_mass, q16_mass, q50_mass, q84_mass = _binned_weighted_quantiles(
            logmh_by_z[z_obs],
            delta_muv_by_z[z_obs],
            weight_by_z[z_obs],
            logmh_edges,
        )
        ax_mass.plot(centers_mass, q50_mass, color=colors[z_obs], lw=2.4, label=labels[z_obs])
        ax_mass.fill_between(centers_mass, q16_mass, q84_mass, color=colors[z_obs], alpha=0.16)
        compact_data[f"{z_key}_mass_centers"] = centers_mass
        compact_data[f"{z_key}_mass_q16"] = q16_mass
        compact_data[f"{z_key}_mass_q50"] = q50_mass
        compact_data[f"{z_key}_mass_q84"] = q84_mass

        fixed_payload, fixed_meta = fixed_results[z_obs]
        sfr_ratio = np.asarray(fixed_payload["sfr_ratio_final"], dtype=float)
        sfr_valid = np.isfinite(sfr_ratio) & (sfr_ratio > 0.0)
        if np.any(sfr_valid):
            sfr_edges = np.linspace(0.0, 1.2, 49, dtype=float)
            sfr_centers, sfr_pdf = _weighted_histogram_pdf(
                sfr_ratio[sfr_valid],
                np.ones(np.count_nonzero(sfr_valid), dtype=float),
                sfr_edges,
            )
            ax_sfr.plot(sfr_centers, sfr_pdf, color=colors[z_obs], lw=2.4, label=labels[z_obs])
            sfr_q16, sfr_q50, sfr_q84 = np.nanpercentile(sfr_ratio[sfr_valid], [16.0, 50.0, 84.0])
            ax_sfr.axvline(sfr_q50, color=colors[z_obs], ls="--", lw=1.4, alpha=0.9)
            td_myr = 1.0e3 * np.nanmedian(fixed_payload["td_final_gyr"])
            compact_data[f"{z_key}_fixed_sfr_centers"] = sfr_centers
            compact_data[f"{z_key}_fixed_sfr_pdf"] = sfr_pdf
            compact_data[f"{z_key}_fixed_sfr_q16_q50_q84"] = np.array([sfr_q16, sfr_q50, sfr_q84], dtype=float)
            summary_lines.extend(
                [
                    f"fixed_mass z={z_obs:g}",
                    f"  fixed_mass_td_median_myr={td_myr:.6f}",
                    f"  fixed_mass_kappa_td_median_myr={EXTENDED_BURST_KAPPA * td_myr:.6f}",
                    f"  fixed_mass_mean_delay_median_myr={2.0 * EXTENDED_BURST_KAPPA * td_myr:.6f}",
                    f"  fixed_mass_sfr_ratio_p16={sfr_q16:.6f}",
                    f"  fixed_mass_sfr_ratio_p50={sfr_q50:.6f}",
                    f"  fixed_mass_sfr_ratio_p84={sfr_q84:.6f}",
                    f"  fixed_mass_uv_delta_muv_p50={np.nanmedian(fixed_payload['uv_muv_delay'] - fixed_payload['uv_muv_no']):.6f}",
                    f"  fixed_mass_timing_no_delay={float(fixed_meta['timing_no_delay'][0]):.6f}",
                    f"  fixed_mass_timing_delay={float(fixed_meta['timing_delay'][0]):.6f}",
                    "",
                ]
            )

        lookback = np.asarray(fixed_payload["lookback_myr"], dtype=float)
        mdot_ratio_grid = np.asarray(fixed_payload["mdot_ratio_grid"], dtype=float)
        valid_mdot = np.isfinite(mdot_ratio_grid) & (mdot_ratio_grid > 0.0)
        lookback_axis = np.linspace(0.0, 120.0, 121, dtype=float)
        mdot_median = np.full_like(lookback_axis, np.nan, dtype=float)
        for i, delta_t in enumerate(lookback_axis):
            nearest = np.argmin(np.abs(lookback - delta_t), axis=1)
            sampled = mdot_ratio_grid[np.arange(mdot_ratio_grid.shape[0]), nearest]
            sampled_valid = np.isfinite(sampled) & (sampled > 0.0)
            if np.any(sampled_valid):
                mdot_median[i] = float(np.nanmedian(sampled[sampled_valid]))
        ax_mdot.plot(lookback_axis, mdot_median, color=colors[z_obs], lw=2.4, label=labels[z_obs])
        td_myr = 1.0e3 * np.nanmedian(fixed_payload["td_final_gyr"])
        compact_data[f"{z_key}_fixed_mdot_lookback_myr"] = lookback_axis
        compact_data[f"{z_key}_fixed_mdot_median_ratio"] = mdot_median
        compact_data[f"{z_key}_fixed_td_kappatd_meandelay_myr"] = np.array(
            [td_myr, EXTENDED_BURST_KAPPA * td_myr, 2.0 * EXTENDED_BURST_KAPPA * td_myr],
            dtype=float,
        )
        ax_mdot.axvline(
            EXTENDED_BURST_KAPPA * td_myr,
            color=colors[z_obs],
            ls=":",
            lw=1.5,
            alpha=0.85,
        )
        ax_mdot.axvline(
            2.0 * EXTENDED_BURST_KAPPA * td_myr,
            color=colors[z_obs],
            ls="--",
            lw=1.2,
            alpha=0.6,
        )

    ax_hist.set_xlabel(r"$\Delta M_{\rm UV} = M_{\rm UV}^{\rm delay} - M_{\rm UV}^{\rm no\,delay}$")
    ax_hist.set_ylabel("Weighted PDF")
    ax_hist.set_title("How much each halo is dimmed")
    ax_hist.grid(alpha=0.25)
    ax_hist.legend(frameon=False)

    ax_mass.set_xlabel(r"$\log_{10}(M_h/M_\odot)$")
    ax_mass.set_ylabel(r"Weighted median $\Delta M_{\rm UV}$")
    ax_mass.set_title("Mass dependence of the dimming")
    ax_mass.grid(alpha=0.25)
    ax_mass.legend(frameon=False)

    ax_sfr.set_xlabel(r"${\rm SFR}_{\rm delay}/{\rm SFR}_{\rm no\,delay}$")
    ax_sfr.set_ylabel("Track PDF")
    ax_sfr.set_xlim(0.0, 1.15)
    ax_sfr.set_title(rf"Final-SFR suppression at fixed $M_h={args.fixed_mh:.0e}\,M_\odot$")
    ax_sfr.grid(alpha=0.25)
    ax_sfr.legend(frameon=False)

    ax_mdot.set_xlabel("Lookback from observation [Myr]")
    ax_mdot.set_ylabel(r"Median $\dot M_h(t)/\dot M_h(t_{\rm obs})$")
    ax_mdot.set_xlim(0.0, 120.0)
    ax_mdot.set_ylim(0.0, 1.05)
    ax_mdot.set_title(rf"Recent accretion history at fixed $M_h={args.fixed_mh:.0e}\,M_\odot$")
    ax_mdot.grid(alpha=0.25)
    ax_mdot.legend(frameon=False)

    summary_lines.extend(
        [
            "panel_note_1: top-left shows per-halo UV dimming, not UVLF ratio at fixed bin",
            "panel_note_2: top-right shows whether the dimming is concentrated in a narrow mass range",
            "panel_note_3: bottom-left isolates the direct SFR suppression before the UVLF bright-end slope amplifies it",
            "panel_note_4: bottom-right compares recent mdot growth against the delay timescale; dotted lines are kappa*td, dashed lines are 2*kappa*td",
        ]
    )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    txt_path = output_prefix.with_suffix(".txt")
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)

    txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    np.savez_compressed(
        data_path,
        z_values=np.asarray(z_values, dtype=float),
        delta_edges=delta_edges,
        logmh_edges=logmh_edges,
        fixed_mh=np.array([float(args.fixed_mh)], dtype=float),
        **compact_data,
    )

    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")
    print(f"saved_txt={txt_path}")
    print(f"saved_npz={data_path}")


if __name__ == "__main__":
    main()
