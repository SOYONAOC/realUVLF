"""He II Case-B predictions conditional on existing random-q burst events.

Run from the project root with PYTHONPATH=. and the project Python.
Flux thresholds are mathematical cuts, not instrument sensitivity claims.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from auroralf.experiments.artifacts import (
    digest,
    read_completed_manifest,
    read_toml,
    record_experiment_sources,
    require,
    resolve_path,
)
from auroralf.experiments.heii import cluster_sum_and_se, evaluate_kernel, load_kernel
from auroralf.mah import Cosmology
from auroralf.ssp import interpolate_ssp_luminosity
from auroralf.uvlf import uv_luminosity_to_muv


def weighted_quantiles(values, weights):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    require(len(v) > 0 and w.sum() > 0, "empty weighted distribution")
    return np.interp([0.16, 0.5, 0.84], (np.cumsum(w) - 0.5 * w) / w.sum(), v).tolist()


def estimate(arrays):
    mean, se = cluster_sum_and_se(arrays)
    return {"value": np.asarray(mean).tolist(), "cluster_mc_se": np.asarray(se).tolist()}


def cumulative_rows(values, weights, thresholds, selected=None):
    out = []
    for i, row in enumerate(values):
        v = row if selected is None else row[selected[i]]
        # NaNs (left censored events) do not have assigned line predictions.
        v = np.sort(v[np.isfinite(v)])
        out.append((len(v) - np.searchsorted(v, thresholds, side="left")) * weights[i])
    return np.asarray(out)


def analyze_case(case, config, base, kernel, uv_path, source_hashes):
    cosmology = Cosmology()
    astro = FlatLambdaCDM(H0=cosmology.h0_km_s_mpc, Om0=cosmology.omega_m, Ob0=cosmology.omega_b)
    z = case["z"]
    distance = astro.luminosity_distance(z).to_value("cm")
    flux_conversion = 4 * np.pi * distance**2
    eps = config["reference_efficiency"]
    cuts = np.asarray(config["flux_cuts"])
    flux_grid = np.logspace(-21, -17, 49)
    line_edges = np.arange(38, 44.01, 0.25)
    age_edges = np.array([0, 1, 2, 3, 5, 10, 30, 100, 1000.0])
    stat = {}
    populations = {k: [] for k in ("flux", "age", "mass", "q", "popiii_uv_fraction")}
    population_weights = []
    seeds, run_records = set(), []

    def add(name, value):
        stat.setdefault(name, []).append(value)

    for run_value in case["runs"]:
        path = resolve_path(base, run_value)
        m = read_completed_manifest(path, required_products=("samples.npz", "source.tar.gz"))
        c = m["config"]
        require(
            c["z"] == z and c["seed"] not in seeds, "redshift mismatch or duplicate realization"
        )
        require(c["q_log10_mean"] == 0.5 and c["q_log10_sigma"] == 1.5, "different q model")
        require(c["lookback_myr"] == 100.0, "different UV lookback")
        require(m["input_sha256"][c["popiii_ssp"]] == digest(uv_path), "different executed UV SSP")
        # These two definitions set the burst mass and ages in the frozen run.
        for name in ("auroralf/constants.py", "auroralf/mah/models.py"):
            require(digest(name) == m["code_sha256"][name], f"executed cosmology differs: {name}")
        seeds.add(c["seed"])
        source_hashes[str(path / "manifest.json")] = digest(path / "manifest.json")
        run_records.append(
            {
                "path": str(path),
                "seed": c["seed"],
                "products": m["products"],
                "n_mass": c["n_mass"],
                "n_tracks": c["n_tracks"],
            }
        )
        with np.load(path / "samples.npz") as data:
            s = {
                k: data[k]
                for k in (
                    "status",
                    "age_myr",
                    "burst_halo_mass_msun",
                    "popii",
                    "popiii_per_efficiency",
                    "weight_per_track",
                    "logq",
                )
            }
        shape = (c["n_mass"], c["n_tracks"])
        w = s["weight_per_track"]
        require(
            w.shape == (shape[0],) and np.isfinite(w).all() and np.all(w > 0), "invalid weights"
        )
        require(all(s[k].shape == shape for k in s if k != "weight_per_track"), "sample shapes")
        require(np.isin(s["status"], [0, 1, 2]).all(), "unknown event status")
        resolved, censored = s["status"] == 1, s["status"] == 2
        age, mass = s["age_myr"], s["burst_halo_mass_msun"]
        require(np.isfinite(age[resolved]).all() and np.all(age[resolved] >= 0), "event ages")
        require(np.isfinite(mass[resolved]).all() and np.all(mass[resolved] > 0), "event masses")
        require(
            np.isnan(age[~resolved]).all() and np.isnan(mass[~resolved]).all(),
            "non-event coordinates",
        )
        for key in ("popii", "popiii_per_efficiency", "logq"):
            require(np.isfinite(s[key]).all(), f"invalid {key}")
        require(np.all(s["popii"] >= 0) and np.all(s["popiii_per_efficiency"] >= 0), "negative UV")
        unit_mass = cosmology.omega_b / cosmology.omega_m * mass[resolved]
        check_uv = np.zeros(shape)
        active = resolved & (age <= c["lookback_myr"])
        check_uv[active] = (
            cosmology.omega_b
            / cosmology.omega_m
            * mass[active]
            * interpolate_ssp_luminosity(age[active], kernel.age_myr, kernel.uv_per_msun)
        )
        require(
            np.allclose(check_uv, s["popiii_per_efficiency"], rtol=2e-12, atol=0),
            "burst-to-UV replay differs",
        )
        p3 = s["popiii_per_efficiency"]
        total_uv = s["popii"] + eps * p3
        selected = uv_luminosity_to_muv(total_uv) <= config["uv_cut"]
        baseline = uv_luminosity_to_muv(s["popii"]) <= config["uv_cut"]
        added = selected & ~baseline
        line = np.zeros(shape)
        line[censored] = np.nan
        line[resolved] = eps * unit_mass * evaluate_kernel(age[resolved], kernel)
        flux = line / flux_conversion
        add("uv_bright_density", selected.sum(axis=1) * w)
        add("uv_baseline_density", baseline.sum(axis=1) * w)
        add("uv_added_density", added.sum(axis=1) * w)
        add("uv_bright_left_censored_density", (selected & censored).sum(axis=1) * w)
        add("uv_added_age_below_2_density", (added & (age < 2)).sum(axis=1) * w)
        add("uv_added_age_below_3_density", (added & (age < 3)).sum(axis=1) * w)
        add(
            "uv_bright_age_below_first_ssp_density",
            (selected & (age < kernel.age_myr[0])).sum(axis=1) * w,
        )
        add(
            "uv_bright_age_bins",
            np.array([np.histogram(a[m], bins=age_edges)[0] for a, m in zip(age, selected)])
            * w[:, None],
        )
        add("heii_flux_cuts_all_density", cumulative_rows(flux, w, cuts))
        add("heii_flux_cuts_uv_bright_density", cumulative_rows(flux, w, cuts, selected))
        add("heii_flux_cuts_uv_added_density", cumulative_rows(flux, w, cuts, added))
        add("heii_flux_cumulative", cumulative_rows(flux, w, flux_grid))
        add("heii_flux_cumulative_uv_bright", cumulative_rows(flux, w, flux_grid, selected))
        add("line_luminosity_density", np.nansum(line, axis=1) * w)
        for efficiency in config["efficiencies"]:
            scaled = line * efficiency / eps
            rows = [
                np.histogram(np.log10(row[np.isfinite(row) & (row > 0)]), bins=line_edges)[0]
                for row in scaled
            ]
            add(f"line_lf_eps{efficiency:g}", np.asarray(rows) * w[:, None] / np.diff(line_edges))
        for method in ("linear_age", "log_log_age"):
            alternative = np.zeros(shape)
            alternative[censored] = np.nan
            alternative[resolved] = (
                eps * unit_mass * evaluate_kernel(age[resolved], kernel, method) / flux_conversion
            )
            add(
                f"{method}_flux_cuts_uv_bright_density",
                cumulative_rows(alternative, w, cuts, selected),
            )
        # Quantiles refer to all UV-bright objects with a known event state;
        # never silently assign a luminosity to a left-censored burst.
        known = selected & ~censored
        popiii_bright = known & resolved
        weights = np.broadcast_to(w[:, None], shape)
        population_weights.append(weights[popiii_bright] / len(case["runs"]))
        for key, arr in (
            ("flux", flux),
            ("age", age),
            ("mass", eps * cosmology.omega_b / cosmology.omega_m * mass),
            ("q", 10 ** s["logq"]),
            (
                "popiii_uv_fraction",
                np.divide(eps * p3, total_uv, out=np.zeros(shape), where=total_uv > 0),
            ),
        ):
            populations[key].append(arr[popiii_bright])
        print(f"Verified and analyzed {c['run_id']} z={z}", flush=True)

    result = {name: estimate(values) for name, values in stat.items()}
    # Correlated numerator/denominator: delta-method SE from the same clusters.
    denom = result["uv_bright_density"]["value"]
    for name in ("heii", "linear_age", "log_log_age"):
        source = f"{name}_flux_cuts_uv_bright_density"
        ratio = np.asarray(result[source]["value"]) / denom
        residuals = [
            numerator - denominator[:, None] * ratio
            for numerator, denominator in zip(stat[source], stat["uv_bright_density"])
        ]
        _, error = cluster_sum_and_se(residuals)
        result[f"{name}_uv_bright_fraction"] = {
            "value": ratio.tolist(),
            "cluster_mc_se": (error / denom).tolist(),
        }
    popweights = np.concatenate(population_weights)
    result["resolved_uv_bright_quantiles_16_50_84"] = {
        key: weighted_quantiles(np.concatenate(values), popweights)
        for key, values in populations.items()
    }
    result.update(
        z=z,
        runs=run_records,
        distance_cm=distance,
        observed_wavelength_um=1640.42 * (1 + z) / 1e4,
        flux_cuts=cuts.tolist(),
        flux_grid=flux_grid.tolist(),
        line_edges=line_edges.tolist(),
        age_edges=age_edges.tolist(),
        cosmology=asdict(cosmology),
    )
    return result


def plot_results(results, kernel, figures):
    plt.style.use("apj")
    figures.mkdir(parents=True, exist_ok=True)
    age = kernel.age_myr
    keep = age <= 5
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    ax[0].semilogy(
        age[keep], kernel.line_per_msun[keep] / kernel.line_per_msun.max(), "o-", label="He II 1640"
    )
    ax[0].semilogy(
        age[keep],
        kernel.uv_per_msun[keep] / kernel.uv_per_msun.max(),
        "s-",
        label="UV 1500 (total)",
    )
    ax[0].set(
        xlabel="Burst age [Myr]", ylabel="Luminosity / own peak", ylim=(1e-6, 1.4), xlim=(0, 5)
    )
    ax[0].legend()
    ax[1].semilogy(age[keep], kernel.pure_popiii_ew_angstrom[keep], "o-", color="#D55E00")
    ax[1].set(
        xlabel="Burst age [Myr]",
        ylabel=r"Pure Pop III $W_0$(1640) [Å]",
        xlim=(0, 5),
        ylim=(0.003, 150),
    )
    fig.savefig(figures / "ssp_age.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained", sharey=True)
    for ax, result in zip(axes, results):
        x = np.asarray(result["line_edges"])
        centers = (x[:-1] + x[1:]) / 2
        for e, color in ((0.01, "#0072B2"), (0.03, "#009E73"), (0.1, "#D55E00")):
            d = result[f"line_lf_eps{e:g}"]
            y, se = np.array(d["value"]), np.array(d["cluster_mc_se"])
            mask = y > 0
            ax.plot(centers[mask], y[mask], color=color, label=rf"$\epsilon_b={100 * e:g}\%$")
            ax.fill_between(
                centers, np.maximum(y - se, 1e-15), y + se, where=mask, color=color, alpha=0.18
            )
        ax.set(
            xlabel=r"$\log_{10} L_{1640}$ [erg s$^{-1}$]",
            xlim=(38, 44),
            ylim=(1e-9, 1e-1),
            yscale="log",
            title=f"z={result['z']}",
        )
        ax.legend()
    axes[0].set_ylabel(r"$\phi$ [Mpc$^{-3}$ dex$^{-1}$]")
    fig.savefig(figures / "line_lf.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained", sharey=True)
    for ax, result in zip(axes, results):
        x = np.asarray(result["flux_grid"])
        total = result["uv_bright_density"]["value"]
        for key, style, label in (
            ("heii_flux_cumulative", "-", "All resolved emitters"),
            ("heii_flux_cumulative_uv_bright", "--", r"Also $M_{\rm UV}\leq-20$"),
        ):
            y = np.asarray(result[key]["value"])
            se = np.asarray(result[key]["cluster_mc_se"])
            ax.plot(x, y, style, label=label)
            ax.fill_between(x, np.maximum(y - se, 1e-15), y + se, alpha=0.15)
        ax.axhline(total, color=".4", lw=1, ls=":", label="UV-bright total")
        ax.set(
            xlabel=r"Integrated $F_{1640}$ [erg s$^{-1}$ cm$^{-2}$]",
            xscale="log",
            yscale="log",
            xlim=(1e-21, 1e-17),
            ylim=(1e-9, 1e-1),
            title=f"z={result['z']}",
        )
        ax.legend(fontsize=9)
    axes[0].set_ylabel(r"$n(>F)$ [Mpc$^{-3}$]")
    fig.savefig(figures / "flux_cumulative.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/heii_random_q.toml")
    )
    args = parser.parse_args()
    path, config = read_toml(args.config)
    require(
        config["reference_efficiency"] in config["efficiencies"], "reference efficiency missing"
    )
    require(all(0 < e <= 1 for e in config["efficiencies"]), "invalid efficiency")
    require(all(np.isfinite(f) and f > 0 for f in config["flux_cuts"]), "invalid flux cuts")
    uv, line = (resolve_path(path.parent, config[k]) for k in ("uv_ssp", "line_ssp"))
    executed_uv = resolve_path(path.parent, config["executed_uv_ssp"])
    require(
        np.array_equal(np.loadtxt(uv), np.loadtxt(executed_uv)),
        "original and executed UV numeric tables differ",
    )
    kernel = load_kernel(uv, line)
    source_hashes = {
        str(p.resolve()): digest(p)
        for p in (
            path,
            uv,
            executed_uv,
            line,
            uv.parent / "README",
            Path(__file__),
            Path("auroralf/ssp/uv1600.py"),
            Path("auroralf/ssp/convolution.py"),
        )
    }
    record_experiment_sources(source_hashes)
    results = [
        analyze_case(case, config, path.parent, kernel, executed_uv, source_hashes)
        for case in config["cases"]
    ]
    output = resolve_path(path.parent, config["output"])
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(
        config=config,
        source_sha256=source_hashes,
        cases=results,
        caseb_max_relative_error=kernel.caseb_max_relative_error,
        assumptions=[
            "Pop III contribution only; no assigned Pop II/AGN/shock line",
            "Case B, matching logE SSP; zero escape, no dust or lensing",
            "UV selection preserves saved mixed PopII stellar1600 and PopIII total1500 convention",
            "HeII uses all resolved event ages within SSP domain, not the UV 100Myr truncation",
            "Left-censored events have unknown line luminosity and are excluded from line counts",
            "Finite IMF sampling, gas/pristine feedback and photoionization not modeled",
            "MC SE groups 1000 histories sharing one sampled halo mass; excludes SSP/gas/model uncertainty",
            "Interpolation alternatives diagnose the 1Myr SSP grid, not physical uncertainty bounds",
            "Total galaxy EW unavailable: no saved PopII continuum at1640 or PopII nebular continuum",
        ],
        primary_source="https://arxiv.org/html/1008.2114v2",
        source_sections=["Table1", "2.1.3 equation3", "4.3", "5.3"],
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    np.savetxt(
        output / "ssp_kernel.csv",
        np.c_[
            kernel.age_myr,
            kernel.line_per_msun,
            kernel.q2_per_msun,
            kernel.pure_popiii_ew_angstrom,
            kernel.uv_per_msun,
        ],
        delimiter=",",
        header="age_myr,L1640_erg_s_Msun,Q2_s-1_Msun,pure_popIII_EW_A,Lnu1500_erg_s_Hz_Msun",
    )
    plot_results(results, kernel, resolve_path(path.parent, config["figures"]))
    print(output / "summary.json")


if __name__ == "__main__":
    main()
