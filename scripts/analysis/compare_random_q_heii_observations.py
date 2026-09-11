"""UV-conditioned He II comparison using verified existing random-q populations.

Population redshifts remain explicit: distance conversion does not evolve a sample.
This produces population CDF diagnostics, not an instrument likelihood or a fit.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from auroralf.experiments.artifacts import digest, require, resolve_path, verify_files
from auroralf.experiments.heii import evaluate_kernel, load_kernel
from auroralf.mah import Cosmology
from auroralf.uvlf import uv_luminosity_to_muv
from scripts.analysis.analyze_random_q_heii import weighted_quantiles

METHODS = ("linear_log_age", "linear_age", "log_log_age")


def conditional_stats(flux, weight, cluster, threshold):
    """Preserve zero emitters; exclude unknowns with their weight reported explicitly."""
    require(len(flux) > 0 and np.all(weight > 0), "empty or invalid population")
    require(np.isfinite(weight).all(), "nonfinite weights")
    known = np.isfinite(flux)
    require(known.any() and np.all(flux[known] >= 0), "invalid known flux")
    f, w, g = flux[known], weight[known], cluster[known]
    total = w.sum()
    group_weights = np.bincount(g, weights=w)
    return {
        "n_selected": len(flux),
        "n_known": len(f),
        "density_selected_mpc3": float(weight.sum()),
        "unknown_weight_fraction": float(weight[~known].sum() / weight.sum()),
        "zero_fraction_known": float(w[f == 0].sum() / total),
        "flux_q16_q50_q84": weighted_quantiles(f, w),
        "fraction_at_or_below_reference_flux": float(w[f <= threshold].sum() / total),
        "mass_clusters_known": int(np.count_nonzero(group_weights)),
        "effective_mass_clusters": float(total**2 / np.sum(group_weights**2)),
    }


def load_population(case, eps, kernel, hashes):
    cosmology = Cosmology()
    parts = {key: [] for key in ("muv", "mass", "age", "status", "weight", "cluster")}
    cluster_offset = 0
    seeds = set()
    for record in case["runs"]:
        root = Path(record["path"])
        # Parent summary binds the manifest and complete product hashes to a validated UV replay.
        manifest_path = root / "manifest.json"
        require(str(manifest_path) in hashes, "unbound run manifest")
        manifest = json.loads(manifest_path.read_text())
        require(manifest["status"] == "complete", "incomplete run")
        require(manifest["products"] == record["products"], "run product record changed")
        verify_files(root, record["products"])
        require(manifest["config"]["z"] == case["z"], "population redshift differs")
        seed = manifest["config"]["seed"]
        require(seed == record["seed"] and seed not in seeds, "seed mismatch or duplicate")
        seeds.add(seed)
        with np.load(root / "samples.npz") as sample:
            muv = uv_luminosity_to_muv(sample["popii"] + eps * sample["popiii_per_efficiency"])
            select = (muv >= -22) & (muv < -18)
            shape = muv.shape
            require(shape == (record["n_mass"], record["n_tracks"]), "sample shape changed")
            rows = np.nonzero(select)[0]
            parts["muv"].append(muv[select])
            parts["mass"].append(
                sample["burst_halo_mass_msun"][select] * eps * cosmology.omega_b / cosmology.omega_m
            )
            parts["age"].append(sample["age_myr"][select])
            parts["status"].append(sample["status"][select])
            parts["weight"].append(sample["weight_per_track"][rows] / len(case["runs"]))
            parts["cluster"].append(rows + cluster_offset)
            cluster_offset += shape[0]
    population = {key: np.concatenate(value) for key, value in parts.items()}
    resolved = population["status"] == 1
    require(np.isin(population["status"], [0, 1, 2]).all(), "invalid status")
    for method in METHODS:
        line = np.zeros(len(resolved))
        line[population["status"] == 2] = np.nan
        line[resolved] = population["mass"][resolved] * evaluate_kernel(
            population["age"][resolved], kernel, method
        )
        population[method] = line
    return population


def compare(target, population, astro):
    conversion = target["magnification"] / (
        4 * np.pi * astro.luminosity_distance(target["z"]).to_value("cm") ** 2
    )
    result = {"target": target, "flux_per_luminosity": conversion, "bins": [], "target_windows": []}

    def summarize(lo, hi):
        mask = (population["muv"] >= lo) & (population["muv"] < hi)
        return {
            "muv_low": lo,
            "muv_high": hi,
            "methods": {
                method: conditional_stats(
                    population[method][mask] * conversion,
                    population["weight"][mask],
                    population["cluster"][mask],
                    target["flux"],
                )
                for method in METHODS
            },
        }

    for lo in np.arange(-22, -18, 0.5):
        result["bins"].append(summarize(float(lo), float(lo + 0.5)))
    for width in (0.25, 0.5):
        result["target_windows"].append(summarize(target["muv"] - width, target["muv"] + width))
    return result


def plot(results, figures, previews):
    plt.style.use("apj")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), sharey=True)
    for ax, result in zip(axes, results):
        target = result["target"]
        bins = result["bins"]
        x = [(b["muv_low"] + b["muv_high"]) / 2 for b in bins]
        quantiles = np.array([b["methods"][METHODS[0]]["flux_q16_q50_q84"] for b in bins])
        require(np.all(quantiles > 0), "zero quantile needs explicit non-log display")
        ax.fill_between(
            x,
            quantiles[:, 0],
            quantiles[:, 2],
            color="#4477AA",
            alpha=0.22,
            label=r"Population 16--84\%",
        )
        ax.plot(x, quantiles[:, 1], "o-", color="#225588", label=r"Median: $L$ vs $\log a$")
        alternative = [b["methods"]["log_log_age"]["flux_q16_q50_q84"][1] for b in bins]
        ax.plot(x, alternative, "--", color="#228833", label=r"Median: $\log L$ vs $\log a$")
        upper = target["measurement"] == "upper_limit"
        ax.errorbar(
            target["muv"],
            target["flux"],
            xerr=target["muv_error"],
            yerr=0.45 * target["flux"] if upper else target["flux_error"],
            uplims=upper,
            fmt="s",
            color="#CC3311",
            capsize=4,
            ms=6,
            label=r"JWST: $3\sigma$ upper limit" if upper else r"JWST: deblended flux $\pm1\sigma$",
        )
        ax.set_title(
            f"{target['name']}: z = {target['z']}\nPopulation z = {target['population_z']}; adopted "
            + rf"$\mu = {target['magnification']}$",
            fontsize=12,
        )
        ax.set_xlabel(r"Intrinsic $M_{\rm UV}$ [AB mag]")
        ax.set_yscale("log")
        ax.set_xlim(-22, -18)
        ax.set_ylim(8e-22, 1e-17)
        ax.legend(loc="lower left", fontsize=9, frameon=False)
    axes[0].set_ylabel(r"$F_{\rm HeII\,1640}$ [erg s$^{-1}$ cm$^{-2}$]")
    fig.suptitle(r"Pop III Case B, $\epsilon_b=0.03$: nearby-redshift comparison", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figures.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "uv_heii_observations.pdf")
    fig.savefig(previews / "uv_heii_observations.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-summary", type=Path, default=Path("data_save/heii_random_q_20260910/summary.json")
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("external_data/observations/heii/jwst_targets.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("data_save/heii_observations_20260910"))
    parser.add_argument(
        "--figures", type=Path, default=Path("slides/heii_random_q_20260910/assets")
    )
    parser.add_argument("--previews", type=Path, default=Path("outputs/heii_observations_20260910"))
    args = parser.parse_args()
    parent = json.loads(args.parent_summary.read_text())
    verify_files(Path.cwd(), parent["source_sha256"])
    config = parent["config"]
    require(config["reference_efficiency"] == 0.03, "expected epsilon=0.03")
    base = Path("configs/experiments")
    kernel = load_kernel(
        resolve_path(base, config["uv_ssp"]), resolve_path(base, config["line_ssp"])
    )
    cosmology = Cosmology()
    astro = FlatLambdaCDM(H0=cosmology.h0_km_s_mpc, Om0=cosmology.omega_m, Ob0=cosmology.omega_b)
    observations = json.loads(args.observations.read_text())
    results = []
    for target in observations["targets"]:
        require(target["measurement"] in ("deblended_flux", "upper_limit"), "unknown measurement")
        case = next(case for case in parent["cases"] if case["z"] == target["population_z"])
        population = load_population(
            case, config["reference_efficiency"], kernel, parent["source_sha256"]
        )
        results.append(compare(target, population, astro))
        print(target["name"], json.dumps(results[-1]["target_windows"][0]), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "epsilon": 0.03,
        "results": results,
        "source_sha256": {
            str(p.resolve()): digest(p)
            for p in (args.parent_summary, args.observations, Path(__file__))
        },
        "parent_source_sha256": parent["source_sha256"],
        "assumptions": [
            "Neighboring-redshift population approximation: only luminosity distance changes; histories, HMF weights and UV stay at population_z.",
            "PopIII-only Case-B, zero escape and no dust; no assigned PopII, AGN or shock line.",
            "Saved mixed UV proxy (PopII stellar1600 + PopIII total1500) retained, not full synthetic spectrophotometry.",
            "Quantiles are weighted population spread, not confidence intervals on a fit or Monte Carlo errors.",
            "Unknown left-censored events excluded with weight fraction recorded; status0 retained as zero PopIII line.",
            "0.5mag bins; target half-width0.25 and0.5 are finite selection windows, NOT marginalization over observed UV uncertainty.",
            "CDF at measured flux or upper limit is a true-model-flux diagnostic; not a p-value, exclusion probability, or noise-convolved nondetection probability.",
            "Interpolation alternatives are sensitivity checks, not physical uncertainty bounds.",
            "No total galaxy EW, line profile, aperture forward model or survey selection likelihood.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot(results, args.figures, args.previews)


if __name__ == "__main__":
    main()
