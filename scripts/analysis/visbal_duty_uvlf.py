"""Controlled Visbal duty transplant onto saved Pop II samples; not causal evolution."""

import argparse
import json
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from auroralf.constants import (
    PLANCK18_H0_KM_S_MPC,
    PLANCK18_OMEGA_B,
    PLANCK18_OMEGA_M,
)
from auroralf.experiments.artifacts import (
    digest,
    read_completed_manifest,
    record_experiment_sources,
    require,
)
from auroralf.experiments.visbal import (
    VisbalDutyConfig,
    active_weights,
    mag,
    uv_coefficient,
)
from auroralf.ssp import load_popiii_uv_luminosity_table

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/experiments/visbal_duty.toml"
    )
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = VisbalDutyConfig.load(config_path)
    cfg = config.as_metadata()
    out = config.output
    if out.exists():
        raise FileExistsError(out)
    hashes = {}
    record_experiment_sources(hashes)

    def record(path):
        hashes[str(path)] = digest(path)

    record(config_path)
    record(Path(__file__))
    samples = []
    for run in (cfg["full_run"], cfg["low_run"]):
        path = config.samples_root / run
        manifest = read_completed_manifest(
            path, required_products=(f"samples_z{config.z:g}.npz", "uvlf.npz")
        )
        record(path / "manifest.json")
        for name in manifest["products"]:
            record(path / name)
        with np.load(path / f"samples_z{cfg['z']:g}.npz") as d:
            samples.append({k: d[k] for k in ("popii", "weight", "halo_mass_msun")})
        with np.load(path / "uvlf.npz") as d:
            floor = float(d["atomic_mass_msun"][0])
            if run == cfg["low_run"]:
                split = 10 ** float(d["logM_max"][0])
    full, low = samples
    mass = np.r_[full["halo_mass_msun"], low["halo_mass_msun"]]
    p2 = np.r_[full["popii"], low["popii"]]
    w = np.r_[full["weight"] * (full["halo_mass_msun"] > split), low["weight"]]
    require(
        np.all(np.isfinite(p2)) and np.all(w >= 0),
        "Samples contain invalid luminosities or weights",
    )
    eligible = (mass >= floor) & (mass <= 2 * floor)
    ssp = config.popiii_ssp
    record(ssp)
    ages, lum = load_popiii_uv_luminosity_table(ssp)
    coefficient = uv_coefficient(ages, lum, cfg["csfr_age_myr"])
    checkx = np.r_[0.0, np.geomspace(ages[0], cfg["csfr_age_myr"], 100000)]
    checky = np.interp(np.log(np.maximum(checkx, ages[0])), np.log(ages), lum)
    np.testing.assert_allclose(coefficient, np.trapezoid(checky, checkx) * 1e6, rtol=1e-6)
    cosmo = FlatLambdaCDM(
        H0=PLANCK18_H0_KM_S_MPC, Om0=PLANCK18_OMEGA_M, Ob0=PLANCK18_OMEGA_B, Tcmb0=0
    )
    th = (1 / cosmo.H(cfg["z"])).to_value(u.yr)
    fb = PLANCK18_OMEGA_B / PLANCK18_OMEGA_M
    original = config.baseline
    record(original)
    old = np.load(original)
    edges = old["bin_edges"]
    widths = np.diff(edges)
    centers = (edges[1:] + edges[:-1]) / 2

    def hist(luminosity, weight):
        return np.histogram(mag(luminosity), bins=edges, weights=weight)[0] / widths

    baseline = hist(p2, w)
    np.testing.assert_allclose(baseline, old[f"z{config.z:g}_baseline_phi"], rtol=1e-9, atol=1e-14)
    curves = {"baseline": baseline, "burst10": old[f"z{config.z:g}_eps0.1_phi"]}
    rows = []
    for duty in cfg["duties"]:
        wa, wi = active_weights(w, eligible, duty)
        sfr = cfg["fstar"] * fb * mass / (duty * th)
        l3 = sfr * coefficient
        curves[f"duty{duty}"] = hist(p2, wi) + hist(p2 + l3, wa)
        rows.append(
            dict(
                duty=duty,
                effective_time_myr=duty * th / 1e6,
                brightest_popiii=float(np.min(mag(l3[(wa > 0)]))),
                mean_popiii_emissivity=float(np.sum(wa * l3)),
                density_total_le20=float(
                    np.sum(wi * (mag(p2) <= -20) + wa * (mag(p2 + l3) <= -20))
                ),
            )
        )
    np.testing.assert_allclose(
        [r["mean_popiii_emissivity"] for r in rows], rows[0]["mean_popiii_emissivity"], rtol=1e-12
    )
    plt.style.use("apj")
    fig, ax = plt.subplots(figsize=(9, 6.7))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.28, top=0.87)
    ax.plot(centers, baseline, color="black", lw=2, label="Pop II only")
    ax.plot(
        centers, curves["burst10"], color=".5", ls=":", lw=2, label=r"First-crossing burst (10\%)"
    )
    for duty, color in zip(cfg["duties"], ["#a44b20", "#008c85", "#3973ac"]):
        y = curves[f"duty{duty}"]
        ax.plot(
            centers,
            np.where(y > 0, y, np.nan),
            color=color,
            lw=2,
            label=rf"Visbal-style: duty = {100 * duty:g}\%",
        )
    model_handles, _ = ax.get_legend_handles_labels()
    handles = []
    for rel, label, marker, color in [
        (
            "redshift_14/whitler25_jades_z14p3.npz",
            r"Whitler+25: JADES, $z\geq14$ (median 14.3)",
            "o",
            "#333333",
        ),
        (
            "redshift_15/donnan24_primer_z14p5.npz",
            r"Donnan+24: PRIMER, $13.5<z<15.5$ (tentative)",
            "s",
            "#945399",
        ),
        (
            "redshift_15/naidu26_mom_jades_spectroscopic_z14p5.npz",
            r"Naidu+26: MoM+JADES spec., $14<z<15$",
            "D",
            "#207798",
        ),
    ]:
        path = config.observations / rel
        record(path)
        d = np.load(path)
        require(
            not np.any(d["is_upper_limit"]),
            "Selected observation table contains unsupported upper limits",
        )
        handles.append(
            ax.errorbar(
                d["muverr"],
                d["phierr"],
                xerr=d["mag_err"],
                yerr=[d["phi_err_lo"], d["phi_err_up"]],
                fmt=marker,
                color=color,
                mfc="white",
                ms=7,
                capsize=3,
                label=label,
                zorder=10,
            )
        )
    ax.set(
        xlim=(-22, -12),
        ylim=(3e-8, 0.2),
        yscale="log",
        xlabel=r"$M_{\rm UV}$ [AB mag]",
        ylabel=r"$\phi$ [Mpc$^{-3}$ mag$^{-1}$]",
    )
    ax.legend(handles=model_handles, loc="upper left", fontsize=10, frameon=False)
    fig.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0.12, 0.06), fontsize=10, frameon=False
    )
    fig.suptitle(rf"Visbal-style statistical comparison: $z={config.z:g}$", y=0.97, fontsize=16)
    fig.text(
        0.13,
        0.91,
        rf"$f_\star={100 * config.fstar:g}\%$; project $M_{{\rm cool}}$; constant-SFR UV calibration at {config.csfr_age_myr:g} Myr",
        fontsize=11,
    )
    fig.text(
        0.13,
        0.025,
        "Formal duty mixture, not a causal burst model. Same SSP / Pop II baseline; no dust.",
        fontsize=10,
    )
    out.mkdir(parents=True)
    np.savez_compressed(out / "uvlf.npz", bin_edges=edges, **curves)
    report = dict(
        config=cfg,
        mcool_msun=floor,
        hubble_time_myr=th / 1e6,
        uv_per_sfr=coefficient,
        rows=rows,
        inputs_sha256=hashes,
        limitations=[
            "Project threshold/cosmology/logE, not exact paper reproduction.",
            "100 Myr CSFR UV normalization is additional; not a paper HeII-to-UV conversion.",
            "Independent duty occupation added to fixed PopII; inactive objects keep PopII.",
            "Duty times Hubble time can be shorter than calibration age: no finite-burst interpretation.",
            "No cooling/enrichment/feedback evolution, redshift averaging or new histories.",
            "SSP first age held constant to zero; no old-age extrapolation.",
            "No new sampling convergence claim or joint fit to overlapping observations.",
        ],
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    fig.savefig(out / "uvlf.png", dpi=170)
    fig.savefig(out / "uvlf.pdf")
    plt.close(fig)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("inputs_sha256", "config")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
