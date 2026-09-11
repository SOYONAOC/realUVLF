"""Validate random-q products, cluster-weighted UVLF and z14.5 observation plot."""

import argparse
import json
from pathlib import Path

import numpy as np

from auroralf.experiments.artifacts import digest, read_completed_manifest
from auroralf.uvlf import uv_luminosity_to_muv


def observation_specs(z):
    """Explicit project datasets; do not substitute another redshift's observations."""
    if z == 14.5:
        return [
            (
                "redshift_14/whitler25_jades_z14p3.npz",
                "Whitler+25: JADES, median z=14.3",
                "o",
                "#333333",
                True,
            ),
            (
                "redshift_15/donnan24_primer_z14p5.npz",
                "Donnan+24: PRIMER (tentative)",
                "s",
                "#945399",
                True,
            ),
            (
                "redshift_15/naidu26_mom_jades_spectroscopic_z14p5.npz",
                r"Naidu+26: MoM+JADES spec., $14<z<15$",
                "D",
                "#207798",
                True,
            ),
        ]
    if z == 12.5:
        # These three established project tables contain detections and asymmetric
        # errors, using the legacy schema without an upper-limit flag column.
        return [
            ("redshift_12p5/donnan24.npz", r"Donnan+24: $z\simeq12.5$", "o", "#333333", False),
            ("redshift_12p5/bouwens.npz", r"Bouwens+23: $z\simeq12$--$13$", "s", "#945399", False),
            (
                "redshift_12p5/harikane23_uvlf_z12.npz",
                r"Harikane+23: $z\simeq12$",
                "D",
                "#207798",
                False,
            ),
        ]
    raise ValueError("No explicitly selected observation set for redshift " + str(z))


def summarize_run(path, edges):
    manifest = read_completed_manifest(path, required_products=("samples.npz",))
    c = manifest["config"]
    with np.load(path / "samples.npz") as d:
        samples = dict(d)
    n, nt = c["n_mass"], c["n_tracks"]
    w = samples["weight_per_track"]
    if w.shape != (n,) or np.any(w <= 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights")
    for key in ("popii", "popiii_per_efficiency", "fixed_q1_per_efficiency", "logq", "status"):
        if samples[key].shape != (n, nt) or not np.all(np.isfinite(samples[key])):
            raise ValueError(key)
    status = samples["status"]
    if not np.isin(status, [0, 1, 2]).all():
        raise ValueError("status")
    for key in ("burst_time_gyr", "burst_halo_mass_msun", "age_myr"):
        if not (
            np.isfinite(samples[key][status == 1]).all()
            and np.isnan(samples[key][status != 1]).all()
        ):
            raise ValueError("event coordinates " + key)
    if np.any(samples["popiii_per_efficiency"][status != 1] != 0):
        raise ValueError("non-event light")
    p2 = samples["popii"]
    p3 = samples["popiii_per_efficiency"]
    curves = {"baseline": p2, "fixed_q1_eps0.1": p2 + 0.1 * samples["fixed_q1_per_efficiency"]}
    curves.update({f"eps{e:g}": p2 + e * p3 for e in c["efficiencies"]})
    width = np.diff(edges)
    outputs = {}
    summary = []
    for name, luminosity in curves.items():
        mag = uv_luminosity_to_muv(luminosity)
        per_mass = np.array([np.histogram(row, bins=edges)[0] for row in mag]) * w[:, None] / width
        phi = per_mass.sum(axis=0)
        se = np.sqrt(n * np.var(per_mass, axis=0, ddof=1))
        counts = np.histogram(mag, bins=edges)[0]
        outputs[name] = (phi, se, counts)
        row = dict(model=name, thresholds=[])
        for cut in (-18.0, -19.0, -20.0, -21.0, -22.0):
            selected = mag <= cut
            baseline = uv_luminosity_to_muv(p2) <= cut
            if np.any(baseline & ~selected):
                raise ValueError("positive light lost a bright galaxy")
            contrib = selected.sum(axis=1) * w
            added = (selected & ~baseline).sum(axis=1) * w
            row["thresholds"].append(
                dict(
                    cut=cut,
                    total=float(contrib.sum()),
                    added=float(added.sum()),
                    total_se=float(np.sqrt(n * np.var(contrib, ddof=1))),
                    added_se=float(np.sqrt(n * np.var(added, ddof=1))),
                    added_tracks=int(np.sum(selected & ~baseline)),
                    added_from_q_gt100=float(
                        ((selected & ~baseline & (samples["logq"] > 2)).sum(axis=1) * w).sum()
                    )
                    if name.startswith("eps")
                    else 0.0,
                )
            )
        if name.startswith("eps"):
            e = float(name[3:])
            row["brightest_popiii"] = (
                float(np.nanmin(uv_luminosity_to_muv(e * p3))) if np.any(p3 > 0) else None
            )
        summary.append(row)
    logq = samples["logq"]
    diag = dict(
        logq_mean=float(logq.mean()),
        logq_std=float(logq.std()),
        q_central_fraction=float(np.mean((logq >= -1) & (logq <= 2))),
        q_high_fraction=float(np.mean(logq > 2)),
        not_triggered_fraction=float(np.mean(status == 0)),
        left_censored_fraction=float(np.mean(status == 2)),
        resolved_fraction=float(np.mean(status == 1)),
        uv_active_fraction=float(np.mean(p3 > 0)),
    )
    return c, outputs, summary, diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    edges = np.arange(-26.0, -7.99, 0.5)
    loaded = [summarize_run(path, edges) for path in a.runs]
    cfg = loaded[0][0]
    if len({x[0]["seed"] for x in loaded}) != len(loaded):
        raise ValueError("duplicate seeds")
    for c, *_ in loaded[1:]:
        for key in cfg:
            if key not in ("seed", "run_id", "popii_ssp", "popiii_ssp") and c[key] != cfg[key]:
                raise ValueError("incompatible batches " + key)
    data = {"bin_edges": edges}
    rows = []
    for key in loaded[0][1]:
        stack = np.array([x[1][key][0] for x in loaded])
        data[key] = stack.mean(axis=0)
        data[key + "_cluster_se"] = np.sqrt(
            np.sum([x[1][key][1] ** 2 for x in loaded], axis=0)
        ) / len(loaded)
        data[key + "_batch_se"] = (
            stack.std(axis=0, ddof=1) / np.sqrt(len(loaded))
            if len(loaded) > 1
            else np.full(len(edges) - 1, np.nan)
        )
        data[key + "_counts"] = np.sum([x[1][key][2] for x in loaded], axis=0)
        index = [r["model"] for r in loaded[0][2]].index(key)
        rows_i = [x[2][index] for x in loaded]
        row = dict(model=key, thresholds=[])
        for j in range(len(rows_i[0]["thresholds"])):
            vals = [r["thresholds"][j] for r in rows_i]
            out = {"cut": vals[0]["cut"]}
            for field in ("total", "added", "added_from_q_gt100"):
                out[field] = float(np.mean([v[field] for v in vals]))
            for field in ("total_se", "added_se"):
                out[field] = float(np.sqrt(np.sum([v[field] ** 2 for v in vals])) / len(vals))
            out["added_tracks"] = sum(v["added_tracks"] for v in vals)
            row["thresholds"].append(out)
        if "brightest_popiii" in rows_i[0]:
            vals = [r["brightest_popiii"] for r in rows_i if r["brightest_popiii"] is not None]
            row["brightest_popiii"] = min(vals) if vals else None
        rows.append(row)
    import matplotlib.pyplot as plt

    plt.style.use("apj")
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.28)
    x = (edges[1:] + edges[:-1]) / 2
    for key, color, label, ls in [
        ("baseline", "black", "Pop II only", "-"),
        ("fixed_q1_eps0.1", ".5", r"Fixed $q=1$, $\epsilon_b=10\%$", ":"),
    ] + [
        (f"eps{e:g}", color, rf"Random $q$: $\epsilon_b={100 * e:g}\%$", ls)
        for e, color, ls in zip(
            cfg["efficiencies"],
            ["#9564bf", "#3973ac", "#008c85", "#d75a00", "#874e21"],
            [":", "--", "-", "-.", "-"],
        )
    ]:
        y = data[key]
        se = data[key + "_cluster_se"]
        good = (y > 0) & (data[key + "_counts"] >= 8)
        ax.plot(x, np.where(good, y, np.nan), color=color, ls=ls, lw=1.8, label=label)
        if key.startswith("eps"):
            ax.fill_between(
                x,
                np.where(good & (y > se), y - se, np.nan),
                np.where(good, y + se, np.nan),
                color=color,
                alpha=0.12,
            )
    mh, _ = ax.get_legend_handles_labels()
    obs = []
    sources = {}
    for name, label, marker, color, has_limit_column in observation_specs(cfg["z"]):
        f = a.observations / name
        sources[str(f)] = digest(f)
        with np.load(f) as d:
            if has_limit_column and "is_upper_limit" not in d:
                raise ValueError("missing required upper-limit metadata")
            if "is_upper_limit" in d and np.any(d["is_upper_limit"]):
                raise ValueError("upper limits require distinct rendering")
            obs.append(
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
    visible = (x >= -24) & (x <= -12)
    top = max(np.max(data[k][visible]) for k in loaded[0][1])
    ax.set(
        xlim=(-24, -12),
        ylim=(3e-9, max(0.02, top * 1.5)),
        yscale="log",
        xlabel=r"$M_{\rm UV}$ [AB mag]",
        ylabel=r"$\phi$ [Mpc$^{-3}$ mag$^{-1}$]",
    )
    ax.legend(handles=mh, loc="upper left", frameon=False, fontsize=10)
    fig.legend(
        handles=obs, loc="lower left", bbox_to_anchor=(0.12, 0.065), frameon=False, fontsize=10
    )
    fig.suptitle(rf"Random first-crossing Pop III burst: $z={cfg['z']:g}$", y=0.975, fontsize=16)
    fig.text(
        0.13,
        0.915,
        r"$\log_{10}q\sim\mathcal{N}(0.5,1.5^2)$; one draw per track; no tail truncation",
        fontsize=11,
    )
    fig.text(
        0.13,
        0.035,
        "Pop II stellar 1600A + Pop III total 1500A; no dust. No pristine/enrichment closure.",
        fontsize=9,
    )
    fig.text(
        0.13,
        0.012,
        "Shading: mass-cluster MC SE; not cosmic variance or model uncertainty. No curve smoothing.",
        fontsize=9,
    )
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output / "uvlf.npz", **data)
    report = dict(
        config=cfg,
        runs=[str(p) for p in a.runs],
        rows=rows,
        diagnostics=[x[3] for x in loaded],
        per_batch_rows=[x[2] for x in loaded],
        manifests_sha256={str(p / "manifest.json"): digest(p / "manifest.json") for p in a.runs},
        observations_sha256=sources,
        analysis_sha256=digest(Path(__file__)),
        limitations=[
            "User-assumed independent q distribution, not calibrated physics.",
            "No pristine survival or feedback; fixed PopII paired baseline.",
            "Finite final halo mass integration 1e5-1e12 Msun, NOT a q truncation.",
            "100Myr UV window and left-censored treatment; not a full stellar population history.",
            "Single-batch cluster SE or multiple-batch comparisons are diagnostics, not a convergence proof.",
        ],
    )
    (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    fig.savefig(a.output / "uvlf.png", dpi=170)
    fig.savefig(a.output / "uvlf.pdf")
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
