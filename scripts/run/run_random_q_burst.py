"""Frozen-config SLURM runner for the random-q sensitivity experiment."""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from auroralf.experiments.artifacts import digest as digest
from auroralf.experiments.random_q import Config, initialize_worker, one_mass


def check_compute_site(site, env):
    partition = env.get("SLURM_JOB_PARTITION")
    if site == "cp6" and partition == "cp6":
        return
    if site == "fat2" and partition == "fat" and env.get("SLURM_JOB_NODELIST") == "fat2":
        return
    if (
        site == "node123"
        and partition == "cpu"
        and env.get("SLURM_JOB_NODELIST") in ("node[1-3]", "node1,node2,node3")
        and env.get("SLURMD_NODENAME") in ("node1", "node2", "node3")
    ):
        return
    raise RuntimeError(
        "Allocation does not match explicit compute site; all debug partitions are forbidden"
    )


def shard_indices(n_mass, index, count):
    if not 1 <= count <= n_mass or not 0 <= index < count:
        raise ValueError("invalid shard coordinates")
    return np.arange(n_mass * index // count, n_mass * (index + 1) // count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--compute-site", choices=("cp6", "fat2", "node123"), default="cp6")
    p.add_argument("--slurm-shards", type=int, default=1)
    a = p.parse_args()
    if not a.validate_only:
        check_compute_site(a.compute_site, os.environ)
    cfg = Config.load(a.config)
    shard_index = int(os.environ["SLURM_PROCID"]) if a.slurm_shards > 1 else 0
    indices = shard_indices(cfg.n_mass, shard_index, a.slurm_shards)
    initialize_worker(cfg)
    target = ROOT / "data_save" / cfg.run_id
    if a.slurm_shards > 1:
        target = target.with_name(f"{cfg.run_id}-shard{shard_index}-of{a.slurm_shards}")
    stage = target.with_name("." + target.name + ".partial")
    if target.exists() or stage.exists():
        raise FileExistsError(target)
    if a.validate_only:
        print("validated", cfg.run_id, flush=True)
        return
    if (
        not os.environ.get("SLURM_JOB_ID")
        or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) < cfg.workers
    ):
        raise RuntimeError("SLURM allocation required")
    stage.mkdir(parents=True)
    sources = list((ROOT / "auroralf").rglob("*.py")) + [
        Path(__file__),
        ROOT / "scripts/experiments/random_q_burst.py",
        a.config.resolve(),
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    hashes = {str(s.relative_to(ROOT)): digest(s) for s in sources}
    inputs = {s: digest(s) for s in (cfg.popii_ssp, cfg.popiii_ssp)}
    manifest = dict(
        config=asdict(cfg),
        code_sha256=hashes,
        input_sha256=inputs,
        job_id=os.environ["SLURM_JOB_ID"],
        compute_site=a.compute_site,
        partition=os.environ.get("SLURM_JOB_PARTITION"),
        node_list=os.environ.get("SLURM_JOB_NODELIST"),
        status="running",
        started_utc=datetime.now(timezone.utc).isoformat(),
        scope="User-prescribed UV-only, fixed PopII, no pristine/enrichment/feedback model; PopII stellar1600A and PopIII total1500A, no dust",
        left_censored_policy="no fabricated event; zero within 100Myr UV window only because history span is longer",
        interpolation="first log(M/Mcool) crossing linear in time; logM interpolation at crossing",
        q_distribution="untruncated independent lognormal, once per track; not literature calibrated",
    )
    if a.slurm_shards > 1:
        manifest["sharding"] = dict(
            index=shard_index,
            count=a.slurm_shards,
            global_n_mass=cfg.n_mass,
            weight_normalization="global_n_mass; sum disjoint shards, never average",
        )
    import tarfile

    with tarfile.open(stage / "source.tar.gz", "w:gz") as archive:
        for s in sources:
            archive.add(s, arcname=str(s.relative_to(ROOT)), recursive=False)
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        from auroralf.mah import Cosmology
        from auroralf.seeding import derive_hmf_mass_seed
        from auroralf.uvlf.hmf_sampling import prepare_reed07_hmf_interpolator

        rng = np.random.default_rng(derive_hmf_mass_seed(cfg.seed, cfg.z))
        mass = 10 ** rng.uniform(cfg.logmass_min, cfg.logmass_max, cfg.n_mass)
        hmf = prepare_reed07_hmf_interpolator(
            log10_halo_mass_min_msun=cfg.logmass_min,
            log10_halo_mass_max_msun=cfg.logmass_max,
            z_obs=cfg.z,
            cosmology=Cosmology(),
        )
        weight = (
            (cfg.logmass_max - cfg.logmass_min)
            * np.log(10)
            * mass
            * hmf.evaluate(mass)
            / cfg.n_mass
            / cfg.n_tracks
        )
        samples = {}
        start = time.monotonic()
        with ProcessPoolExecutor(
            max_workers=cfg.workers,
            mp_context=mp.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(cfg,),
        ) as pool:
            # Python 3.13 has no Executor.map buffersize: explicitly bound in-flight batches.
            for first in range(0, len(indices), 2 * cfg.workers):
                tasks = [(int(i), mass[i]) for i in indices[first : first + 2 * cfg.workers]]
                for i, values in pool.map(one_mass, tasks):
                    for key, v in values.items():
                        if key not in samples:
                            samples[key] = np.empty((len(indices), cfg.n_tracks), dtype=v.dtype)
                        samples[key][i - int(indices[0])] = v
                done = first + len(tasks)
                print(
                    f"shard={shard_index} mass_done={done}/{len(indices)} elapsed_s={time.monotonic() - start:.1f}",
                    flush=True,
                )
        samples.update(final_halo_mass_msun=mass[indices], weight_per_track=weight[indices])
        if a.slurm_shards > 1:
            samples["global_mass_index"] = indices
        for key in ("popii", "popiii_per_efficiency", "fixed_q1_per_efficiency", "logq"):
            if not np.all(np.isfinite(samples[key])):
                raise ValueError("invalid sample: " + key)
        np.savez_compressed(stage / "samples.npz", **samples)
        if any(digest(ROOT / k) != h for k, h in hashes.items()) or any(
            digest(k) != h for k, h in inputs.items()
        ):
            raise RuntimeError("inputs changed during run")
        manifest.update(
            status="complete",
            finished_utc=datetime.now(timezone.utc).isoformat(),
            products={f.name: digest(f) for f in stage.iterdir() if f.name != "manifest.json"},
        )
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(target)
        print("completed_artifacts=" + str(target), flush=True)
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
