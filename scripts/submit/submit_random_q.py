"""Freeze and submit the registered random-q campaign, without time/memory requests."""

import argparse
import json
import re
import shlex
import subprocess
from dataclasses import asdict
from functools import partial
from pathlib import Path, PurePosixPath

from auroralf.experiments.deployment import (
    RemoteSite,
    freeze_release,
    random_q_release_files,
    verification_command,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SITE_CONFIG = ROOT / "configs/compute/random_q_cp6.toml"


def ssh(cmd, *, host):
    return subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, cmd], text=True
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--configs", nargs="+")
    p.add_argument("--site-config", type=Path, default=DEFAULT_SITE_CONFIG)
    a = p.parse_args()
    if not re.fullmatch("[a-z0-9-]+", a.release):
        raise ValueError("release name")
    local = ROOT / "outputs/deployments" / a.release
    site = RemoteSite.load(a.site_config)
    remote = str(PurePosixPath(site.release_root) / a.release)
    run_remote = partial(ssh, host=site.host)
    configs = a.configs or ["configs/experiments/random_q_R032.toml"]
    if any(not re.fullmatch(r"configs/experiments/random_q_R\d{3}\.toml", c) for c in configs):
        raise ValueError("invalid config path")
    if not 1 <= len(configs) <= 4:
        raise ValueError(
            "submission accepts one to four explicit science configs; no separate preflight"
        )
    if a.prepare:
        files = random_q_release_files(
            ROOT,
            configs,
            [
                "scripts/experiments/random_q_burst.py",
                "scripts/run/run_random_q_burst.py",
                "scripts/submit/submit_random_q.py",
                "tests/test_random_q_burst.py",
            ],
        )
        freeze_release(ROOT, local, files)
        run_remote("mkdir " + shlex.quote(remote))
        subprocess.run(
            ["rsync", "-a", str(local) + "/", site.host + ":" + remote + "/"], check=True
        )
        run_remote(
            "ln -s " + shlex.quote(site.python_environment) + " " + shlex.quote(remote + "/.venv")
        )
        run_remote("mkdir " + shlex.quote(remote + "/outputs"))
        print("prepared " + remote, flush=True)
        return
    if not (local / "deployment.json").is_file():
        raise FileNotFoundError(local)
    nodes = run_remote('sinfo -N -p cp6 -h -o "%N|%t|%c|%C"')
    jobs = run_remote("squeue -r -u " + shlex.quote(site.user) + ' -h -o "%i|%j|%T|%D"')
    usable = [
        row.split("|")
        for row in nodes.splitlines()
        if not any(s in row.split("|")[1] for s in ("down", "drain", "drng", "maint", "inval"))
    ]
    if not usable or any(int(row[2]) != 56 for row in usable):
        raise RuntimeError("unexpected cp6 CPUs")
    # Conservative all-user allocated/requested nodes; never touch another job.
    reserved = sum(int(row.split("|")[3]) for row in jobs.splitlines())
    if reserved + len(configs) > 10:
        raise RuntimeError("ten node ceiling")
    base = [
        "sbatch",
        "--parsable",
        "--partition=cp6",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=56",
        "--exclusive",
        "--chdir=" + remote,
    ]
    verify = verification_command()
    cases = " ".join(f"{i}) cfg={name} ;;" for i, name in enumerate(configs))
    selection = (
        ("cfg=" + configs[0])
        if len(configs) == 1
        else ('case "$SLURM_ARRAY_TASK_ID" in ' + cases + " *) exit 64 ;; esac")
    )
    array = [] if len(configs) == 1 else [f"--array=0-{len(configs) - 1}%{len(configs)}"]
    batch = (
        base
        + array
        + [
            "--job-name=AUR-random-q",
            "--output=" + remote + "/outputs/batch_%j.out",
            "--error=" + remote + "/outputs/batch_%j.err",
            "--wrap",
            "set -eu; export PYTHONPATH=.; export MPLBACKEND=Agg; "
            + verify
            + "; "
            + selection
            + '; exec .venv/bin/python scripts/run/run_random_q_burst.py --config "$cfg"',
        ]
    )
    record = dict(remote=remote, site=asdict(site), nodes=nodes, jobs=jobs, batch=batch)
    mode = "apply" if a.apply else "dry"
    out = ROOT / "outputs" / f"{a.release}_{mode}.json"
    if out.exists():
        raise FileExistsError(out)
    print(shlex.join(batch), flush=True)
    try:
        if a.apply:
            jid = run_remote(shlex.join(batch)).strip()
            record["submission_response"] = jid
            if not jid.isdigit():
                raise RuntimeError("ambiguous job response: " + jid)
            record["batch_job_id"] = jid
            print("submitted", record["batch_job_id"], flush=True)
    finally:
        out.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
