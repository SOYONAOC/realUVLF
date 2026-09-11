"""Explicit local placement: fat2 or node1-3, one science realization, no preflight."""

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path

from auroralf.experiments.artifacts import digest as digest
from auroralf.experiments.deployment import (
    freeze_release,
    random_q_release_files,
    verification_command,
    verify_release,
)
from auroralf.experiments.random_q import Config

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ["configs/experiments/random_q_R028.toml"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--config", default=CONFIGS[0])
    parser.add_argument("--compute-site", choices=("fat2", "node123"), default="fat2")
    parser.add_argument(
        "--allow-queue",
        action="store_true",
        help="Keep explicit CPU request pending if currently busy; never oversubscribe",
    )
    args = parser.parse_args()
    if not re.fullmatch("[a-z0-9-]+", args.release):
        raise ValueError("release")
    if not re.fullmatch(r"configs/experiments/random_q_R\d{3}\.toml", args.config):
        raise ValueError("config path")
    config_paths = [args.config]
    node_names = ["node1", "node2", "node3"] if args.compute_site == "node123" else ["fat2"]
    node_list = ",".join(node_names)
    partition = "cpu" if args.compute_site == "node123" else "fat"
    target = ROOT / "outputs/deployments" / args.release
    if args.prepare:
        files = random_q_release_files(
            ROOT,
            config_paths,
            [
                "scripts/run/run_random_q_burst.py",
                "scripts/experiments/random_q_burst.py",
                "scripts/analysis/merge_random_q_shards.py",
                "scripts/submit/submit_random_q_fat2.py",
                "tests/test_random_q_burst.py",
            ],
        )
        freeze_release(ROOT, target, files)
        (target / ".venv").symlink_to(ROOT / ".venv", target_is_directory=True)
        (target / "outputs").mkdir()
        print("prepared", target)
        return
    verify_release(target)
    configs = [Config.load(target / name) for name in config_paths]
    workers = configs[0].workers
    if any(c.workers != workers for c in configs):
        raise ValueError("workers mismatch")
    if args.compute_site == "node123" and workers != 3:
        raise ValueError("node123 authorization requires three CPUs per node")
    nodes = subprocess.check_output(
        ["sinfo", "-N", "-n", node_list, "-h", "-o", "%N|%P|%t|%C"], text=True
    )
    rows = [row.split("|") for row in nodes.splitlines()]
    if (
        len(rows) != len(node_names)
        or {r[0] for r in rows} != set(node_names)
        or any(
            r[1] != partition or any(b in r[2] for b in ("down", "drain", "drng", "maint", "inval"))
            for r in rows
        )
    ):
        raise RuntimeError("requested nodes unavailable")
    idle_by_node = {r[0]: int(r[3].split("/")[1]) for r in rows}
    idle = min(idle_by_node.values())
    if idle < workers and not args.allow_queue:
        raise RuntimeError(f"only {idle} idle CPUs, need {workers}; no oversubscription")
    jobs = subprocess.check_output(
        ["squeue", "-a", "-w", node_list, "-h", "-o", "%i|%j|%u|%T|%C"], text=True
    )
    verify = verification_command()
    runs = "; ".join(
        ".venv/bin/python scripts/run/run_random_q_burst.py --compute-site fat2 --config " + name
        for name in config_paths
    )
    if args.compute_site == "node123":
        runs = (
            "srun --mpi=none --nodes=3 --ntasks=3 --ntasks-per-node=1 --cpus-per-task=3 --cpu-bind=cores "
            ".venv/bin/python scripts/run/run_random_q_burst.py --compute-site node123 --slurm-shards 3 --config "
            + args.config
            + "; .venv/bin/python scripts/analysis/merge_random_q_shards.py --run-id "
            + configs[0].run_id
            + " --shard-count 3"
        )
    wrapped = "set -eu; export PYTHONPATH=.; export MPLBACKEND=Agg; " + verify + "; " + runs
    command = [
        "sbatch",
        "--parsable",
        f"--partition={partition}",
        f"--nodelist={node_list}",
        f"--nodes={len(node_names)}",
        f"--ntasks={len(node_names)}",
        "--ntasks-per-node=1",
        f"--cpus-per-task={workers}",
        "--job-name=AUR-q-z12p5",
        f"--chdir={target}",
        f"--output={target}/outputs/{args.compute_site}_%j.out",
        f"--error={target}/outputs/{args.compute_site}_%j.err",
        "--wrap",
        wrapped,
    ]
    record = dict(
        command=command,
        node_snapshot=nodes,
        jobs=jobs,
        requested_cpus=workers * len(node_names),
        cpus_per_node=workers,
        idle_cpus_by_node=idle_by_node,
        allow_queue=args.allow_queue,
        controller_sha256=digest(Path(__file__)),
        allocation_authorization="Explicit user-selected nodes and CPU count; no existing jobs changed",
        configs=config_paths,
    )
    path = ROOT / "outputs" / f"{args.release}_{'apply' if args.apply else 'dry'}.json"
    if path.exists():
        raise FileExistsError(path)
    print(shlex.join(command), flush=True)
    try:
        if args.apply:
            response = subprocess.check_output(command, text=True).strip()
            record["submission_response"] = response
            if not response.isdigit():
                raise RuntimeError("ambiguous submission response; do not auto-retry")
            record["job_id"] = response
            print("submitted", response, flush=True)
    finally:
        path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
