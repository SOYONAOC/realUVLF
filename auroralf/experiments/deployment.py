"""Freeze and verify experiment releases without performing scheduler actions."""

import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .artifacts import digest, read_toml, require, verify_files


@dataclass(frozen=True)
class RemoteSite:
    host: str
    user: str
    release_root: str
    python_environment: str

    @classmethod
    def load(cls, path: Path) -> "RemoteSite":
        _, data = read_toml(path)
        site = cls(**data)
        require(all(isinstance(v, str) and v for v in data.values()), "invalid remote site")
        require(
            not site.host.startswith("-") and not any(c.isspace() for c in site.host),
            "invalid SSH host",
        )
        for value in (site.release_root, site.python_environment):
            require(PurePosixPath(value).is_absolute(), "remote paths must be absolute")
        return site


def random_q_release_files(root: Path, configs: list[str], entrypoints: list[str]) -> list[str]:
    """Derive SSP inputs from the same TOML files the science runner will read."""
    from .random_q import Config

    files = [str(p.relative_to(root)) for p in (root / "auroralf").rglob("*.py")]
    files += configs + ["pyproject.toml", "uv.lock"] + entrypoints
    for name in configs:
        config = Config.load(root / name)
        for value in (config.popii_ssp, config.popiii_ssp):
            try:
                relative = Path(value).relative_to(root)
            except ValueError as error:
                raise ValueError(f"release SSP must be inside repository: {value}") from error
            files.append(str(relative))
    return sorted(set(files))


def freeze_release(root: Path, target: Path, files: list[str]) -> dict:
    """Copy declared sources and inputs, recording the exact bytes and git state."""
    paths = sorted(set(files))
    for name in paths:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"release paths must be relative to the repository: {name}")
        if not (root / path).is_file():
            raise FileNotFoundError(root / path)
    target.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source": str(root),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "dirty": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "files": {},
    }
    for name in paths:
        source, dest = root / name, target / name
        expected = digest(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        if digest(dest) != expected or digest(source) != expected:
            raise RuntimeError(f"source changed during snapshot: {source}")
        manifest["files"][name] = expected
    (target / "deployment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_release(root: Path) -> None:
    manifest = json.loads((root / "deployment.json").read_text())
    verify_files(root, manifest["files"])


def verification_command() -> str:
    """Shell-safe startup check, with explicit exceptions instead of assert."""
    return ".venv/bin/python -c " + shlex.quote(
        "from pathlib import Path; "
        "from auroralf.experiments.deployment import verify_release; "
        "verify_release(Path('deployment.json').parent)"
    )
