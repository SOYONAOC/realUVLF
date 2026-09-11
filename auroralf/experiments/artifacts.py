"""Explicit path resolution and integrity checks shared by experiment workflows."""

import hashlib
import json
import tomllib
from pathlib import Path


def resolve_path(base: Path, value: str | Path) -> Path:
    """Resolve a declared path against its configuration directory."""
    return (base / Path(value).expanduser()).resolve()


def read_toml(path: str | Path) -> tuple[Path, dict]:
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        return path, tomllib.load(stream)


def digest(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record_experiment_sources(hashes: dict[str, str]) -> None:
    """Include extracted helpers in analysis provenance, alongside the CLI source."""
    for path in sorted(Path(__file__).parent.glob("*.py")):
        hashes[str(path)] = digest(path)


def require(condition: bool, message: str) -> None:
    """Validate external inputs even when Python runs with optimization enabled."""
    if not condition:
        raise ValueError(message)


def verify_files(root: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        path = root / name
        if digest(path) != expected:
            raise ValueError(f"corrupt file (SHA-256 mismatch): {path}")


def read_completed_manifest(path: Path, *, required_products: tuple[str, ...]) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    require(manifest["status"] == "complete", f"incomplete run: {path}")
    products = manifest["products"]
    require(isinstance(products, dict) and bool(products), f"missing products: {path}")
    for name in required_products:
        require(name in products, f"unverified product {name}: {path}")
    verify_files(path, products)
    return manifest
