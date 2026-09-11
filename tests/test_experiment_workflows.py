"""Portable configurations and integrity checks for opt-in experiment workflows."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from auroralf.experiments.artifacts import digest, read_completed_manifest
from auroralf.experiments.deployment import freeze_release, verify_release
from auroralf.experiments.visbal import VisbalCombineConfig, VisbalDutyConfig, combine_window

ROOT = Path(__file__).resolve().parents[1]


def test_visbal_paths_follow_toml_location_after_relocation(tmp_path, monkeypatch):
    directory = tmp_path / "relocated" / "configs" / "experiments"
    directory.mkdir(parents=True)
    for name in ("visbal_duty.toml", "visbal_combine.toml", "visbal_combine_wide.toml"):
        (directory / name).write_bytes((ROOT / "configs" / "experiments" / name).read_bytes())
    monkeypatch.chdir(tmp_path)
    duty = VisbalDutyConfig.load(directory / "visbal_duty.toml")
    assert duty.source_root == tmp_path / "relocated"
    assert duty.output == tmp_path / "relocated/data_save/AUR-EX-0006-R001"
    for name, bounds in (("visbal_combine.toml", (1, 2)), ("visbal_combine_wide.toml", (0.1, 100))):
        config = VisbalCombineConfig.load(directory / name)
        assert (config.mass_min_ratio, config.mass_max_ratio) == bounds
        assert all(p.is_relative_to(tmp_path / "relocated") for p in config.runs)


def test_combination_supports_explicit_batch_count(tmp_path):
    text = (ROOT / "configs/experiments/visbal_combine.toml").read_text()
    text = text.replace('  "../../data_save/visbal-resume-20260908-01/AUR-EX-0006-R013",\n', "")
    path = tmp_path / "combine.toml"
    path.write_text(text)
    config = VisbalCombineConfig.load(path)
    assert len(config.runs) == 3
    result, error = combine_window(np.array([10.0]), np.array([[1.0], [2.0], [3.0]]))
    np.testing.assert_allclose(result, [12.0])
    np.testing.assert_allclose(error, [1 / np.sqrt(3)])
    path.write_text(text.replace("R012", "R011"))
    with pytest.raises(ValueError, match="distinct"):
        VisbalCombineConfig.load(path)


def test_manifest_requires_consumed_product_and_rejects_corruption(tmp_path):
    product = tmp_path / "samples.npz"
    product.write_bytes(b"explicit integrity fixture")
    manifest = {"status": "complete", "products": {"samples.npz": digest(product)}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    read_completed_manifest(tmp_path, required_products=("samples.npz",))
    with pytest.raises(ValueError, match="unverified product"):
        read_completed_manifest(tmp_path, required_products=("uvlf.npz",))
    product.write_bytes(b"corrupt fixture")
    with pytest.raises(ValueError, match="SHA-256"):
        read_completed_manifest(tmp_path, required_products=("samples.npz",))


def test_integrity_checks_survive_python_optimization(tmp_path):
    (tmp_path / "samples.npz").write_bytes(b"corrupt")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "products": {"samples.npz": "0" * 64},
            }
        )
    )
    code = (
        "import sys; from pathlib import Path; "
        "from auroralf.experiments.artifacts import read_completed_manifest; "
        "read_completed_manifest(Path(sys.argv[1]), required_products=('samples.npz',))"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr


def test_release_freezes_exact_declared_bytes_and_detects_changes(tmp_path):
    root, target = tmp_path / "repo", tmp_path / "release"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    source = root / "input.dat"
    source.write_bytes(b"declared source bytes")
    manifest = freeze_release(root, target, ["input.dat", "input.dat"])
    assert manifest["files"] == {"input.dat": digest(source)}
    verify_release(target)
    source.write_bytes(b"later working copy")
    assert (target / "input.dat").read_bytes() == b"declared source bytes"
    (target / "input.dat").write_bytes(b"tampered snapshot")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_release(target)
    with pytest.raises(FileNotFoundError):
        freeze_release(root, tmp_path / "missing-release", ["missing.dat"])
    assert not (tmp_path / "missing-release").exists()
