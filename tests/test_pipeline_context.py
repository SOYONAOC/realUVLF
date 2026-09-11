from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from auroralf.mah import Cosmology
from auroralf.mah.physics import compute_bryan_norman_virial_terms
from auroralf.seeding import derive_pipeline_random_seeds
from auroralf.sfr import compute_sfr_from_tracks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COSMOLOGY_CALLS = {
    "classify_halo_stellar_channels",
    "compute_atomic_cooling_mass_msun",
    "compute_halo_mass_function_dndm",
    "compute_regulator_metallicity",
    "compute_popiii_sfr_visbal2015_from_grids",
    "compute_popiii_duty_cycle",
    "compute_popiii_sfr_from_grids",
    "compute_popiii_upper_mass_msun",
    "compute_reed07_halo_mass_function_dndm",
    "compute_sfr_from_tracks",
    "generate_halo_histories",
    "generate_thesan_halo_histories",
    "generate_tng_halo_histories",
    "run_halo_uv_pipeline",
    "sample_uvlf_from_hmf",
    "_compute_lw_proxy",
    "_integrate_sfrd_at_z",
}


def test_bryan_norman_virial_terms_match_flat_lcdm_formula() -> None:
    cosmology = Cosmology(omega_m=0.4, omega_b=0.08, omega_lambda=0.6)
    redshift = np.array([0.0, 6.0, 10.0, 20.0], dtype=float)
    matter_term = cosmology.omega_m * (1.0 + redshift) ** 3
    expected_omega_m = matter_term / (matter_term + cosmology.omega_lambda)
    density_offset = expected_omega_m - 1.0
    expected_overdensity = (
        18.0 * np.pi**2 + 82.0 * density_offset - 39.0 * density_offset**2
    )

    expansion_squared, omega_m_at_redshift, virial_overdensity = compute_bryan_norman_virial_terms(
        redshift,
        cosmology=cosmology,
    )

    np.testing.assert_array_equal(expansion_squared, matter_term + cosmology.omega_lambda)
    np.testing.assert_array_equal(omega_m_at_redshift, expected_omega_m)
    np.testing.assert_array_equal(virial_overdensity, expected_overdensity)


def _project_python_sources(source_root: Path):
    """Inspect owned sources without traversing nested virtual environments."""
    for directory, subdirectories, files in source_root.walk():
        subdirectories[:] = sorted(
            name for name in subdirectories
            if name not in {".venv", ".git", "__pycache__"}
        )
        for name in sorted(files):
            if name.endswith(".py"):
                yield directory / name


def test_science_source_scan_excludes_nested_environment(tmp_path: Path) -> None:
    owned = tmp_path / "experiment" / "model.py"
    owned.parent.mkdir()
    owned.write_text("# owned source\n")
    dependency = owned.parent / ".venv" / "lib" / "encoded_fixture.py"
    dependency.parent.mkdir(parents=True)
    dependency.write_bytes(b"# third-party non-UTF8 fixture: \xa4\xa4\n")
    assert list(_project_python_sources(tmp_path)) == [owned]


def test_repository_science_calls_pass_explicit_cosmology() -> None:
    missing: list[str] = []
    for source_root in (PROJECT_ROOT / "auroralf", PROJECT_ROOT / "scripts"):
        for path in _project_python_sources(source_root):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    function_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    function_name = node.func.attr
                else:
                    continue
                if function_name not in REQUIRED_COSMOLOGY_CALLS:
                    continue
                if not any(keyword.arg == "cosmology" for keyword in node.keywords):
                    relative = path.relative_to(PROJECT_ROOT)
                    missing.append(f"{relative}:{node.lineno}:{function_name}")

    assert missing == [], "required cosmology keyword missing from:\n" + "\n".join(missing)


def test_cosmology_owned_apis_do_not_accept_independent_baryon_fraction() -> None:
    from auroralf.chemistry import compute_regulator_metallicity
    from auroralf.sfr import (
        compute_popiii_sfr_from_grids,
        compute_popiii_sfr_visbal2015_from_grids,
    )

    for function in (
        compute_popiii_sfr_from_grids,
        compute_popiii_sfr_visbal2015_from_grids,
        compute_regulator_metallicity,
    ):
        signature = inspect.signature(function)
        assert "cosmology" in signature.parameters
        assert signature.parameters["cosmology"].default is inspect.Parameter.empty
        assert "baryon_fraction" not in signature.parameters


def _manual_tracks() -> dict[str, np.ndarray]:
    raw_rate = np.full(3, 2.0e9, dtype=float)
    return {
        "halo_id": np.zeros(3, dtype=int),
        "step": np.arange(3, dtype=int),
        "z": np.array([8.0, 7.0, 6.0], dtype=float),
        "t_gyr": np.array([0.64, 0.76, 0.93], dtype=float),
        "Mh": np.full(3, 1.0e10, dtype=float),
        "dMh_dt_raw": raw_rate.copy(),
        "dMh_dt_sfr": raw_rate.copy(),
        "dMh_dt_clipped": np.zeros(3, dtype=bool),
        "active_flag": np.ones(3, dtype=bool),
    }


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        ({"h0": np.nan}, "h0.*finite"),
        ({"omega_m": np.inf}, "omega_m.*finite"),
        ({"omega_b": np.nan}, "omega_b.*finite"),
        ({"omega_lambda": np.inf}, "omega_lambda.*finite"),
        ({"h0": 0.0}, "h0.*positive"),
        ({"omega_m": 0.0}, "omega_m.*positive"),
        ({"omega_b": 0.0}, "omega_b.*positive"),
        ({"omega_m": 0.3, "omega_b": 0.31, "omega_lambda": 0.7}, "omega_b.*omega_m"),
        ({"omega_m": 1.1, "omega_b": 0.05, "omega_lambda": -0.1}, "omega_lambda.*non-negative"),
        ({"omega_m": 0.3, "omega_b": 0.05, "omega_lambda": 0.6}, "flat"),
    ],
)
def test_cosmology_rejects_invalid_physical_parameters(
    overrides: dict[str, float],
    error_match: str,
) -> None:
    defaults = {
        "h0": Cosmology().h0,
        "omega_m": 0.3,
        "omega_b": 0.05,
        "omega_lambda": 0.7,
    }
    defaults.update(overrides)

    with pytest.raises(ValueError, match=error_match):
        Cosmology(**defaults)


@pytest.mark.parametrize("field", ["h0", "omega_m", "omega_b", "omega_lambda"])
@pytest.mark.parametrize("value", ["0.5", True])
def test_cosmology_rejects_non_real_parameters(field: str, value: object) -> None:
    values: dict[str, object] = {
        "h0": Cosmology().h0,
        "omega_m": 0.3,
        "omega_b": 0.05,
        "omega_lambda": 0.7,
    }
    values[field] = value

    with pytest.raises(TypeError, match=f"{field}.*real"):
        Cosmology(**values)  # type: ignore[arg-type]


def test_cosmology_normalizes_numpy_reals_to_python_float() -> None:
    cosmology = Cosmology(
        h0=np.float64(Cosmology().h0),
        omega_m=np.float32(0.25),
        omega_b=np.float64(0.08),
        omega_lambda=np.float64(0.75),
    )

    assert all(
        type(getattr(cosmology, field)) is float
        for field in ("h0", "omega_m", "omega_b", "omega_lambda")
    )


def test_legacy_cosmology_set_is_not_public() -> None:
    import auroralf.mah as mah

    assert not hasattr(mah, "CosmologySet")


def test_compute_sfr_requires_keyword_only_cosmology() -> None:
    tracks = _manual_tracks()

    with pytest.raises(TypeError, match="cosmology"):
        compute_sfr_from_tracks(tracks)
    with pytest.raises(TypeError):
        compute_sfr_from_tracks(tracks, Cosmology())


def test_compute_sfr_uses_supplied_cosmology_for_virial_and_baryon_scaling() -> None:
    tracks = _manual_tracks()
    reference = Cosmology(
        h0=Cosmology().h0,
        omega_m=0.30,
        omega_b=0.03,
        omega_lambda=0.70,
    )
    alternate = Cosmology(
        h0=2.0 * Cosmology().h0,
        omega_m=0.40,
        omega_b=0.08,
        omega_lambda=0.60,
    )

    reference_result = compute_sfr_from_tracks(tracks, cosmology=reference)
    alternate_result = compute_sfr_from_tracks(tracks, cosmology=alternate)

    assert not np.allclose(reference_result["r_vir"], alternate_result["r_vir"])
    expected_sfr_ratio = (alternate.omega_b / alternate.omega_m) / (
        reference.omega_b / reference.omega_m
    )
    np.testing.assert_allclose(
        alternate_result["SFR"] / reference_result["SFR"],
        expected_sfr_ratio,
    )


def test_compute_sfr_rejects_non_cosmology_context() -> None:
    with pytest.raises(TypeError, match="cosmology.*Cosmology"):
        compute_sfr_from_tracks(_manual_tracks(), cosmology=object())


def test_delay_sfr_uses_custom_cosmology_for_all_core_quantities() -> None:
    cosmology = Cosmology(
        h0=1.2 * Cosmology().h0,
        omega_m=0.4,
        omega_b=0.08,
        omega_lambda=0.6,
    )

    result = compute_sfr_from_tracks(
        _manual_tracks(),
        cosmology=cosmology,
        enable_time_delay=True,
    )

    for name in ("r_vir", "V_c", "T_vir", "tau_del", "td_burst", "SFR"):
        assert np.all(np.isfinite(result[name])), name
    assert np.all(result["SFR"] >= 0.0)
    reference = compute_sfr_from_tracks(
        _manual_tracks(),
        cosmology=Cosmology(),
        enable_time_delay=True,
    )
    assert not np.allclose(result["tau_del"], reference["tau_del"])
    assert not np.allclose(result["td_burst"], reference["td_burst"])


def test_mcbride_default_mass_floor_uses_supplied_cosmology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.mah.generator as generator
    import auroralf.mah.physics as halo_physics

    cosmology = Cosmology(
        h0=2.0 * Cosmology().h0,
        omega_m=0.4,
        omega_b=0.08,
        omega_lambda=0.6,
    )
    contexts: list[Cosmology] = []

    def atomic_floor(z_obs, *, cosmology: Cosmology, **kwargs):
        contexts.append(cosmology)
        return np.full(np.asarray(z_obs).shape, 1.0, dtype=float)

    monkeypatch.setattr(
        halo_physics,
        "compute_atomic_cooling_mass_msun",
        atomic_floor,
    )
    generator.generate_halo_histories(
        n_tracks=1,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=cosmology,
        z_start_max=7.0,
        dz=1.0,
        M_min=None,
    )

    assert contexts == [cosmology]
    assert contexts[0] is cosmology


def test_compute_sfr_rejects_nonfinite_core_physical_quantities() -> None:
    tracks = _manual_tracks()
    tracks["Mh"][1] = np.inf

    with pytest.raises(RuntimeError, match="core physical quantities.*finite"):
        compute_sfr_from_tracks(tracks, cosmology=Cosmology())


def test_generate_halo_histories_requires_keyword_only_cosmology() -> None:
    from auroralf.mah import generate_halo_histories

    with pytest.raises(TypeError, match="cosmology"):
        generate_halo_histories(n_tracks=1, z_final=6.0, Mh_final=1.0e10)


def test_pipeline_passes_one_cosmology_instance_to_mah_and_sfr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.uvlf.pipeline as pipeline
    from auroralf.chemistry import RegulatorMetallicityParameters
    from auroralf.sfr import PopIIISFRParameters

    cosmology = Cosmology(
        h0=1.1 * Cosmology().h0,
        omega_m=0.32,
        omega_b=0.052,
        omega_lambda=0.68,
    )
    mah_contexts: list[Cosmology] = []
    sfr_contexts: list[Cosmology] = []
    popiii_contexts: list[Cosmology] = []
    regulator_contexts: list[Cosmology] = []
    ssp_age_calls: list[np.ndarray] = []
    real_generate_halo_histories = pipeline.generate_halo_histories
    real_compute_sfr_from_tracks = pipeline.compute_sfr_from_tracks
    real_compute_popiii_sfr_from_grids = pipeline.compute_popiii_sfr_from_grids
    real_compute_regulator_metallicity = pipeline.compute_regulator_metallicity
    real_compute_final_ssp_observable = pipeline.compute_final_ssp_observable_from_sfr_grid
    real_prepare_shared_halo_batch = pipeline.prepare_shared_halo_batch
    captured_shared_batches = []

    def generate_spy(*args, **kwargs):
        mah_contexts.append(kwargs["cosmology"])
        return real_generate_halo_histories(*args, **kwargs)

    def sfr_spy(*args, **kwargs):
        sfr_contexts.append(kwargs["cosmology"])
        return real_compute_sfr_from_tracks(*args, **kwargs)

    def popiii_spy(*args, **kwargs):
        popiii_contexts.append(kwargs["cosmology"])
        return real_compute_popiii_sfr_from_grids(*args, **kwargs)

    def regulator_spy(*args, **kwargs):
        regulator_contexts.append(kwargs["cosmology"])
        return real_compute_regulator_metallicity(*args, **kwargs)

    def ssp_convolution_spy(*args, **kwargs):
        ssp_age_calls.append(np.asarray(kwargs["ssp_age_myr"], dtype=float).copy())
        return real_compute_final_ssp_observable(*args, **kwargs)

    def prepare_shared_spy(*args, **kwargs):
        shared = real_prepare_shared_halo_batch(*args, **kwargs)
        captured_shared_batches.append(shared)
        return shared

    monkeypatch.setattr(pipeline, "generate_halo_histories", generate_spy)
    monkeypatch.setattr(pipeline, "compute_sfr_from_tracks", sfr_spy)
    monkeypatch.setattr(pipeline, "compute_popiii_sfr_from_grids", popiii_spy)
    monkeypatch.setattr(pipeline, "compute_regulator_metallicity", regulator_spy)
    monkeypatch.setattr(
        pipeline,
        "compute_final_ssp_observable_from_sfr_grid",
        ssp_convolution_spy,
    )
    monkeypatch.setattr(pipeline, "prepare_shared_halo_batch", prepare_shared_spy)
    monkeypatch.setattr(
        pipeline,
        "load_uv1600_table",
        lambda file_path, **kwargs: (
            np.array([1.0e-2, 100.0], dtype=float),
            np.array([1.0, 1.0], dtype=float),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "load_popiii_uv_luminosity_table",
        lambda file_path: (
            np.array([2.0e-2, 80.0], dtype=float),
            np.array([1.0, 1.0], dtype=float),
        ),
    )

    result = pipeline.run_halo_uv_pipeline(
        n_tracks=1,
        z_final=6.0,
        Mh_final=1.0e10,
        z_start_max=8.0,
        n_grid=4,
        ssp_file="dummy.dat",
        cosmology=cosmology,
        random_seeds=derive_pipeline_random_seeds(7, redshift=6.0, mass_index=0),
        workers=1,
        enable_popiii=True,
        popiii_ssp_file="popiii.dat",
        popiii_sfr_parameters=PopIIISFRParameters(
            upper_mass_mode="fixed",
            upper_mass_msun=1.0e12,
        ),
        regulator_metallicity_parameters=RegulatorMetallicityParameters(
            gas_fraction_norm=0.2,
            metallicity_scatter_dex=0.0,
        ),
    )

    assert mah_contexts == [cosmology]
    assert sfr_contexts == [cosmology]
    assert mah_contexts[0] is cosmology
    assert sfr_contexts[0] is cosmology
    assert popiii_contexts[0] is cosmology
    assert regulator_contexts[0] is cosmology
    assert any(np.array_equal(ages, np.array([1.0e-2, 100.0])) for ages in ssp_age_calls)
    assert any(np.array_equal(ages, np.array([2.0e-2, 80.0])) for ages in ssp_age_calls)
    assert (
        result.metadata["uv_convolution_method"]
        == "shared_prepared_batch_final_ssp_observable_v2"
    )
    assert len(captured_shared_batches) == 1
    assert type(result.sfr_tracks) is dict
    assert type(result.histories.tracks) is dict
    for public_array in (
        result.sfr_tracks["SFR"],
        result.histories.tracks["Mh"],
        result.active_grid,
    ):
        assert public_array.flags.writeable is True
    assert result.uv_luminosities.flags.writeable is True
    result.sfr_tracks["SFR"][0] = result.sfr_tracks["SFR"][0]
    result.histories.tracks["Mh"][0] = result.histories.tracks["Mh"][0]


def test_pipeline_rejects_non_cosmology_context() -> None:
    from auroralf.uvlf.pipeline import run_halo_uv_pipeline

    with pytest.raises(TypeError, match="cosmology.*Cosmology"):
        run_halo_uv_pipeline(
            n_tracks=1,
            z_final=6.0,
            Mh_final=1.0e10,
            cosmology=object(),
            random_seeds=derive_pipeline_random_seeds(7, redshift=6.0, mass_index=0),
        )


def test_pipeline_requires_cosmology() -> None:
    from auroralf.uvlf.pipeline import run_halo_uv_pipeline

    with pytest.raises(TypeError, match="cosmology"):
        run_halo_uv_pipeline(
            n_tracks=1,
            z_final=6.0,
            Mh_final=1.0e10,
            random_seeds=derive_pipeline_random_seeds(7, redshift=6.0, mass_index=0),
        )
