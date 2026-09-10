import pytest
from scripts.run.run_random_q_burst import check_compute_site
from scripts.submit.submit_random_q_fat2 import CONFIGS


def test_explicit_authorized_fat2():
    env = {'SLURM_JOB_PARTITION':'fat','SLURM_JOB_NODELIST':'fat2'}
    check_compute_site('fat2',env)
    with pytest.raises(RuntimeError):
        check_compute_site('cp6',env)
    with pytest.raises(RuntimeError):
        check_compute_site('fat2',{**env,'SLURM_JOB_NODELIST':'fat1'})


@pytest.mark.parametrize('site', ['cp6','fat2'])
def test_debug_never_allowed(site):
    with pytest.raises(RuntimeError):
        check_compute_site(site, {'SLURM_JOB_PARTITION':'debug6','SLURM_JOB_NODELIST':'fat2'})


def test_only_one_science_config():
    assert CONFIGS == ['configs/experiments/random_q_R028.toml']
