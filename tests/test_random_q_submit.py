import json
import sys

import pytest

from scripts.submit import submit_random_q as submit


@pytest.mark.parametrize('apply', [False, True])
@pytest.mark.parametrize('count', [1,4])
def test_science_array_without_preflight(tmp_path, monkeypatch, apply,count):
    release = 'test-science-only'
    snapshot = tmp_path/'outputs/deployments'/release
    snapshot.mkdir(parents=True)
    (snapshot/'deployment.json').write_text('{}')
    monkeypatch.setattr(submit, 'ROOT', tmp_path)
    writes = []

    def fake_ssh(command):
        if command.startswith('sinfo '):
            return 'fixture-node|idle|56|0/56/0/56\n'
        if command.startswith('squeue '):
            return '1|unrelated|RUNNING|4\n'
        assert command.startswith('sbatch ')
        writes.append(command)
        return '123456\n'

    monkeypatch.setattr(submit, 'ssh', fake_ssh)
    configs=[f'configs/experiments/random_q_R{i:03d}.toml' for i in range(24,28)] if count==4 else ['configs/experiments/random_q_R032.toml']
    monkeypatch.setattr(sys, 'argv', ['submit', '--release', release] + (['--configs']+configs if count==4 else []) + (['--apply'] if apply else []))
    submit.main()
    record = json.loads((tmp_path/'outputs'/f'{release}_{"apply" if apply else "dry"}.json').read_text())
    command = record['batch']
    assert '--partition=cp6' in command
    if count==4:
        assert '--array=0-3%4' in command
    else:
        assert not any(x.startswith('--array') for x in command)
    assert not any(x.startswith(('--dependency', '--time', '--mem')) for x in command)
    assert 'preflight' not in record
    for config in configs:
        assert config in command[-1]
    assert 'deployment.json' in command[-1]
    assert len(writes) == int(apply)
    if apply:
        assert record['batch_job_id'] == '123456'
