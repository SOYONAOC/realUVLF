import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.run.run_random_q_burst import shard_indices, check_compute_site
from scripts.analysis.merge_random_q_shards import merge, digest
from scripts.submit import submit_random_q_fat2 as submit


@pytest.mark.parametrize('n,count',[(3600,3),(8,3),(7,1)])
def test_disjoint_complete_global_indices(n,count):
    parts=[shard_indices(n,i,count) for i in range(count)]
    np.testing.assert_array_equal(np.concatenate(parts),np.arange(n))
    assert max(map(len,parts))-min(map(len,parts))<=1


def test_node123_guard():
    for node in ('node1','node2','node3'):
        check_compute_site('node123',dict(SLURM_JOB_PARTITION='cpu',SLURM_JOB_NODELIST='node[1-3]',SLURMD_NODENAME=node))
    with pytest.raises(RuntimeError):
        check_compute_site('node123',dict(SLURM_JOB_PARTITION='debug6',SLURM_JOB_NODELIST='node[1-3]',SLURMD_NODENAME='node1'))


def fixture_shards(root):
    # Explicit synthetic integrity fixtures, never used as science input.
    paths=[]
    for i in range(3):
        p=root/f'shard{i}'; p.mkdir()
        idx=shard_indices(8,i,3)
        np.savez_compressed(p/'samples.npz',global_mass_index=idx,
            popii=idx[:,None]*np.ones((1,2)),weight_per_track=(idx+1)/8.)
        (p/'source.tar.gz').write_bytes(b'explicit test fixture')
        m=dict(status='complete',config=dict(n_mass=8),code_sha256={},input_sha256={},
            sharding=dict(index=i,count=3,global_n_mass=8),
            products={f.name:digest(f) for f in p.iterdir()})
        (p/'manifest.json').write_text(json.dumps(m)); paths.append(p)
    return paths


def test_merge_reorders_without_renormalizing(tmp_path):
    paths=fixture_shards(tmp_path)
    out=tmp_path/'joined'
    merge(paths[::-1],out)
    with np.load(out/'samples.npz') as d:
        np.testing.assert_array_equal(d['popii'][:,0],np.arange(8))
        np.testing.assert_array_equal(d['weight_per_track'],np.arange(1,9)/8.)
    m=json.loads((out/'manifest.json').read_text())
    assert 'sharding' not in m and len(m['shards'])==3
    assert all(digest(out/k)==v for k,v in m['products'].items())


def test_merge_rejects_missing_and_corrupt(tmp_path):
    paths=fixture_shards(tmp_path)
    with pytest.raises(ValueError,match='missing or duplicate'):
        merge(paths[:2],tmp_path/'joined')
    (paths[0]/'samples.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='corrupt'):
        merge(paths,tmp_path/'joined')
    assert not (tmp_path/'joined').exists()


@pytest.mark.parametrize('queue', [False,True])
def test_node_submission_is_three_by_three_no_preflight(tmp_path,monkeypatch,queue):
    name='test-three-node'
    target=tmp_path/'outputs/deployments'/name
    target.mkdir(parents=True)
    (target/'deployment.json').write_text('{"files":{}}')
    monkeypatch.setattr(submit,'ROOT',tmp_path)
    monkeypatch.setattr(submit.Config,'load',lambda p:SimpleNamespace(workers=3,run_id='AUR-EX-0006-R031'))
    def fake_read(command,**kwargs):
        if command[0]=='sinfo':
            return '\n'.join(f'node{i}|cpu|mix|34/2/0/36' if queue else f'node{i}|cpu|mix|33/3/0/36' for i in range(1,4))
        assert command[0]=='squeue'
        return ''
    monkeypatch.setattr(submit.subprocess,'check_output',fake_read)
    monkeypatch.setattr(sys,'argv',['submit','--release',name,'--compute-site','node123','--config','configs/experiments/random_q_R031.toml']+(['--allow-queue'] if queue else []))
    submit.main()
    c=json.loads((tmp_path/'outputs'/f'{name}_dry.json').read_text())['command']
    for arg in ('--nodes=3','--ntasks=3','--ntasks-per-node=1','--cpus-per-task=3','--nodelist=node1,node2,node3','--partition=cpu'):
        assert arg in c
    assert not any(arg.startswith(('--time','--mem','--dependency','--exclusive')) for arg in c)
    assert '--slurm-shards 3' in c[-1] and 'merge_random_q_shards.py' in c[-1]
