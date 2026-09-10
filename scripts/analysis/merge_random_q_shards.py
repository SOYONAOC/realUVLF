"""Verify and join disjoint global-index shards of ONE random-q realization."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def merge(paths, target):
    manifests = [json.loads((p/'manifest.json').read_text()) for p in paths]
    ref = manifests[0]
    count = ref['sharding']['count']
    if len(paths) != count or sorted(m['sharding']['index'] for m in manifests)!=list(range(count)):
        raise ValueError('missing or duplicate shard')
    if target.exists() or target.with_name('.'+target.name+'.partial').exists():
        raise FileExistsError(target)
    data = []
    for path,m in zip(paths,manifests):
        if m['status']!='complete' or m['sharding']['count']!=count or m['sharding']['global_n_mass']!=ref['config']['n_mass']:
            raise ValueError('incomplete or incompatible shard')
        for key in ('config','code_sha256','input_sha256'):
            if m[key]!=ref[key]:
                raise ValueError('incompatible shard '+key)
        for name,h in m['products'].items():
            if digest(path/name)!=h:
                raise ValueError('corrupt shard '+name)
        with np.load(path/'samples.npz') as values:
            data.append(dict(values))
    indices = np.concatenate([d['global_mass_index'] for d in data])
    if not np.array_equal(np.sort(indices),np.arange(ref['config']['n_mass'])):
        raise ValueError('global coverage has gaps or duplicates')
    order = np.argsort(indices)
    fields = set(data[0])
    if any(set(d)!=fields for d in data):
        raise ValueError('inconsistent shard fields')
    for d in data:
        n = len(d['global_mass_index'])
        if any(v.shape[0]!=n for v in d.values()):
            raise ValueError('inconsistent shard shapes')
    joined = {k:np.concatenate([d[k] for d in data],axis=0)[order]
              for k in fields if k!='global_mass_index'}
    stage = target.with_name('.'+target.name+'.partial')
    stage.mkdir(parents=True)
    np.savez_compressed(stage/'samples.npz',**joined)
    shutil.copyfile(paths[0]/'source.tar.gz', stage/'source.tar.gz')
    result = dict(ref)
    result.pop('sharding')
    result['shards'] = {str(p/'manifest.json'):digest(p/'manifest.json') for p in paths}
    result['merge_policy'] = 'Join unique global indices with original global HMF weights; no extra normalization'
    result['merge_code_sha256'] = digest(Path(__file__))
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    for i,(p,m) in enumerate(zip(paths,manifests)):
        shutil.copyfile(p/'manifest.json',stage/f'shard_manifest_{i}.json')
    result['products'] = {p.name:digest(p) for p in stage.iterdir()}
    (stage/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    stage.rename(target)
    print('merged_artifacts='+str(target),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--shard-count',type=int,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]/'data_save'
    merge([root/f'{args.run_id}-shard{i}-of{args.shard_count}' for i in range(args.shard_count)],root/args.run_id)
