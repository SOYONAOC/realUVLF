"""Freeze and submit the registered random-q campaign, without time/memory requests."""
from pathlib import Path
import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import re

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/fs2/home/xuelei/zhuhr/AuroraLF/releases/'


def ssh(cmd):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'sc', cmd], text=True)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--apply', action='store_true')
    p.add_argument('--configs', nargs='+')
    a = p.parse_args()
    if not re.fullmatch('[a-z0-9-]+', a.release):
        raise ValueError('release name')
    local = ROOT/'outputs/deployments'/a.release
    remote = REMOTE+a.release
    configs = a.configs or ['configs/experiments/random_q_R032.toml']
    if any(not re.fullmatch(r'configs/experiments/random_q_R\d{3}\.toml', c) for c in configs):
        raise ValueError('invalid config path')
    if not 1 <= len(configs) <= 4:
        raise ValueError('submission accepts one to four explicit science configs; no separate preflight')
    if a.prepare:
        from scripts.experiments.random_q_burst import Config
        for name in configs:
            Config.load(ROOT/name)
        local.mkdir(parents=True, exist_ok=False)
        files = [str(f.relative_to(ROOT)) for f in (ROOT/'auroralf').rglob('*.py')]
        files += configs + ['pyproject.toml', 'uv.lock', 'scripts/experiments/random_q_burst.py',
            'scripts/run/run_random_q_burst.py', 'scripts/submit/submit_random_q.py',
            'tests/test_random_q_burst.py',
            'external_data/ssp_spectra/bpass_byrne23_imf135_300/BASEL/spectra-bin-imf135_300.BASEL.z001.a+00.dat',
            'external_data/ssp_spectra/schaerer2010_pop3/pop3_ge0_logE_500_001_is5.25']
        manifest = dict(source=str(ROOT), revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            dirty=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True), files={})
        for name in files:
            dst = local/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, dst)
            if sha(dst) != sha(ROOT/name):
                raise RuntimeError('snapshot mismatch')
            manifest['files'][name] = sha(dst)
        (local/'deployment.json').write_text(json.dumps(manifest, indent=2)+'\n')
        ssh('mkdir '+shlex.quote(remote))
        subprocess.run(['rsync', '-a', str(local)+'/', 'sc:'+remote+'/'], check=True)
        ssh('ln -s /fs2/home/xuelei/zhuhr/AuroraLF/releases/atomic-memory-20260907-01/.venv '+shlex.quote(remote+'/.venv'))
        ssh('mkdir '+shlex.quote(remote+'/outputs'))
        print('prepared '+remote, flush=True)
        return
    if not (local/'deployment.json').is_file():
        raise FileNotFoundError(local)
    nodes = ssh('sinfo -N -p cp6 -h -o "%N|%t|%c|%C"')
    jobs = ssh('squeue -r -u xuelei -h -o "%i|%j|%T|%D"')
    usable = [row.split('|') for row in nodes.splitlines() if not any(s in row.split('|')[1] for s in ('down','drain','drng','maint','inval'))]
    if not usable or any(int(row[2]) != 56 for row in usable):
        raise RuntimeError('unexpected cp6 CPUs')
    # Conservative all-user allocated/requested nodes; never touch another job.
    reserved = sum(int(row.split('|')[3]) for row in jobs.splitlines())
    if reserved+len(configs) > 10:
        raise RuntimeError('ten node ceiling')
    base = ['sbatch', '--parsable', '--partition=cp6', '--nodes=1', '--ntasks=1', '--cpus-per-task=56', '--exclusive', '--chdir='+remote]
    verify = '.venv/bin/python -c '+shlex.quote("import json,hashlib,pathlib; d=json.load(open('deployment.json')); assert all(hashlib.sha256(pathlib.Path(k).read_bytes()).hexdigest()==v for k,v in d['files'].items())")
    cases = ' '.join(f'{i}) cfg={name} ;;' for i,name in enumerate(configs))
    selection = ('cfg='+configs[0]) if len(configs)==1 else ('case "$SLURM_ARRAY_TASK_ID" in '+cases+' *) exit 64 ;; esac')
    array = [] if len(configs)==1 else [f'--array=0-{len(configs)-1}%{len(configs)}']
    batch = base + array + ['--job-name=AUR-random-q', '--output='+remote+'/outputs/batch_%j.out', '--error='+remote+'/outputs/batch_%j.err',
        '--wrap', 'set -eu; export PYTHONPATH=.; export MPLBACKEND=Agg; '+verify+'; '+selection+'; exec .venv/bin/python scripts/run/run_random_q_burst.py --config "$cfg"']
    record = dict(remote=remote, nodes=nodes, jobs=jobs, batch=batch)
    mode = 'apply' if a.apply else 'dry'
    out = ROOT/'outputs'/f'{a.release}_{mode}.json'
    if out.exists():
        raise FileExistsError(out)
    print(shlex.join(batch), flush=True)
    try:
        if a.apply:
            jid = ssh(shlex.join(batch)).strip()
            record['submission_response'] = jid
            if not jid.isdigit():
                raise RuntimeError('ambiguous job response: '+jid)
            record['batch_job_id'] = jid
            print('submitted', record['batch_job_id'], flush=True)
    finally:
        out.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
