"""Combine four independent window estimates with a disjoint original high tail."""
from pathlib import Path
import json
import hashlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scripts.analysis.visbal_duty_uvlf import active_weights, mag

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/home/zhuhourui/AstroCode/AuroraLF')


def combine_window(high, batches):
    batches=np.asarray(batches)
    if len(batches)<2: raise ValueError('need independent batches')
    return high+batches.mean(axis=0),batches.std(axis=0,ddof=1)/np.sqrt(len(batches))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wide',action='store_true',help='Use explicit 0.1–100 Mcool experiment R014–R017')
    args=parser.parse_args()
    lower,upper=(.1,100.) if args.wide else (1.,2.)
    run_start=14 if args.wide else 10
    release='visbal-wide-20260908-01' if args.wide else 'visbal-resume-20260908-01'
    out=ROOT/('data_save/visbal_wide_20260908' if args.wide else 'data_save/visbal_combined_20260908')
    if out.exists(): raise FileExistsError(out)
    hashes={}
    def record(p):
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    record(Path(__file__))
    priorpath=ROOT/'data_save/AUR-EX-0006-R001/summary.json';record(priorpath)
    prior=json.loads(priorpath.read_text());cfg=prior['config']
    floor=prior['mcool_msun'];coefficient=prior['uv_per_sfr'];th=prior['hubble_time_myr']*1e6
    from auroralf.constants import PLANCK18_OMEGA_B,PLANCK18_OMEGA_M
    fb=PLANCK18_OMEGA_B/PLANCK18_OMEGA_M
    ssp=SOURCE/'external_data/ssp_spectra/schaerer2010_pop3/pop3_ge0_logE_500_001_is5.25'
    record(ssp);assert hashes[str(ssp)]==prior['inputs_sha256'][str(ssp)]
    oldpath=ROOT/'data_save/AUR-EX-0006-R001/uvlf.npz';record(oldpath)
    old=np.load(oldpath);edges=old['bin_edges'];widths=np.diff(edges);x=(edges[1:]+edges[:-1])/2
    def hist(light,w):return np.histogram(mag(light),edges,weights=w)[0]/widths
    def read(path):
        record(path/'manifest.json');m=json.loads((path/'manifest.json').read_text())
        assert m['status']=='complete'
        for name,digest in m['products'].items():
            record(path/name);assert hashes[str(path/name)]==digest
        with np.load(path/'samples_z14.5.npz') as d:
            s={k:d[k] for k in ('popii','weight','halo_mass_msun')}
        assert all(np.all(np.isfinite(v)) for v in s.values())
        assert np.all(s['weight']>0) and np.all(s['popii']>=0)
        assert len(s['weight'])==m['config']['N_mass']*m['config']['n_tracks']
        return m,s
    fm,full=read(SOURCE/'data_save/supercomputer_tracks1000_20260907/AUR-EX-0001-R021')
    w=full['weight']*(full['halo_mass_msun']>upper*floor)
    high=hist(full['popii'],w)
    high20=float(np.sum(w*(mag(full['popii'])<=-20)))
    del full
    batches={str(d):[] for d in cfg['duties']};bases=[];rows=[];seeds=[]
    for i in range(run_start,run_start+4):
        run=f'AUR-EX-0006-R{i:03d}'
        m,s=read(ROOT/'data_save'/release/run)
        c=m['config'];seeds.append(c['base_seed'])
        assert c['N_mass']==3600 and c['n_tracks']==1000 and c['n_grid']==960
        np.testing.assert_allclose(10**c['logM_max'],upper*floor,rtol=1e-12)
        assert c.get('mass_min_ratio',1.)==lower
        p2=s['popii'];mass=s['halo_mass_msun'];w=s['weight']
        assert np.all((mass>=lower*floor)&(mass<=upper*floor))
        base=hist(p2,w);bases.append(base)
        info=dict(run=run,seed=c['base_seed'],window_density=float(w.sum()),duties=[])
        for duty in cfg['duties']:
            wa,wi=active_weights(w,np.ones(len(w),dtype=bool),duty)
            l3=cfg['fstar']*fb*mass/(duty*th)*coefficient
            curve=hist(p2,wi)+hist(p2+l3,wa);batches[str(duty)].append(curve)
            info['duties'].append(dict(duty=duty,brightest_popiii=float(np.min(mag(l3))),
                emissivity=float(np.sum(wa*l3)),density_le20=high20+float(np.sum(wi*(mag(p2)<=-20)+wa*(mag(p2+l3)<=-20)))))
        np.testing.assert_allclose([v['emissivity'] for v in info['duties']],info['duties'][0]['emissivity'],rtol=1e-12)
        rows.append(info);print(run+' verified',flush=True)
    assert len(set(seeds))==4
    baseline,base_se=combine_window(high,bases)
    curves={};errors={};diagnostics=[]
    for duty in cfg['duties']:
        key=str(duty); curves[key],errors[key]=combine_window(high,batches[key])
        enhancement=curves[key]-baseline
        peak=int(np.argmax(enhancement))
        priorcurve=old[f'duty{duty}']
        diagnostics.append(dict(duty=duty,peak_bin_muv=float(x[peak]),
            peak_phi=float(curves[key][peak]),batch_se_at_peak=float(errors[key][peak]),
            previous_phi_at_peak=float(priorcurve[peak]),
            change_at_peak_percent=float(100*(curves[key][peak]/priorcurve[peak]-1))))
    plt.style.use('apj');fig,ax=plt.subplots(figsize=(9,6.7))
    fig.subplots_adjust(left=.13,right=.97,bottom=.28,top=.87)
    ax.plot(x,baseline,color='black',lw=2,label='Pop II only')
    ax.plot(x,old['burst10'],color='.5',ls=':',lw=2,label=r'Previous first-crossing burst (10\%)')
    for duty,color in zip(cfg['duties'],['#a44b20','#008c85','#3973ac']):
        key=str(duty);y=curves[key];err=errors[key]
        ax.plot(x,np.where(y>0,y,np.nan),color=color,lw=2,label=rf'Visbal-style: duty = {100*duty:g}\%')
        ax.fill_between(x,y-err,y+err,where=y>err,color=color,alpha=.2,linewidth=0)
    modelhandles,_=ax.get_legend_handles_labels();observations=[]
    for rel,label,marker,color in [
        ('redshift_14/whitler25_jades_z14p3.npz',r'Whitler+25: JADES, $z\geq14$ (median 14.3)','o','#333333'),
        ('redshift_15/donnan24_primer_z14p5.npz',r'Donnan+24: PRIMER, $13.5<z<15.5$ (tentative)','s','#945399'),
        ('redshift_15/naidu26_mom_jades_spectroscopic_z14p5.npz',r'Naidu+26: MoM+JADES spec., $14<z<15$','D','#207798')]:
        path=SOURCE/'external_data/observations/uvlf'/rel;record(path);d=np.load(path)
        assert not np.any(d['is_upper_limit'])
        observations.append(ax.errorbar(d['muverr'],d['phierr'],xerr=d['mag_err'],
            yerr=[d['phi_err_lo'],d['phi_err_up']],fmt=marker,color=color,mfc='white',ms=7,capsize=3,label=label,zorder=10))
    ax.set(xlim=(-24,-12) if args.wide else (-22,-12),ylim=(3e-9,3.) if args.wide else (3e-8,.2),yscale='log',xlabel=r'$M_{\rm UV}$ [AB mag]',ylabel=r'$\phi$ [Mpc$^{-3}$ mag$^{-1}$]')
    ax.legend(handles=modelhandles,loc='upper left',fontsize=10,frameon=False)
    fig.legend(handles=observations,loc='lower left',bbox_to_anchor=(.12,.06),fontsize=10,frameon=False)
    fig.suptitle(r'Extended window: $0.1$--$100\,M_{\rm cool}$, $z=14.5$' if args.wide else r'Visbal-style statistical comparison: $z=14.5$',y=.97,fontsize=16)
    fig.text(.13,.91,r'Window: $4\times3600$ masses $\times1000$ tracks; $f_\star=10\%$; CSFR 100 Myr',fontsize=11)
    fig.text(.13,.025,'Shading: window batch SE only. Formal duty mixture; no dust; same radiation assumptions.',fontsize=9)
    out.mkdir(parents=True)
    fig.savefig(out/'uvlf.png',dpi=170);fig.savefig(out/'uvlf.pdf');plt.close(fig)
    np.savez_compressed(out/'uvlf.npz',bin_edges=edges,baseline=baseline,high=high,
        **{f'duty{k}':v for k,v in curves.items()},**{f'duty{k}_batch_se':v for k,v in errors.items()},
        **{f'duty{k}_batches':v for k,v in batches.items()})
    report=dict(inputs_sha256=hashes,runs=rows,diagnostics=diagnostics,mass_window_mcool=[lower,upper],
        method='Mean of four independent complete-window estimates plus original full-range weights restricted above window maximum; old window replaced, not added. No halos below lower bound included.',
        limitations=prior['limitations']+['Shading is four-batch window SE, excludes common high-tail MC uncertainty; not full convergence certification.'])
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(diagnostics,indent=2))


if __name__=='__main__':main()
