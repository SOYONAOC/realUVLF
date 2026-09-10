# Visbal duty statistical comparison

Created 2026-09-07 on codex/popiii-visbal-duty from 6c846e5.

User-selected scope: independent reproduction-style statistical comparison of
Visbal, Haiman & Bryan (2015), arXiv:1501.03177, section 4 equations 9–11.
This is not the complete causal Pop III production model.

- Active SFR = f_star (Omega_b/Omega_m) M_h / (epsilon_duty t_H).
- Population occupancy = epsilon_duty within [M_cool, 2 M_cool].
- Do not reinterpret this as measured per-halo accretion or a causal birth history.
- Derive the UV conversion from the real SSP with explicitly documented constant
  SFR duration; the paper's He II coefficient is not a UV conversion coefficient.
- Account for active/inactive weights without duplicating the halo population.
- Preserve the old Pop II baseline and explicitly state the assumption if adding
  Pop III to that baseline, rather than treating the mixture as author-published.
- Initial deliverable: standalone z=14.5 UVLF with the same observations; no slides.
- Keep production defaults unchanged and original worktrees unmodified.

Original worktree has extensive uncommitted changes; these were not copied into
this branch. Existing SSP and run products must be accessed through explicit,
provenance-recorded paths, not fabricated or silently substituted.

Status: AUR-EX-0006-R001 completed as local post-processing of R021/R024;
no new histories or simulation submitted. Configuration is
configs/experiments/visbal_duty.toml, executable scripts/analysis/visbal_duty_uvlf.py.
Three focused tests pass; input hashes, number conservation, duty-independent
mean luminosity and independent SSP quadrature checks pass.

The UV extension uses logE total 1500 A luminosity after 100 Myr of constant
SFR. This is an additional diagnostic normalization, not an author-published
UV coefficient or a finite-duration burst consistent with duty times Hubble time.
Project threshold and cosmology are retained for the controlled comparison.
At fstar=0.1, duty=0.01/0.03/0.1, brightest sampled Pop III magnitudes are
-18.1823/-16.9895/-15.6823. No claim of sampling convergence or complete physics.
See data_save/AUR-EX-0006-R001/summary.json and uvlf.png for results.

2026-09-08: user requested denser sampling. R002–R005 are four independent
3600 mass x 1000 track batches over [M_atomic,2M_atomic] at z14.5, seeds
610102–610105, grid960. The optimized historical sampler supplies Pop II
luminosities and HMF weights only; its burst Pop III output is not the duty
prediction. Frozen release: visbal-sampling-20260908-01. Config copies
atomic_burst_R027–R030.toml retain the actual AUR-EX-0006 run IDs internally.
Login-side dry-run was stopped while delayed reading Python libraries; unchanged
validate-only checks were submitted to compute-node preflight 11571057.
Sampling array 11571058 (tasks 0–3 = R002–R005) depends on its successful exit.
No completion or new sampling-convergence result is claimed yet.

User then selected local fat2, leaving one idle core: 41 idle => request40,
shared allocation. Job156101 is RUNNING; R006–R009 sequentially replace R002–R005
with identical seeds/science and workers40. Frozen source is under main worktree
temp_data/sc-deployments/visbal-fat2-20260908-01. Remote duplicate array11571058
was requested cancelled after it unexpectedly started ahead of its prediction;
its partial data must not be added as independent realizations.

Latest user instruction switched back to the supercomputer. Both old array
11571058 and fat2 job156101 are cancelled. R010–R013 replace the same four seeds
at56 workers in release visbal-resume-20260908-01: preflight11571301,
dependent array11571302. Preserve all cancelled attempts; do not count partial
products or claim completed new sampling.

R010–R013 now completed (all exit0), retrieved and all declared hashes checked.
scripts/analysis/combine_visbal_batches.py averages independent window integrals,
replaces old window samples and adds the original disjoint high tail.
data_save/visbal_combined_20260908 contains the inspected standalone plot and
per-batch arrays. Peak-bin densities decrease6.4–6.7% versus R001; window-only
batch standard errors at these bins are0.45–1.05%. No full-curve convergence
claim; model/cosmology/radiation limitations remain unchanged. Four focused
tests passed this turn. Slides remain unchanged.

User-requested expanded window0.1–100Mcool: R014–R017, seeds610114–610117,
same3600x1000 and grid960, submitted to supercomputer array11576600 after
preflight11576598. Explicit mass_min_ratio0.1 in configs R039–R042;
old sampler defaults stay1. This includes halo ranges outside the paper's
pristine-occupation assumption and is only a statistical sensitivity test.
Use combine_visbal_batches.py --wide only after complete products are retrieved.
No wide-window figure/result exists yet; old results are preserved.

## 2026-09-09: random first-crossing q experiment (not snapshot duty)

User adopted log10(q) ~ Normal(0.5, 1.5^2), untruncated, independently once
per history. Scripts `experiments/random_q_burst.py`, `run/run_random_q_burst.py`
and `submit/submit_random_q.py` implement a separate UV-only experiment.
The SFR calculator now has an explicit numerically equivalent direct-convolution
backend; the production default remains dense. Interpolate first crossing in log(Mh/Mcool) versus
cosmic time, then form epsilon_b fb Mh(cross). Evaluate the SSP at actual burst
age, retaining the project 100 Myr UV lookback, not a CSFR coefficient or duty.
Uncrossed histories remain untriggered; left-censored histories are marked, not
given a fabricated start-time event. Same histories provide PopII and q=1 controls.
PopII uses its own atomic active mask; random triggers use the full stored DM
history. This is not a merger-forest/pristine/enrichment/feedback model.

R018: 8 mass x100 tracks preflight. R019-R022: four independent 3600x1000
batches, 960 time nodes, log10 final halo mass 5-12, track chunks100, 56workers.
Use `scripts/analysis/analyze_random_q.py --runs ... --output ... --observations ...`
only on completed checksum-verified products. No q result is claimed yet.
Supercomputer release random-q-20260909-01; preflight11585226, array11585227.
Final full tests after restoring real SSP files: 1100 passed, one unrelated
ML notebook cell-count failure (24 versus 25). Earlier four benchmark failures
were caused by this turn's SSP symlink and were repaired by restoring real files;
they were not pre-existing failures. Do not describe the full suite as green.

2026-09-09 correction: all debug partitions are forbidden, including preflights.
R019-R022 / array11585227 were canceled after the incorrect debug6 move; no
complete science products exist for those attempts. R023 is the accelerated
preflight; R024-R027 retry the original four seeds on cp6 only. Logs and snapshots
remain preserved. Release02 was never submitted (invalid unpadded run IDs);
release03 uses corrected validated IDs. No scheduler time or memory is requested.
Submitted on cp6: R023 job11585420; R024-R027 array11585421_0-3, dependent on
successful preflight. Verified pending at 12:22; no active debug jobs remain.

Later user correction: do not queue a separate preflight. Removed the dependency
from array11585421 before canceling R023/job11585420 (zero runtime). R024-R027
retain their original IDs and submissions, now directly pending on cp6/Priority.
The frozen release03 hashes were verified read-only; it was not changed. Future
submissions perform startup file validation inside each science job and submit
no additional preflight. Two scheduler-mocked submission tests passed.

R024-R027 are now complete on cp6, each exit0, elapsed4:49/4:49/4:16/4:20;
all 14.4 million tracks retrieved and product/source/input hashes verified.
Results: data_save/random_q_combined_20260909_rev02/ (UVLF plot, arrays,
summary and per-batch cumulative statistics). At z14.5, cumulative MUV<=-20
densities are 2.50944e-8 Mpc^-3 for PopII, 3.87609e-6 for epsilon3%, and
4.67042e-5 for epsilon10%. The broad random threshold reaches the observed
bright-end range in this prescribed UV experiment, not a validated causal
formation model or simultaneous fit to all observations. Bright-bin cluster
MC SE is about5-6%; this is not a sample-size convergence proof. Slides unchanged.

z12.5 follow-up: user explicitly chose fat2 and all idle cores, then requested
one batch first rather than four. Only R028 is submitted (job156653,12cores,
3600x1000tracks,960times,seed610228). No separate preflight, no time/memory
requests, no other jobs canceled. Explicit --compute-site fat2 requires the fat
partition and fat2 allocation; default cp6/debug ban remains. Observation plot
uses the three existing redshift_12p5 tables, with redshift ranges labeled.

User requested a separate 10-core scheduling attempt without canceling R028.
R029/job156659 uses the same seed610228 and science configuration, only workers10
and a distinct run ID/output snapshot. R028/job156653 is untouched. These two
runs are duplicates for science and must not be combined as independent samples.

One-core trial requested next: R030/job156665, same seed and science, workers1.
At this turn's first inspection R028/R029 were already CANCELLED at16:46:47;
no cancellation was performed in this turn. Only R030 was newly submitted.

User then approved node1-3,3cores each: R031/job156678, one 3600x1000
realization distributed over three disjoint global mass-index shards. The
runner uses --compute-site node123 --slurm-shards 3 under srun --mpi=none;
successful shards are hash/coverage checked and joined automatically without
renormalizing the original global weights. No independent extra batches or
queued preflight. Release random-q-z12p5-node123-20260909-02 permits the explicit
3core requests to wait if a node is temporarily busy; never oversubscribes.
R030 was already canceled externally to this turn at16:49:20. No old job changed.

Supercomputer recovery, 2026-09-09: R031 was observed CANCELLED at17:01:38,
zero runtime; this turn did not cancel it. Submitted R032/job11588898 on cp6,
cn83116,56cores, one unsharded 3600x1000 realization, seed610228. No queued
preflight/dependency, debug partition, requested time/memory, or extra batches.
Started17:12:24, completed17:17:33, exit0, elapsed5:09, batch MaxRSS35852476K.
Source hashes match the frozen random-q-z12p5-cp6-20260909-01 deployment;
both SSP hashes match the frozen release and real local input tables.
Product checksums and event/weight validity passed in analyze_random_q.py.

Results: data_save/random_q_z12p5_20260909/uvlf.png,uvlf.pdf,uvlf.npz,summary.json.
At z12.5, cumulative MUV<=-20 densities (Mpc^-3) are PopII6.03933e-7,
epsilon1%2.36688e-6,3%1.79778e-5,10%1.28093e-4; cluster SE respectively
4.32888e-8,1.54533e-7,1.30510e-6,9.21435e-6. At bin center-20.25,
phi(3%)=2.06123e-5 and phi(10%)=1.40807e-4 Mpc^-3 mag^-1, compared with
the existing Donnan24 table value1.6e-5. Three-percent efficiency approaches
the observed scale; ten percent generally overshoots. This is not a fit.
Observed-bin relative cluster SE is10.56-11.41% for3% and10.72-13.65% for10%.
Single-batch SE is not a convergence proof; the fixed-q comparator is especially
noisy near-18 (cumulative SE49.6%). No extra batch is automatically submitted.
The single PNG was visually inspected; observation redshift ranges are labeled;
no curve smoothing or slides update. All UV-only/pristine-closure limitations
remain. R028-R031 share this seed and must never be combined as independent runs.
