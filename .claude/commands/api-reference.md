# realUVLF API Quick Reference

## `mah.generate_halo_histories()`
`from mah import generate_halo_histories`

Monte Carlo halo mass accretion histories.

**Parameters:**
- `n_tracks` (int) -- number of tracks
- `z_final` (float) -- terminal redshift
- `Mh_final` (float) -- halo mass at `z_final`
- `z_start_max` = `50.0`
- `M_min` = `None` (uses `massfunc.SFRD().M_vir(mu=0.61, Tvir=1e4, z)`; accepts scalar, array, or callable `M_min(z)`)
- `cosmology` = `None` (`mah.Cosmology`)
- `random_seed` = `None`
- `time_grid_mode` -- `"uniform_in_z"` | `"uniform_in_t"` | `"custom"`
- `dt` -- time step in Gyr (for `uniform_in_t`)
- `dz` -- redshift step (for `uniform_in_z`)
- `custom_grid` -- custom redshift grid (for `custom`)
- `store_inactive_history` = `None`
- `sampler` -- `"mcbride"` | `"gaussian"`
- `pilot_samples` -- pilot count for `sampler="gaussian"`

**Returns:** `HaloHistoryResult` with `tracks` (dict) and `metadata` (dict).
`tracks` keys: `halo_id`, `step`, `z`, `t_gyr`, `dt_gyr`, `Mh`, `dMh_dt`, `active_flag`, `termination_flag`

---

## `sfr.compute_sfr_from_tracks()`
`from sfr import compute_sfr_from_tracks`

**Parameters:**
- `tracks` -- `dict[str, np.ndarray]` (requires `halo_id`, `step`, `z`, `t_gyr`, `Mh`, `dMh_dt`)
- `mu` = `0.61`
- `atomic_cooling_temperature` = `1e4`
- `enable_time_delay` = `False`
- `burst_kappa` = `0.1`
- `burst_lookback_max_myr` = `100.0`
- `model_parameters` = `None` (`SFRModelParameters`)

**Returns:** `dict[str, np.ndarray]` -- input columns plus: `r_vir`, `V_c`, `T_vir`, `sigma_vbc_rms`, `V_cool_H2`, `M_cool_H2`, `M_atom`, `tau_del`, `td_burst`, `t_src`, `Mh_src`, `dMh_dt_src`, `fstar_src`, `fstar_now`, `pop2_active_flag`, `branch_active_flag`, `SFR_pop2`, `mdot_burst`, `SFR_total`, `SFR` (Msun/yr)

---

## `sfr.minihalo_mass_floor()`
`from sfr import minihalo_mass_floor`

H2-cooling minihalo mass threshold. `V_cool_H2 = sqrt(a^2 + (b*v_bc)^2)`, a=3.714 km/s, b=4.015.

**Parameters:** `redshift`, `v_bc_kms`=`None`, `cosmology`=`None`
**Returns:** `M_cool_H2` in Msun

---

## `ssp.load_uv1600_table()`
`from ssp import load_uv1600_table`

**Parameters:** `file_path`, `wavelength_a`=`1600.0`, `metallicity` (HDF5 only, Z/Zsun)
**Returns:** `(ages_myr, luminosity_per_msun)` -- erg/s/Hz/Msun. Internally cached.

## `ssp.interpolate_uv1600_luminosity_per_msun()`
`from ssp import interpolate_uv1600_luminosity_per_msun`

**Parameters:** `time_myr`, `file_path`, `wavelength_a`=`1600.0`, `metallicity`
**Returns:** float or ndarray, erg/s/Hz/Msun. Linear interp on log10(age), clamped.

## `ssp.compute_halo_uv_luminosity()`
`from ssp import compute_halo_uv_luminosity`

**Parameters:** `t_obs`, `t_history`, `mh_history`, `sfr_history` (Msun/yr), `ssp_age_grid`, `ssp_luv_grid` (erg/s/Hz/Msun), `M_min`, `t_z50`, `time_unit_in_years`=`1e9`, `return_details`=`False`
**Returns:** `L_uv_halo` (erg/s/Hz), or dict with `L_uv_halo`, `ti`, `mask_used`, `age_used`, `t_used`, `kernel_used`, `integrand_used`, `t_cross_Mmin` when `return_details=True`.

---

## `uvlf.run_halo_uv_pipeline()`
`from uvlf import run_halo_uv_pipeline`

Full pipeline: MAH -> SFR -> SSP UV convolution.

**Parameters:**
- `n_tracks` (int)
- `z_final` (float)
- `Mh_final` (float)
- `z_start_max` = `50.0`
- `n_grid` = `240`
- `ssp_file` = `"data_save/ssp_uv1600_topheavy_imf100_300_z0005.npz"`
- `cosmology` = `None`
- `random_seed` = `None`
- `sampler` = `"mcbride"`
- `enable_time_delay` = `False`
- `workers` -- reserved (currently serial)
- `burst_lookback_max_myr` = `100.0`
- `ssp_lookback_max_myr` -- SSP UV convolution max lookback
- `sfr_model_parameters` = `DEFAULT_SFR_MODEL_PARAMETERS`

**Returns:** `HaloUVPipelineResult`
Fields: `histories`, `sfr_tracks`, `uv_luminosities` (erg/s/Hz), `muv`, `redshift_grid`, `floor_mass`, `active_grid`, `metadata`

---

## `uvlf.sample_uvlf_from_hmf()`
`from uvlf import sample_uvlf_from_hmf`

HMF-weighted Monte Carlo UVLF sampling.

**Parameters:**
- `z_obs` (float)
- `N_mass` = `3000`
- `n_tracks` = `1000`
- `random_seed` = `None`
- `quantity` = `"Muv"` | `"luminosity"`
- `bins` = `40` (int or edges array)
- `logM_min` = `9`
- `logM_max` = `13`
- `z_start_max` = `50.0`
- `n_grid` = `240`
- `sampler` = `"mcbride"`
- `enable_time_delay` = `False`
- `pipeline_workers` -- outer-loop parallelism
- `ssp_file` = `"data_save/ssp_uv1600_topheavy_imf100_300_z0005.npz"`
- `progress_path` = `None`
- `sfr_model_parameters` = `DEFAULT_SFR_MODEL_PARAMETERS`

**Returns:** `UVLFSamplingResult`
`samples` keys: `logMh`, `Mh`, `mass_weight`, `track_index`, `luminosity`, `Muv`, `sample_weight`
`uvlf` keys: `quantity`, `bin_edges`, `bin_centers`, `bin_width`, `weighted_counts`, `phi`

---

## `uvlf.compute_dust_attenuated_uvlf()`
`from uvlf import compute_dust_attenuated_uvlf`

**Parameters:** `intrinsic_muv`, `intrinsic_phi` (Mpc^-3 mag^-1), `z`, `muv_obs`=`None`, `c0`/`c1`/`m0` (dust model), `clip_to_bounds`=`True`
**Returns:** dict with `Muv_obs`, `Muv_intrinsic`, `A_uv`, `dMuv_dMuv_obs`, `phi_nodust_obs`, `phi_intrinsic_interp`, `phi_obs`, `phi_obs_eval`, `transition_index`

## Dust helpers
`from uvlf import uv_continuum_slope_beta, uv_dust_attenuation, intrinsic_muv_from_observed, intrinsic_muv_jacobian`
- `uv_dust_attenuation(muv_obs, z, c0=2.10, c1=4.85, m0=-19.5)` -> A_UV
- `intrinsic_muv_from_observed(muv_obs, z)` -> M_UV = M_UV^obs - A_UV

---

## `massfunc.Mass_func.dndmst()`
`from massfunc import Mass_func`

**Parameters:** `M` (mass), `z` (redshift)
**Returns:** `dndm_st` (Mpc^-3 Msun^-1), Sheth-Tormen mass function
