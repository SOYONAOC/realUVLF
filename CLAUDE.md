# realUVLF Project

## Rules

- **SLURM job submission**: Always use the `slurm-auto-node-select` skill. Never manually write sbatch scripts or run `sbatch` directly.
- **Heavy computation**: Any Python computation expected to take >30s must go through a SLURM compute node (via the skill above).
- **Plotting**: Before writing or editing any Python file that uses matplotlib (`import matplotlib`, `plt.savefig`, `plt.subplots`, etc.), invoke the `physics-aware-plotting` skill first.
