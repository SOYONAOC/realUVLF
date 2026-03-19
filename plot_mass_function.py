#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from massfunc import Mass_func


OUTPUT_DIR = Path("outputs")
PNG_PATH = OUTPUT_DIR / "mass_function_comparison.png"
PDF_PATH = OUTPUT_DIR / "mass_function_comparison.pdf"
TXT_PATH = OUTPUT_DIR / "mass_function_comparison.txt"

Z_VALUES = [0.0, 6.0, 10.0, 20.0]
LOGM_MIN = 7.0
LOGM_MAX = 15.0
N_MASS = 400


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    masses = np.logspace(LOGM_MIN, LOGM_MAX, N_MASS)
    mf = Mass_func()
    mf.sigma2_interpolation_set()
    mf.dsig2dm_interpolation_set()

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)

    summary_lines = [
        "Mass function from massfunc.Mass_func().dndmst(M, z)",
        f"log10(M/Msun) range: [{LOGM_MIN}, {LOGM_MAX}]",
        f"n_mass: {N_MASS}",
        "z values: " + ", ".join(f"{z:g}" for z in Z_VALUES),
        "",
    ]

    for z in Z_VALUES:
        dndm = np.asarray(mf.dndmst(masses, z), dtype=float)
        finite = np.isfinite(dndm) & (dndm > 0.0)
        ax.plot(masses[finite], dndm[finite], linewidth=2.0, label=rf"$z={z:g}$")

        if np.any(finite):
            summary_lines.append(
                f"z={z:g}: dndm[min,max]=[{float(np.min(dndm[finite])):.6e}, {float(np.max(dndm[finite])):.6e}]"
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_{\rm h}\ [M_\odot]$")
    ax.set_ylabel(r"$dn/dM$")
    ax.set_title("Halo Mass Function from massfunc")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=10)

    fig.savefig(PNG_PATH, dpi=240)
    fig.savefig(PDF_PATH)
    plt.close(fig)

    TXT_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(PNG_PATH.resolve())
    print(PDF_PATH.resolve())
    print(TXT_PATH.resolve())


if __name__ == "__main__":
    main()
