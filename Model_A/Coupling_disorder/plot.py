"""
Model A – Coupling disorder  (plot)
Overlays uniform, random-sign, and Gaussian-random P_1(t).
Shaded bands show seed-to-seed variance for stochastic cases.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

t           = np.load("data/t.npy")
P1_uniform  = np.load("data/P1_uniform.npy")
P1_rsign    = np.load("data/P1_rsign.npy")   # (N_seeds, Nt)
P1_gauss    = np.load("data/P1_gauss.npy")   # (N_seeds, Nt)
FGR         = np.load("data/FGR.npy")
N, Delta_E, V, E1, Gamma = np.load("data/params.npy")

fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

for ax, yscale in zip(axes, ["linear", "log"]):
    # uniform
    ax.plot(t, P1_uniform, lw=1.5, color="steelblue", label="Uniform")

    # random-sign: mean ± std
    mu_rs  = P1_rsign.mean(axis=0)
    std_rs = P1_rsign.std(axis=0)
    ax.plot(t, mu_rs, lw=1.5, color="darkorange", label="Random-sign (mean)")
    ax.fill_between(t, mu_rs - std_rs, mu_rs + std_rs,
                    color="darkorange", alpha=0.2)

    # Gaussian: mean ± std
    mu_g  = P1_gauss.mean(axis=0)
    std_g = P1_gauss.std(axis=0)
    ax.plot(t, mu_g, lw=1.5, color="forestgreen", label="Gaussian (mean)")
    ax.fill_between(t, mu_g - std_g, mu_g + std_g,
                    color="forestgreen", alpha=0.2)

    # FGR
    ax.plot(t, FGR, "k--", lw=1.2, label=r"FGR $e^{-\Gamma t}$")

    ax.set_ylabel(r"$P_1(t)$")
    ax.set_yscale(yscale)
    ax.legend(fontsize=8, frameon=False)

axes[0].set_title(rf"Coupling disorder   ($N={int(N)},\ \Delta E={Delta_E},\ V_\mathrm{{rms}}={V}$) – linear")
axes[1].set_title("Log scale")
axes[1].set_xlabel(r"Time $t$ (dimensionless)")

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Figure_coupling_disorder.pdf", bbox_inches="tight")
fig.savefig("figures/Figure_coupling_disorder.png", dpi=150, bbox_inches="tight")
print("Saved figures/Figure_coupling_disorder.pdf")
plt.show()
