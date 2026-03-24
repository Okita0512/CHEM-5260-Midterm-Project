"""
Model B – Coupling disorder  (plot)
Three-panel one-row figure: P_0(t), P_1(t), P_K(t).
Overlays uniform, random-sign V_{1k}, and Gaussian V_{1k}.
Mirrors the layout of Model A / Coupling_disorder (Fig 6).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

t        = np.load("data/t.npy")
P0_unif  = np.load("data/P0_unif.npy")
P1_unif  = np.load("data/P1_unif.npy")
PK_unif  = np.load("data/PK_unif.npy")
P0_rs    = np.load("data/P0_rs.npy")
P1_rs    = np.load("data/P1_rs.npy")
PK_rs    = np.load("data/PK_rs.npy")
P0_gauss = np.load("data/P0_gauss.npy")
P1_gauss = np.load("data/P1_gauss.npy")
PK_gauss = np.load("data/PK_gauss.npy")
params   = np.load("data/params.npy")
N, Delta_E, V_1k, V_01, E0, E1, Gamma = params

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)

for ax, P_unif, P_rs, P_gauss, ylabel, title in zip(
        axes,
        [P0_unif, P1_unif, PK_unif],
        [P0_rs,   P1_rs,   PK_rs],
        [P0_gauss, P1_gauss, PK_gauss],
        [r"$P_0(t)$", r"$P_1(t)$", r"$P_K(t)$"],
        [r"Upstream $P_0(t)$", r"Doorway $P_1(t)$",
         r"Bath $P_K(t) = 1 - P_0 - P_1$"]):

    # uniform – single deterministic curve
    ax.plot(t, P_unif, lw=1.5, color="steelblue", label="Uniform $V_{1k}$")

    # random-sign V_{1k}: mean ± std
    mu_rs  = P_rs.mean(axis=0)
    std_rs = P_rs.std(axis=0)
    ax.plot(t, mu_rs, lw=1.5, color="darkorange", ls="--",
            label="Random-sign $V_{1k}$ (mean)")
    ax.fill_between(t, mu_rs - std_rs, mu_rs + std_rs,
                    color="darkorange", alpha=0.2)

    # Gaussian V_{1k}: mean ± std
    mu_g  = P_gauss.mean(axis=0)
    std_g = P_gauss.std(axis=0)
    ax.plot(t, mu_g, lw=1.5, color="forestgreen",
            label=r"Gaussian $V_{1k}$ (mean)")
    ax.fill_between(t, mu_g - std_g, mu_g + std_g,
                    color="forestgreen", alpha=0.2)

    ax.set_xlabel(r"Time $t$ (dimensionless)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, t[-1])

axes[0].legend(fontsize=8, frameon=False)

# Panel labels – upper left
for ax, lbl in zip(axes, ["A", "B", "C"]):
    ax.text(0.03, 0.97, lbl, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

fig.suptitle(
    rf"Model B: Coupling disorder   "
    rf"($N={int(N)},\ \Delta E={Delta_E},"
    rf"\ V_{{1k,\rm rms}}={V_1k},\ V_{{01}}={V_01},\ \Gamma={Gamma:.3f}$)",
    fontsize=10, y=1.02)

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Figure_coupling_disorder_modelB.pdf", bbox_inches="tight")
fig.savefig("figures/Figure_coupling_disorder_modelB.png", dpi=150, bbox_inches="tight")
print("Saved figures/Figure_coupling_disorder_modelB.pdf")
plt.show()
