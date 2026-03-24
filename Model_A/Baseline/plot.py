"""
Model A – Baseline dynamics  (plot)
Loads data/ produced by simulate.py and saves Figure_baseline.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ── load ──────────────────────────────────────────────────────────────────────
t      = np.load("data/t.npy")
P1     = np.load("data/P1.npy")
FGR    = np.load("data/FGR.npy")
params = np.load("data/params.npy")
N, Delta_E, V, E1, Gamma = params

# ── linear plot ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)

for ax, yscale in zip(axes, ["linear", "log"]):
    ax.plot(t, P1,  lw=1.5, label=r"$P_1(t)$ (exact)")
    ax.plot(t, FGR, lw=1.5, ls="--", color="tomato",
            label=r"$e^{-\Gamma t}$ (FGR)")
    ax.set_ylabel(r"Survival probability $P_1(t)$")
    ax.set_yscale(yscale)
    ax.legend(frameon=False)

axes[0].set_title(
    rf"$N={int(N)},\ \Delta E={Delta_E},\ V={V},\ \Gamma={Gamma:.3f}$  (linear)")
axes[1].set_title("Log scale")
axes[1].set_xlabel(r"Time $t$ (dimensionless)")

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Figure_baseline.pdf", bbox_inches="tight")
fig.savefig("figures/Figure_baseline.png", dpi=150, bbox_inches="tight")
print("Saved figures/Figure_baseline.pdf")
plt.show()
