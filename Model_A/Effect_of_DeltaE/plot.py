"""
Model A – Effect of Delta_E  (plot)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

t           = np.load("data/t.npy")
DeltaE_list = np.load("data/DeltaE_list.npy")
N, V, E1    = np.load("data/params.npy")

colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(DeltaE_list)))

fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

for ax, yscale in zip(axes, ["linear", "log"]):
    for dE, col in zip(DeltaE_list, colors):
        rho_dE   = N / dE
        Gamma_dE = 2 * np.pi * V**2 * rho_dE
        tag = str(dE).replace(".", "p")
        P1 = np.load(f"data/P1_dE{tag}.npy")
        ax.plot(t, P1, color=col, lw=1.2,
                label=rf"$\Delta E={dE}$, $\Gamma={Gamma_dE:.3f}$")
        ax.plot(t, np.exp(-Gamma_dE * t), color=col, lw=0.8, ls="--")

    ax.set_ylabel(r"$P_1(t)$")
    ax.set_yscale(yscale)
    ax.legend(fontsize=8, frameon=False, title="Solid: exact,  dashed: FGR")

axes[0].set_title(rf"Effect of $\Delta E$   ($N={int(N)},\ V={V}$) – linear")
axes[1].set_title("Log scale")
axes[1].set_xlabel(r"Time $t$ (dimensionless)")

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Figure_effect_DeltaE.pdf", bbox_inches="tight")
fig.savefig("figures/Figure_effect_DeltaE.png", dpi=150, bbox_inches="tight")
print("Saved figures/Figure_effect_DeltaE.pdf")
plt.show()
