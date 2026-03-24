"""
Model A – Effect of V_{1k}  (plot)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

t            = np.load("data/t.npy")
V_list       = np.load("data/V_list.npy")
N, Delta_E, E1, rho = np.load("data/params.npy")

colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(V_list)))

fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

for ax, yscale in zip(axes, ["linear", "log"]):
    for V, col in zip(V_list, colors):
        Gamma_V = 2 * np.pi * V**2 * rho
        tag = str(V).replace(".", "p")
        P1 = np.load(f"data/P1_V{tag}.npy")
        ax.plot(t, P1, color=col, lw=1.2,
                label=rf"$V={V}$, $\Gamma={Gamma_V:.3f}$")
        # FGR prediction for this V
        ax.plot(t, np.exp(-Gamma_V * t), color=col, lw=0.8, ls="--")

    ax.set_ylabel(r"$P_1(t)$")
    ax.set_yscale(yscale)
    ax.legend(fontsize=8, frameon=False, title="Solid: exact,  dashed: FGR")

axes[0].set_title(rf"Effect of $V_{{1k}}$   ($N={int(N)},\ \Delta E={Delta_E}$) – linear")
axes[1].set_title("Log scale")
axes[1].set_xlabel(r"Time $t$ (dimensionless)")

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Figure_effect_V.pdf", bbox_inches="tight")
fig.savefig("figures/Figure_effect_V.png", dpi=150, bbox_inches="tight")
print("Saved figures/Figure_effect_V.pdf")
plt.show()
