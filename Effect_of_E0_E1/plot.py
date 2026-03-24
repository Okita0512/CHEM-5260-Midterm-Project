"""
Model B – Resonance effect  (plot)
Three-panel one-row figure:
  A: P_0(t) and P_1(t) for all Delta_01
  B: P_K(t) for all Delta_01
  C: extracted gamma_eff vs Delta_01 with WBA Lorentzian
Same color for |Delta_01| pairs; positive = solid, negative = dashed.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

t            = np.load("data/t.npy")
Delta01_list = np.load("data/Delta01_list.npy")
rates        = np.load("data/rates.npy")
params       = np.load("data/params.npz")
N     = int(params["N"])
Delta_E = float(params["Delta_E"])
V_1k  = float(params["V_1k"])
V_01  = float(params["V_01"])
E1    = float(params["E1"])
Gamma = float(params["Gamma"])

# ── color / style assignment ──────────────────────────────────────────────────
# Unique |Delta_01| values, mapped to colors
abs_vals  = sorted(set(abs(d) for d in Delta01_list))
base_cols = plt.cm.plasma(np.linspace(0.1, 0.85, len(abs_vals)))
col_map   = {v: c for v, c in zip(abs_vals, base_cols)}

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

for d01 in Delta01_list:
    tag = str(d01).replace(".", "p").replace("-", "m")
    P0 = np.load(f"data/P0_d{tag}.npy")
    P1 = np.load(f"data/P1_d{tag}.npy")
    PK = np.load(f"data/PK_d{tag}.npy")
    col = col_map[abs(d01)]
    ls  = "-" if d01 >= 0 else "--"
    lbl = rf"$\Delta_{{01}}={d01:+.1f}$"
    # Panel A: P_0 solid/dashed + P_1 with reduced alpha to distinguish
    axes[0].plot(t, P0, color=col, lw=1.4, ls=ls, label=lbl)
    axes[0].plot(t, P1, color=col, lw=1.0, ls=ls, alpha=0.45)
    # Panel B: P_K
    axes[1].plot(t, PK, color=col, lw=1.3, ls=ls, label=lbl)

# Panel A formatting
axes[0].set_xlabel(r"Time $t$ (dimensionless)", fontsize=9)
axes[0].set_ylabel(r"Population", fontsize=9)
axes[0].set_title(r"$P_0(t)$ (bold) and $P_1(t)$ (faint)", fontsize=9)
axes[0].set_xlim(0, t[-1])
axes[0].set_ylim(-0.02, 1.12)
axes[0].legend(fontsize=7.5, frameon=False, ncol=1,
               title="Solid: $\\Delta_{01}>0$, dashed: $<0$",
               title_fontsize=7)

# Panel B formatting
axes[1].set_xlabel(r"Time $t$ (dimensionless)", fontsize=9)
axes[1].set_ylabel(r"$P_K(t)$", fontsize=9)
axes[1].set_title(r"Bath population $P_K(t)=1-P_0-P_1$", fontsize=9)
axes[1].set_xlim(0, t[-1])
axes[1].set_ylim(-0.02, 1.12)
axes[1].legend(fontsize=7.5, frameon=False, ncol=1,
               title="Solid: $\\Delta_{01}>0$, dashed: $<0$",
               title_fontsize=7)

# ── panel C: gamma_eff vs Delta_01 ───────────────────────────────────────────
# WBA Lorentzian: gamma_P0 = V_01^2 * Gamma / (Delta_01^2 + (Gamma/2)^2)
d_dense = np.linspace(-4, 4, 400)
gamma_wba = V_01**2 * Gamma / (d_dense**2 + (Gamma / 2)**2)

# Scatter: same color code as panels A/B
for d01, gamma in zip(Delta01_list, rates):
    col = col_map[abs(d01)]
    ms  = "o" if d01 >= 0 else "s"
    axes[2].scatter(d01, gamma, color=col, marker=ms, s=55, zorder=5)

axes[2].plot(d_dense, gamma_wba, "k-", lw=1.5,
             label=r"WBA Lorentzian (Eqs.~33--34)")
axes[2].axvline(0, color="gray", lw=0.8, ls=":")
axes[2].set_xlabel(r"Detuning $\Delta_{01} = E_0 - E_1$", fontsize=9)
axes[2].set_ylabel(r"Decay rate $\gamma_\mathrm{eff}$", fontsize=9)
axes[2].set_title(r"Resonance profile $\gamma_\mathrm{eff}(\Delta_{01})$", fontsize=9)
axes[2].legend(fontsize=8, frameon=False)
axes[2].set_xlim(-4, 4)
axes[2].set_ylim(0, 0.028)

# Panel labels – placed at lower-left to avoid top-left legends
for ax, lbl in zip(axes, ["A", "B", "C"]):
    ax.text(0.03, 0.05, lbl, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

fig.suptitle(
    rf"Model B: Resonance effect   "
    rf"($N={N},\ \Delta E={Delta_E},\ V_{{1k}}={V_1k},\ "
    rf"V_{{01}}={V_01},\ \Gamma\approx{Gamma:.3f}$)",
    fontsize=10, y=1.02)

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Figure_effect_E0_E1.pdf", bbox_inches="tight")
fig.savefig("figures/Figure_effect_E0_E1.png", dpi=150, bbox_inches="tight")
print("Saved figures/Figure_effect_E0_E1.pdf")
plt.show()
