"""
Model B – Sequential population flow  (plot)
Five-panel layout: upper row A B C (very overdamped / overdamped / critical),
                   lower row D E   (underdamped / strongly underdamped).
Solid: exact TDSE.  Dashed: WBA analytical.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

t        = np.load("data/t.npy")
V01_list = np.load("data/V01_list.npy")
labels   = list(np.load("data/labels.npy"))
params   = np.load("data/params.npz")
N        = int(params["N"])
Delta_E  = float(params["Delta_E"])
V_1k     = float(params["V_1k"])
Gamma    = float(params["Gamma"])
V01_crit = float(params["V01_critical"])

# Equation reference per panel (overdamped→33-34, critical→35, underdamped→36-37)
eq_refs      = ["Eqs.~33-34", "Eqs.~33-34", "Eq.~35", "Eqs.~36-37", "Eqs.~36-37"]
panel_labels = ["A", "B", "C", "D", "E"]
titles = [
    rf"$V_{{01}}={V01_list[0]:.2f}$, $V_{{01}}/(\Gamma/4)={V01_list[0]/V01_crit:.2f}$" + "\n(Very overdamped)",
    rf"$V_{{01}}={V01_list[1]:.2f}$, $V_{{01}}/(\Gamma/4)={V01_list[1]/V01_crit:.2f}$" + "\n(Overdamped)",
    rf"$V_{{01}}=\Gamma/4\approx{V01_crit:.4f}$" + "\n(Critical)",
    rf"$V_{{01}}={V01_list[3]:.2f}$, $V_{{01}}/(\Gamma/4)={V01_list[3]/V01_crit:.2f}$" + "\n(Underdamped)",
    rf"$V_{{01}}={V01_list[4]:.2f}$, $V_{{01}}/(\Gamma/4)={V01_list[4]/V01_crit:.2f}$" + "\n(Strongly underdamped)",
]

fig = plt.figure(figsize=(13, 8))
gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.45, wspace=0.38)

# Upper row: 3 panels spanning 2 columns each
ax_A = fig.add_subplot(gs[0, 0:2])
ax_B = fig.add_subplot(gs[0, 2:4])
ax_C = fig.add_subplot(gs[0, 4:6])
# Lower row: 2 panels centered (columns 1-3 and 3-5)
ax_D = fig.add_subplot(gs[1, 1:3])
ax_E = fig.add_subplot(gs[1, 3:5])

axes = [ax_A, ax_B, ax_C, ax_D, ax_E]

for ax, lbl, title, panel, eq in zip(axes, labels, titles, panel_labels, eq_refs):
    P0  = np.load(f"data/P0_{lbl}.npy")
    P1  = np.load(f"data/P1_{lbl}.npy")
    PK  = np.load(f"data/PK_{lbl}.npy")
    P0w = np.load(f"data/P0_wba_{lbl}.npy")
    P1w = np.load(f"data/P1_wba_{lbl}.npy")

    ax.plot(t, P0, lw=1.6, color="steelblue",   label=r"$P_0$ (exact)")
    ax.plot(t, P1, lw=1.6, color="darkorange",  label=r"$P_1$ (exact)")
    ax.plot(t, PK, lw=1.6, color="forestgreen", label=r"$P_K$ (exact)")
    ax.plot(t, P0w, lw=1.1, color="steelblue",  ls="--", alpha=0.65,
            label=rf"$P_0$ (WBA, {eq})")
    ax.plot(t, P1w, lw=1.1, color="darkorange", ls="--", alpha=0.65,
            label=rf"$P_1$ (WBA, {eq})")

    ax.set_xlabel(r"Time $t$", fontsize=9)
    ax.set_ylabel("Population", fontsize=9)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7, frameon=False, ncol=1)
    # Panel label
    ax.text(0.03, 0.96, panel, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

fig.suptitle(
    rf"Model B: Damping regimes   "
    rf"($N={N},\ \Delta E={Delta_E},\ V_{{1k}}={V_1k},\ "
    rf"\Gamma\approx{Gamma:.3f},\ \Gamma/4\approx{V01_crit:.4f}$)",
    fontsize=10, y=1.01)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Figure_sequential_flow.pdf", bbox_inches="tight")
fig.savefig("figures/Figure_sequential_flow.png", dpi=150, bbox_inches="tight")
print("Saved figures/Figure_sequential_flow.pdf")
plt.show()
