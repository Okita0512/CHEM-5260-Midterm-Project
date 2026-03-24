"""
Model A - Effect of N (plot)
"""

import os

import matplotlib.pyplot as plt
import numpy as np


t = np.load("data/t.npy")
N_list = np.load("data/N_list.npy").astype(int)
V_list = np.load("data/V_list.npy")
T_rec_list = np.load("data/T_rec_list.npy")
FGR = np.load("data/FGR.npy")
params = np.load("data/params.npz")

Delta_E = float(params["Delta_E"])
V_ref = float(params["V_ref"])
Gamma_target = float(params["Gamma_target"])

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_list)))
P1_data = [np.load(f"data/P1_N{N}.npy") for N in N_list]


def first_revival_peak(t, P1, T_rec):
    window = (t >= 0.9 * T_rec) & (t <= 1.2 * T_rec)
    if not np.any(window):
        idx = np.argmin(np.abs(t - T_rec))
        return t[idx], P1[idx]
    indices = np.where(window)[0]
    idx_peak = indices[np.argmax(P1[window])]
    return t[idx_peak], P1[idx_peak]

fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

for ax, yscale in zip(axes, ["linear", "log"]):
    for N, V_N, T_rec, P1, col in zip(N_list, V_list, T_rec_list, P1_data, colors):
        ax.plot(
            t,
            P1,
            color=col,
            lw=1.2,
            label=rf"$N={N}$, $V_N={V_N:.3f}$, $T_{{\rm rec}}={T_rec:.1f}$",
        )
        ax.axvline(T_rec, color=col, lw=0.8, ls=":", alpha=0.35)

    ax.plot(t, FGR, "k--", lw=1.2, label=rf"FGR $e^{{-\Gamma t}}$, $\Gamma={Gamma_target:.3f}$")
    ax.set_ylabel(r"$P_1(t)$")
    ax.set_yscale(yscale)
    ax.set_xlim(0.0, t[-1])
    ax.legend(fontsize=8, frameon=False)

axes[1].set_xlabel(r"Time $t$ (dimensionless)")

for i, (T_rec, P1) in enumerate(zip(T_rec_list, P1_data)):
    peak_t, peak_y = first_revival_peak(t, P1, T_rec)
    text_x = peak_t + (-3.0 if i == 0 else 0.0 if i == 1 else 3.0)
    text_y = min(peak_y + 0.22, 0.92)
    label = "Poincare\nrecurrence" if i == 1 else ""
    axes[0].annotate(
        label,
        xy=(peak_t, peak_y),
        xytext=(text_x, text_y),
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.3),
    )

axes[0].set_title(
    rf"Effect of $N$ at fixed $\Gamma$ ($\Delta E={Delta_E},\ V_{{\rm ref}}={V_ref}$) - linear"
)
axes[1].set_title("Log scale")

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Figure_effect_N.pdf", bbox_inches="tight")
fig.savefig("figures/Figure_effect_N.png", dpi=150, bbox_inches="tight")
print("Saved figures/Figure_effect_N.pdf")
plt.show()
