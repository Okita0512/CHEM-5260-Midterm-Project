"""
Model B – Effect of N  (plot)
Three panels in one row: P_0(t), P_1(t), P_K(t).
Black arrows point to the actual Poincaré revival feature in each curve.
Black dashed line: N→∞ WBA analytical result (Sec. II-C-2).
V_01 = 0.04 (overdamped regime, same as Fig 8).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

t          = np.load("data/t.npy")
N_list     = np.load("data/N_list.npy").astype(int)
V_1k_list  = np.load("data/V_1k_list.npy")
T_rec_list = np.load("data/T_rec_list.npy")
params     = np.load("data/params.npz")
Delta_E      = float(params["Delta_E"])
V_1k_ref     = float(params["V_1k_ref"])
V_01         = float(params["V_01"])
Gamma_target = float(params["Gamma_target"])

# ── WBA (N→∞) analytical curves ───────────────────────────────────────────────
def wba_populations(V01, Gamma, times):
    """Overdamped WBA on resonance (E0=E1=0)."""
    g4   = Gamma / 4.0
    disc = g4**2 - V01**2      # > 0 for overdamped
    Om   = np.sqrt(max(disc, 0.0))
    env  = np.exp(-g4 * times)
    if disc > 1e-12:
        c0    = env * (np.cosh(Om * times) + (g4 / Om) * np.sinh(Om * times))
        c1mag = (V01 / Om) * env * np.sinh(Om * times)
    else:                       # critical / marginal
        c0    = env * (1.0 + g4 * times)
        c1mag = g4 * times * env
    P0_wba = c0**2
    P1_wba = c1mag**2
    PK_wba = 1.0 - P0_wba - P1_wba
    return P0_wba, P1_wba, PK_wba

P0_wba, P1_wba, PK_wba = wba_populations(V_01, Gamma_target, t)

# ── helper: find revival feature near T_rec ───────────────────────────────────
def find_revival(t, P, T_rec, half_win_frac=0.18, feature='max'):
    """Return (t_feat, P_feat) of the local max/min of P nearest to T_rec."""
    hw = half_win_frac * T_rec
    mask = (t >= T_rec - hw) & (t <= T_rec + hw)
    if mask.sum() == 0:
        return T_rec, P[np.argmin(np.abs(t - T_rec))]
    sub = P[mask]
    idx = np.argmax(sub) if feature == 'max' else np.argmin(sub)
    return t[mask][idx], sub[idx]

colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(N_list)))

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)

# ── collect data for arrow anchoring ──────────────────────────────────────────
P0_all, P1_all, PK_all = [], [], []
for N in N_list:
    P0_all.append(np.load(f"data/P0_N{N}.npy"))
    P1_all.append(np.load(f"data/P1_N{N}.npy"))
    PK_all.append(np.load(f"data/PK_N{N}.npy"))

# ── population curves ─────────────────────────────────────────────────────────
for N, V_1k_N, T_rec, col, P0, P1, PK in zip(
        N_list, V_1k_list, T_rec_list, colors, P0_all, P1_all, PK_all):
    lbl = rf"$N={N}$, $V_{{1k}}={V_1k_N:.3f}$, $T_{{\rm rec}}={T_rec:.1f}$"
    axes[0].plot(t, P0, color=col, lw=1.3, label=lbl)
    axes[1].plot(t, P1, color=col, lw=1.3)
    axes[2].plot(t, PK, color=col, lw=1.3)

# ── WBA (N→∞) dashed black – overdamped analytical (Eqs. 33–34) ──────────────
wba_lbl = r"$N\!\to\!\infty$, overdamped WBA (Eqs.~33–34)"
axes[0].plot(t, P0_wba, 'k--', lw=1.4, label=wba_lbl)
axes[1].plot(t, P1_wba, 'k--', lw=1.4)
axes[2].plot(t, PK_wba, 'k--', lw=1.4)

# ── set ylims before drawing arrows ───────────────────────────────────────────
P1_max = max(P.max() for P in P1_all)   # auto-range panel B
axes[0].set_ylim(-0.02, 1.12)
axes[1].set_ylim(-0.005, P1_max * 1.25)
axes[2].set_ylim(-0.02, 1.12)

# ── Poincaré recurrence arrows – panel A only ─────────────────────────────────
arrow_kw = dict(arrowstyle='->', color='black', lw=1.3)
offset_frac = 0.12

for T_rec, P0 in zip(T_rec_list, P0_all):
    t_f, P_f = find_revival(t, P0, T_rec, feature='max')
    span0 = axes[0].get_ylim()[1] - axes[0].get_ylim()[0]
    axes[0].annotate('', xy=(t_f, P_f),
                     xytext=(t_f, P_f + offset_frac * span0),
                     arrowprops=arrow_kw)

# ── formatting ────────────────────────────────────────────────────────────────
ylabels = [r"$P_0(t)$", r"$P_1(t)$", r"$P_K(t)$"]
titles  = [r"Upstream $P_0(t)$", r"Doorway $P_1(t)$",
           r"Bath $P_K(t)=1-P_0-P_1$"]

for ax, ylabel, title in zip(axes, ylabels, titles):
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"Time $t$ (dimensionless)", fontsize=9)

axes[0].legend(fontsize=8, frameon=False)

# Panel labels – upper left
for ax, lbl in zip(axes, ["A", "B", "C"]):
    ax.text(0.03, 0.97, lbl, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

fig.suptitle(
    rf"Model B: Effect of $N$ at fixed $\Gamma={Gamma_target:.3f}$"
    rf"   ($\Delta E={Delta_E},\ V_{{01}}={V_01},\ "
    rf"V_{{1k,\rm ref}}={V_1k_ref}$)",
    fontsize=10, y=1.02)

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Figure_effect_N_modelB.pdf", bbox_inches="tight")
fig.savefig("figures/Figure_effect_N_modelB.png", dpi=150, bbox_inches="tight")
print("Saved figures/Figure_effect_N_modelB.pdf")
plt.show()
