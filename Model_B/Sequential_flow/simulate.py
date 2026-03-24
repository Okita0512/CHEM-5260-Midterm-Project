"""
Model B – Sequential population flow
Five panels covering the full damping landscape (Sec. II-C-2):
  A  V_01 = 0.01   very overdamped   V_01/(Gamma/4) ~ 0.13
  B  V_01 = 0.04   overdamped        V_01/(Gamma/4) ~ 0.51
  C  V_01 = Gamma/4  critical
  D  V_01 = 0.20   underdamped       V_01/(Gamma/4) ~ 2.55
  E  V_01 = 0.50   strongly underdamped  V_01/(Gamma/4) ~ 6.37
Fixed: N=200, Delta_E=10, V_1k=0.05, E0=E1=0, t_max=40.
"""

import numpy as np
import os

# ── fixed parameters ──────────────────────────────────────────────────────────
N       = 200
Delta_E = 10.0
V_1k    = 0.05
E0      = 0.0
E1      = 0.0

rho   = N / Delta_E
Gamma = 2 * np.pi * V_1k**2 * rho   # ≈ 0.3142

V01_critical = Gamma / 4.0
V01_list = [0.01, 0.04, V01_critical, 0.20, 0.50]
labels   = ["very_overdamped", "overdamped", "critical", "underdamped", "strongly_underdamped"]
panel_labels = ["A", "B", "C", "D", "E"]

t_max = 40.0
Nt    = 3000
times = np.linspace(0, t_max, Nt)

# ── helpers ───────────────────────────────────────────────────────────────────
def build_H_B(N, Delta_E, V_1k, V_01, E0=0.0, E1=0.0):
    dim = N + 2
    H = np.zeros((dim, dim))
    H[0, 0] = E0;  H[1, 1] = E1
    H[0, 1] = V_01;  H[1, 0] = V_01
    E_bath = np.linspace(E1 - Delta_E / 2, E1 + Delta_E / 2, N)
    np.fill_diagonal(H[2:, 2:], E_bath)
    H[1, 2:] = V_1k;  H[2:, 1] = V_1k
    return H

def propagate_B(H, times):
    vals, vecs = np.linalg.eigh(H)
    psi0 = np.zeros(H.shape[0]);  psi0[0] = 1.0
    c = vecs.T @ psi0
    P0 = np.zeros(len(times))
    P1 = np.zeros(len(times))
    for i, t in enumerate(times):
        psi_t = vecs @ (np.exp(-1j * vals * t) * c)
        P0[i] = abs(psi_t[0])**2
        P1[i] = abs(psi_t[1])**2
    return P0, P1, 1.0 - P0 - P1

def wba_populations(V01, Gamma, times):
    """Analytical WBA on resonance (E0=E1)."""
    g4 = Gamma / 4.0
    disc = g4**2 - V01**2
    env = np.exp(-g4 * times)
    if disc > 1e-12:           # overdamped
        Om = np.sqrt(disc)
        c0 = env * (np.cosh(Om * times) + (g4 / Om) * np.sinh(Om * times))
        c1_mag = (V01 / Om) * env * np.sinh(Om * times)
    elif disc < -1e-12:        # underdamped
        Om = np.sqrt(-disc)
        c0 = env * (np.cos(Om * times) + (g4 / Om) * np.sin(Om * times))
        c1_mag = (V01 / Om) * env * np.sin(Om * times)
    else:                      # critical
        c0 = env * (1.0 + g4 * times)
        c1_mag = g4 * times * env
    return c0**2, c1_mag**2

# ── run ───────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
np.save("data/t.npy", times)

for V01, lbl in zip(V01_list, labels):
    H = build_H_B(N, Delta_E, V_1k, V01, E0, E1)
    P0, P1, PK = propagate_B(H, times)
    np.save(f"data/P0_{lbl}.npy", P0)
    np.save(f"data/P1_{lbl}.npy", P1)
    np.save(f"data/PK_{lbl}.npy", PK)
    P0w, P1w = wba_populations(V01, Gamma, times)
    np.save(f"data/P0_wba_{lbl}.npy", P0w)
    np.save(f"data/P1_wba_{lbl}.npy", P1w)
    print(f"{lbl:22s}  V_01={V01:.5f}  V_01/(Gamma/4)={V01/V01_critical:.3f}")

np.save("data/V01_list.npy", np.array(V01_list))
np.save("data/labels.npy",   np.array(labels))
np.savez("data/params.npz",
         N=N, Delta_E=Delta_E, V_1k=V_1k, E0=E0, E1=E1,
         Gamma=Gamma, V01_critical=V01_critical)
print(f"\nGamma={Gamma:.4f}  Gamma/4={V01_critical:.5f}")
print("Saved to data/")
