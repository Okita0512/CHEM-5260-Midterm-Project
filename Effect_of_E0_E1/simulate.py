"""
Model B – Resonance effect: varying Delta_01 = E0 - E1
Uses an overdamped upstream coupling V_01=0.04 (< Gamma/4).
Both positive and negative detunings are included to verify symmetry.
Extracts the effective decay rate of P_0(t) for panel C.
"""

import numpy as np
import os

# ── parameters ────────────────────────────────────────────────────────────────
N        = 200
Delta_E  = 10.0
V_1k     = 0.05
V_01     = 0.04          # overdamped: V_01 < Gamma/4
E1       = 0.0
Delta01_list = [-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0]

rho   = N / Delta_E
Gamma = 2 * np.pi * V_1k**2 * rho   # ≈ 0.3142

# Long enough to resolve exponential decay: gamma_0 = 4*V_01^2/Gamma ~ 0.020
t_max = 100.0
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
    P0 = np.zeros(len(times));  P1 = np.zeros(len(times))
    for i, t in enumerate(times):
        psi_t = vecs @ (np.exp(-1j * vals * t) * c)
        P0[i] = abs(psi_t[0])**2
        P1[i] = abs(psi_t[1])**2
    return P0, P1, 1.0 - P0 - P1

def extract_rate(t, P0):
    """
    Fit log(P0) to a line over the latter half of the time window.
    Returns the slope magnitude (population decay rate gamma).
    """
    mask = (t > t[-1] / 2) & (P0 > 1e-6)
    if mask.sum() < 10:
        return np.nan
    slope = np.polyfit(t[mask], np.log(P0[mask]), 1)[0]
    return -slope   # positive decay rate

# ── run ───────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
np.save("data/t.npy", times)

rates = []
for d01 in Delta01_list:
    E0 = E1 + d01
    P0, P1, PK = propagate_B(build_H_B(N, Delta_E, V_1k, V_01, E0, E1), times)
    tag = str(d01).replace(".", "p").replace("-", "m")
    np.save(f"data/P0_d{tag}.npy", P0)
    np.save(f"data/P1_d{tag}.npy", P1)
    np.save(f"data/PK_d{tag}.npy", PK)
    gamma = extract_rate(times, P0)
    rates.append(gamma)
    print(f"Delta_01={d01:+.2f}   gamma_eff={gamma:.5f}")

np.save("data/Delta01_list.npy", np.array(Delta01_list))
np.save("data/rates.npy",        np.array(rates))
np.savez("data/params.npz",
         N=N, Delta_E=Delta_E, V_1k=V_1k, V_01=V_01, E1=E1, Gamma=Gamma)
print(f"\nGamma={Gamma:.4f}  Gamma/4={Gamma/4:.5f}  gamma_0=4*V_01^2/Gamma={4*V_01**2/Gamma:.5f}")
print("Saved to data/")
