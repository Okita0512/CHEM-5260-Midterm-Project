"""
Model B – Effect of N (fixed Gamma)
Varies N at fixed Delta_E=10, V_01=0.05, E0=E1=0.
V_1k is scaled as V_ref*sqrt(N_ref/N) so that Gamma = 2*pi*V_1k^2*rho is
held constant, keeping the bath reorganization energy N*V_1k^2 fixed.
Mirrors the approach of Model A / Effect_of_N (Fig 3).
"""

import numpy as np
import os

# ── parameters ────────────────────────────────────────────────────────────────
Delta_E  = 10.0
V_1k_ref = 0.05
V_01     = 0.04          # overdamped (same regime as Fig 8)
E0       = 0.0
E1       = 0.0
N_list   = [50, 100, 200]

N_ref              = N_list[-1]
hybridization_target = N_ref * V_1k_ref**2   # N * V_1k^2 = const
Gamma_target       = 2 * np.pi * V_1k_ref**2 * (N_ref / Delta_E)

def recurrence_time(N, Delta_E):
    return 2 * np.pi / (Delta_E / (N - 1))

T_rec_list = np.array([recurrence_time(N, Delta_E) for N in N_list])
t_max = 1.2 * T_rec_list.max()
Nt    = int(800 * t_max / T_rec_list.min()) + 1
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

# ── run ───────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
np.save("data/t.npy", times)

V_1k_list = []
for N, T_rec in zip(N_list, T_rec_list):
    V_1k_N = np.sqrt(hybridization_target / N)
    Gamma_N = 2 * np.pi * V_1k_N**2 * (N / Delta_E)
    P0, P1, PK = propagate_B(
        build_H_B(N, Delta_E, V_1k_N, V_01, E0, E1), times)
    np.save(f"data/P0_N{N}.npy", P0)
    np.save(f"data/P1_N{N}.npy", P1)
    np.save(f"data/PK_N{N}.npy", PK)
    V_1k_list.append(V_1k_N)
    print(f"N={N:4d}  V_1k={V_1k_N:.5f}  Gamma={Gamma_N:.4f}  T_rec={T_rec:.2f}")

np.save("data/N_list.npy", np.array(N_list))
np.save("data/V_1k_list.npy", np.array(V_1k_list))
np.save("data/T_rec_list.npy", T_rec_list)
np.savez("data/params.npz",
         Delta_E=Delta_E, V_1k_ref=V_1k_ref, V_01=V_01,
         N_ref=N_ref, Gamma_target=Gamma_target,
         hybridization_target=hybridization_target)
print(f"Fixed: N*V_1k^2 = {hybridization_target:.6f},  Gamma = {Gamma_target:.4f}")
print("Saved to data/")
