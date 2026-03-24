"""
Model B – Coupling disorder
Compares uniform, random-sign, and Gaussian-random V_{1k}.
Also tests randomizing V_01 to show its qualitatively different effect.
Fixed N=200, Delta_E=10, V_1k_rms=0.05, V_01=0.05, E0=E1=0.
"""

import numpy as np
import os

# ── parameters ────────────────────────────────────────────────────────────────
N       = 200
Delta_E = 10.0
V_1k    = 0.05
V_01    = 0.05
E0      = 0.0
E1      = 0.0
N_seeds = 1000

rho   = N / Delta_E
Gamma = 2 * np.pi * V_1k**2 * rho
t_max = 5.0 / Gamma
Nt    = 2000
times = np.linspace(0, t_max, Nt)

# ── helpers ───────────────────────────────────────────────────────────────────
def build_H_B(N, Delta_E, V_1k_arr, V_01, E0=0.0, E1=0.0):
    dim = N + 2
    H = np.zeros((dim, dim))
    H[0, 0] = E0;  H[1, 1] = E1
    H[0, 1] = V_01;  H[1, 0] = V_01
    E_bath = np.linspace(E1 - Delta_E / 2, E1 + Delta_E / 2, N)
    np.fill_diagonal(H[2:, 2:], E_bath)
    H[1, 2:] = V_1k_arr;  H[2:, 1] = V_1k_arr
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
    return P0, P1

# ── 1. Uniform V_{1k} ────────────────────────────────────────────────────────
P0_unif, P1_unif = propagate_B(
    build_H_B(N, Delta_E, np.full(N, V_1k), V_01, E0, E1), times)

# ── 2. Random-sign V_{1k} (only bath couplings randomised) ───────────────────
P0_rs = np.zeros((N_seeds, len(times)))
P1_rs = np.zeros((N_seeds, len(times)))
for s in range(N_seeds):
    rng = np.random.default_rng(s)
    V_arr = rng.choice([-1.0, 1.0], N) * V_1k
    P0_rs[s], P1_rs[s] = propagate_B(
        build_H_B(N, Delta_E, V_arr, V_01, E0, E1), times)

# ── 3. Gaussian V_{1k} ───────────────────────────────────────────────────────
P0_gauss = np.zeros((N_seeds, len(times)))
P1_gauss = np.zeros((N_seeds, len(times)))
for s in range(N_seeds):
    rng = np.random.default_rng(s + 100)
    V_arr = rng.normal(0, V_1k, N)
    P0_gauss[s], P1_gauss[s] = propagate_B(
        build_H_B(N, Delta_E, V_arr, V_01, E0, E1), times)

# ── 4. Randomised V_01 (sign flip) – qualitatively distinct ──────────────────
# Note: with V_01 = +V or -V the interference pattern in P_0, P_1 changes.
P0_rv01 = np.zeros((N_seeds, len(times)))
P1_rv01 = np.zeros((N_seeds, len(times)))
for s in range(N_seeds):
    rng = np.random.default_rng(s + 200)
    V01_s = rng.choice([-1.0, 1.0]) * V_01
    P0_rv01[s], P1_rv01[s] = propagate_B(
        build_H_B(N, Delta_E, np.full(N, V_1k), V01_s, E0, E1), times)

# ── save ──────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
np.save("data/t.npy",         times)
np.save("data/P0_unif.npy",   P0_unif)
np.save("data/P1_unif.npy",   P1_unif)
np.save("data/PK_unif.npy",   1.0 - P0_unif - P1_unif)
np.save("data/P0_rs.npy",     P0_rs)
np.save("data/P1_rs.npy",     P1_rs)
np.save("data/PK_rs.npy",     1.0 - P0_rs - P1_rs)
np.save("data/P0_gauss.npy",  P0_gauss)
np.save("data/P1_gauss.npy",  P1_gauss)
np.save("data/PK_gauss.npy",  1.0 - P0_gauss - P1_gauss)
np.save("data/P0_rv01.npy",   P0_rv01)
np.save("data/P1_rv01.npy",   P1_rv01)
np.save("data/params.npy",    np.array([N, Delta_E, V_1k, V_01, E0, E1, Gamma]))
print(f"Gamma = {Gamma:.4f}")
print("Saved to data/")
