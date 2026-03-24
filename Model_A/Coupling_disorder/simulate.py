"""
Model A – Coupling disorder
Compares uniform, random-sign, and Gaussian-random couplings.
Fixed N=200, Delta_E=10, V_rms=0.05.
Multiple random seeds are used for the stochastic cases.
"""

import numpy as np
import os

# ── parameters ────────────────────────────────────────────────────────────────
N       = 200
Delta_E = 10.0
V       = 0.05          # rms coupling value
E1      = 0.0
N_seeds = 1000          # number of disorder realisations

rho   = N / Delta_E
Gamma = 2 * np.pi * V**2 * rho
t_max = 5.0 / Gamma
Nt    = 2000
times = np.linspace(0, t_max, Nt)

# ── helpers ───────────────────────────────────────────────────────────────────
def build_H(N, Delta_E, V_1k, E1=0.0):
    """V_1k: length-N array of coupling values."""
    H = np.zeros((N + 1, N + 1))
    H[0, 0] = E1
    np.fill_diagonal(H[1:, 1:],
                     np.linspace(E1 - Delta_E / 2, E1 + Delta_E / 2, N))
    H[0, 1:] = V_1k;  H[1:, 0] = V_1k
    return H

def propagate(H, times):
    vals, vecs = np.linalg.eigh(H)
    psi0 = np.zeros(H.shape[0]);  psi0[0] = 1.0
    c = vecs.T @ psi0
    return np.array([abs(vecs[0] @ (np.exp(-1j * vals * t) * c))**2
                     for t in times])

# ── uniform (deterministic) ───────────────────────────────────────────────────
P1_uniform = propagate(build_H(N, Delta_E, np.full(N, V), E1), times)

# ── random-sign (averaged over seeds) ────────────────────────────────────────
P1_rsign = np.zeros((N_seeds, len(times)))
for s in range(N_seeds):
    rng = np.random.default_rng(s)
    V_1k = rng.choice([-1.0, 1.0], N) * V
    P1_rsign[s] = propagate(build_H(N, Delta_E, V_1k, E1), times)

# ── Gaussian-random (averaged over seeds) ────────────────────────────────────
P1_gauss = np.zeros((N_seeds, len(times)))
for s in range(N_seeds):
    rng = np.random.default_rng(s + 100)
    V_1k = rng.normal(0, V, N)
    P1_gauss[s] = propagate(build_H(N, Delta_E, V_1k, E1), times)

# ── save ──────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
np.save("data/t.npy",          times)
np.save("data/P1_uniform.npy", P1_uniform)
np.save("data/P1_rsign.npy",   P1_rsign)    # shape (N_seeds, Nt)
np.save("data/P1_gauss.npy",   P1_gauss)    # shape (N_seeds, Nt)
np.save("data/FGR.npy",        np.exp(-Gamma * times))
np.save("data/params.npy",     np.array([N, Delta_E, V, E1, Gamma]))
print(f"Gamma = {Gamma:.4f},  t_max = {t_max:.2f}")
print("Saved to data/")
