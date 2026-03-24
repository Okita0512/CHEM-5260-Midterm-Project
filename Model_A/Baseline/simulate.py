"""
Model A – Baseline dynamics
Parameters: N=200, Delta_E=10, V=0.05, E1=0
Saves: data/t.npy, data/P1.npy, data/PK.npy, data/FGR.npy
"""

import numpy as np
import os

# ── parameters ────────────────────────────────────────────────────────────────
N       = 200
Delta_E = 10.0
V       = 0.05
E1      = 0.0

rho   = N / Delta_E                       # density of states
Gamma = 2 * np.pi * V**2 * rho            # FGR decay rate
t_max = 6.0 / Gamma                       # ~6 decay lengths
Nt    = 2000
times = np.linspace(0, t_max, Nt)

# ── build Hamiltonian ─────────────────────────────────────────────────────────
def build_H(N, Delta_E, V, E1=0.0):
    dim = N + 1
    H = np.zeros((dim, dim))
    H[0, 0] = E1
    E_bath = np.linspace(E1 - Delta_E / 2, E1 + Delta_E / 2, N)
    np.fill_diagonal(H[1:, 1:], E_bath)
    H[0, 1:] = V
    H[1:, 0] = V
    return H

# ── time propagation ──────────────────────────────────────────────────────────
def propagate(H, times):
    vals, vecs = np.linalg.eigh(H)
    psi0 = np.zeros(H.shape[0])
    psi0[0] = 1.0                         # start in doorway state |1>
    c = vecs.T @ psi0                     # expansion coefficients
    P1 = np.array([abs(vecs[0] @ (np.exp(-1j * vals * t) * c))**2
                   for t in times])
    return P1

# ── run ───────────────────────────────────────────────────────────────────────
H  = build_H(N, Delta_E, V, E1)
P1 = propagate(H, times)
PK  = 1.0 - P1
FGR = np.exp(-Gamma * times)

print(f"rho = {rho:.2f},  Gamma = {Gamma:.4f},  t_max = {t_max:.2f}")
print(f"Hierarchy check:  1/rho = {1/rho:.3f}  <  Gamma = {Gamma:.4f}  <<  Delta_E = {Delta_E}")

# ── save ──────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
np.save("data/t.npy",   times)
np.save("data/P1.npy",  P1)
np.save("data/PK.npy",  PK)
np.save("data/FGR.npy", FGR)
np.save("data/params.npy", np.array([N, Delta_E, V, E1, Gamma]))
print("Saved to data/")
