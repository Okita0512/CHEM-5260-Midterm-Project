"""
Model A – Effect of Delta_E
Varies the bath bandwidth at fixed N=200, V=0.05.
Saves one dataset per Delta_E value in data/.
"""

import numpy as np
import os

# ── parameters ────────────────────────────────────────────────────────────────
N          = 200
V          = 0.05
E1         = 0.0
DeltaE_list = [2.0, 5.0, 10.0, 20.0]

# time range based on reference (Delta_E = 10)
Gamma_ref = 2 * np.pi * V**2 * (N / 10.0)
t_max = 5.0 / Gamma_ref
Nt    = 2000
times = np.linspace(0, t_max, Nt)

# ── helpers ───────────────────────────────────────────────────────────────────
def build_H(N, Delta_E, V, E1=0.0):
    H = np.zeros((N + 1, N + 1))
    H[0, 0] = E1
    np.fill_diagonal(H[1:, 1:],
                     np.linspace(E1 - Delta_E / 2, E1 + Delta_E / 2, N))
    H[0, 1:] = V;  H[1:, 0] = V
    return H

def propagate(H, times):
    vals, vecs = np.linalg.eigh(H)
    psi0 = np.zeros(H.shape[0]);  psi0[0] = 1.0
    c = vecs.T @ psi0
    return np.array([abs(vecs[0] @ (np.exp(-1j * vals * t) * c))**2
                     for t in times])

# ── run ───────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
np.save("data/t.npy", times)

for dE in DeltaE_list:
    rho_dE   = N / dE
    Gamma_dE = 2 * np.pi * V**2 * rho_dE
    P1 = propagate(build_H(N, dE, V, E1), times)
    tag = str(dE).replace(".", "p")
    np.save(f"data/P1_dE{tag}.npy", P1)
    print(f"Delta_E={dE:5.1f}  rho={rho_dE:.1f}  Gamma={Gamma_dE:.4f}")

np.save("data/DeltaE_list.npy", np.array(DeltaE_list))
np.save("data/params.npy", np.array([N, V, E1]))
print("Saved to data/")
