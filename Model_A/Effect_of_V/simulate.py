"""
Model A – Effect of V_{1k} (coupling strength)
Varies V at fixed N=200, Delta_E=10.
Probes underdamped → overdamped transition and FGR breakdown.
"""

import numpy as np
import os
from pathlib import Path

# ── parameters ────────────────────────────────────────────────────────────────
N       = 200
Delta_E = 10.0
E1      = 0.0
V_list  = [0.05, 0.1, 0.15]        # Gamma ranges from weak to ~bandwidth

rho = N / Delta_E

# use a fixed window so outputs are comparable across runs
t_max = 20.0
Nt    = 3000
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

expected_tags = {str(V).replace(".", "p") for V in V_list}
for path in Path("data").glob("P1_V*.npy"):
    tag = path.stem.removeprefix("P1_V")
    if tag not in expected_tags:
        path.unlink()

for V in V_list:
    Gamma_V = 2 * np.pi * V**2 * rho
    P1 = propagate(build_H(N, Delta_E, V, E1), times)
    tag = str(V).replace(".", "p")
    np.save(f"data/P1_V{tag}.npy", P1)
    print(f"V={V:.3f}  Gamma={Gamma_V:.4f}  Gamma/Delta_E={Gamma_V/Delta_E:.3f}")

np.save("data/V_list.npy", np.array(V_list))
np.save("data/params.npy", np.array([N, Delta_E, E1, rho]))
print("Saved to data/")
