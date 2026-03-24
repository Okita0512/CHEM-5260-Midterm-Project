"""
Model A - Effect of N

Compare different reservoir sizes at fixed bandwidth while keeping the
effective continuum coupling fixed. In this equal-coupling toy model, that
means holding N * V_N^2 constant so the FGR rate Gamma stays the same.

Saves one dataset per N value in data/.
"""

import os

import numpy as np


Delta_E = 10.0
V_ref = 0.05
E1 = 0.0
N_list = [50, 100, 200]

N_ref = N_list[-1]
rho_ref = N_ref / Delta_E
Gamma_target = 2 * np.pi * V_ref**2 * rho_ref
hybridization_target = N_ref * V_ref**2


def bath_energies(N, Delta_E, E1=0.0):
    return np.linspace(E1 - Delta_E / 2, E1 + Delta_E / 2, N)


def recurrence_time(N, Delta_E):
    if N < 2:
        return np.inf
    delta_omega = Delta_E / (N - 1)
    return 2 * np.pi / delta_omega


def build_H(N, Delta_E, V_state, E1=0.0):
    H = np.zeros((N + 1, N + 1))
    H[0, 0] = E1
    np.fill_diagonal(H[1:, 1:], bath_energies(N, Delta_E, E1))
    H[0, 1:] = V_state
    H[1:, 0] = V_state
    return H


def propagate(H, times):
    vals, vecs = np.linalg.eigh(H)
    psi0 = np.zeros(H.shape[0])
    psi0[0] = 1.0
    coeffs = vecs.conj().T @ psi0
    doorway_weights = vecs[0, :] * coeffs
    amplitudes = np.exp(-1j * np.outer(times, vals)) @ doorway_weights
    return np.abs(amplitudes) ** 2


T_rec_list = np.array([recurrence_time(N, Delta_E) for N in N_list], dtype=float)
t_max = 1.20 * np.max(T_rec_list)
points_per_shortest_recurrence = 800
Nt = int(np.ceil(points_per_shortest_recurrence * t_max / np.min(T_rec_list))) + 1
times = np.linspace(0.0, t_max, Nt)


os.makedirs("data", exist_ok=True)
np.save("data/t.npy", times)
np.save("data/N_list.npy", np.array(N_list))
np.save("data/T_rec_list.npy", T_rec_list)

V_list = []
Gamma_list = []

for N, T_rec in zip(N_list, T_rec_list):
    rho_N = N / Delta_E
    V_N = np.sqrt(hybridization_target / N)
    Gamma_N = 2 * np.pi * V_N**2 * rho_N

    P1 = propagate(build_H(N, Delta_E, V_N, E1), times)
    np.save(f"data/P1_N{N}.npy", P1)

    V_list.append(V_N)
    Gamma_list.append(Gamma_N)

    print(
        f"N={N:4d}  V_N={V_N:.5f}  rho={rho_N:.2f}  Gamma={Gamma_N:.4f}  "
        f"T_rec={T_rec:.2f}"
    )

np.save("data/V_list.npy", np.array(V_list))
np.save("data/Gamma_list.npy", np.array(Gamma_list))
np.save("data/FGR.npy", np.exp(-Gamma_target * times))
np.savez(
    "data/params.npz",
    Delta_E=Delta_E,
    V_ref=V_ref,
    E1=E1,
    N_ref=N_ref,
    Gamma_target=Gamma_target,
    hybridization_target=hybridization_target,
    t_max=t_max,
    Nt=Nt,
)

print(
    f"Fixed comparison target: N_ref * V_ref^2 = {hybridization_target:.6f}, "
    f"Gamma = {Gamma_target:.4f}"
)
print(f"Time window: t_max = {t_max:.2f}, Nt = {Nt}")
print("Saved to data/")
