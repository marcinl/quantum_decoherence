"""
Ensemble spin-dephasing simulation from measured magnetic-field noise traces.

Physics
-------
Each spin sees an effective field noise B_i(t) along z (pure dephasing):
    H_i(t) = -gamma * B_i(t) * S_z ,   S_z = (hbar/2) sigma_z
Pulses are instantaneous rotations. Sequences (total free-evolution time T):

  Ramsey : Rx(pi/2) - free(T)                                  -> T2*
  Hahn   : Rx(pi/2) - free(T/2) - Rx(pi) - free(T/2)           -> T2
  CPMG-N : Rx(pi/2) - [free(T/2N) - Rx(pi) - free(T/N)]... symmetric

The state is propagated as a 2x2 density matrix rho for EVERY noise trace,
using explicit rotation operators and the free-evolution propagator
    U(phi) = exp(-i phi sigma_z / 2),  phi = gamma * integral B(t) dt.
Ensemble coherence:  W(T) = 2 * | < rho_01 >_traces |.

Coherence times are extracted by fitting W(T) = exp[-(T/T2)^p].
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
GAMMA = 2 * np.pi * 28e9      # rad / (T s)  (electron gyromagnetic ratio)
DT = 2.5e-6                   # s, sampling resolution of the noise traces
CSV_FILE = "Qubit_Ensamble_NoiseSamples_CVS_short.csv"

# ----------------------------------------------------------------------
# Load noise traces  (rows = traces, columns = time samples)
# ----------------------------------------------------------------------
with open(CSV_FILE, encoding="utf-8-sig") as f:
    rows = list(csv.reader(f))
time_axis = np.array([float(x) for x in rows[0][1:]])          # s
B = np.array([[float(x) for x in r[1:]] for r in rows[1:]])     # (100, 7603) T
n_traces, n_samples = B.shape
print(f"Loaded {n_traces} traces x {n_samples} samples, dt = {DT*1e6:.2f} us")
print(f"Noise: mean = {B.mean():.3e} T, std = {B.std():.3e} T")
print(f"Quasi-static estimate T2* ~ sqrt(2)/(gamma*sigma) = "
      f"{np.sqrt(2)/(GAMMA*B.std())*1e6:.1f} us")

# Cumulative phase integral per trace: C[i, k] = gamma * sum_{j<k} B[i,j] * dt
# so the phase accumulated between sample s and e is C[:, e] - C[:, s].
C = GAMMA * DT * np.concatenate(
    [np.zeros((n_traces, 1)), np.cumsum(B, axis=1)], axis=1)   # (100, 7604)

# ----------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------
sx = np.array([[0, 1], [1, 0]], dtype=complex)

def Rx(theta):
    """Rotation about x: exp(-i theta sigma_x / 2)."""
    return (np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * sx)

def Uz(phi):
    """Free evolution exp(-i phi sigma_z / 2) for an array of phases phi.
    Returns stacked 2x2 propagators, shape phi.shape + (2, 2)."""
    U = np.zeros(phi.shape + (2, 2), dtype=complex)
    U[..., 0, 0] = np.exp(-1j * phi / 2)
    U[..., 1, 1] = np.exp(+1j * phi / 2)
    return U

RX90 = Rx(np.pi / 2)
RX180 = Rx(np.pi)

def evolve_ensemble(segment_phases):
    """Propagate rho0 = |0><0| through Rx(pi/2), then alternating free
    evolution / Rx(pi) segments, for all traces at once.

    segment_phases: list of arrays, each shape (n_traces,) - the phase
    gamma*int B dt accumulated in each free-evolution segment.
    Returns ensemble-averaged density matrix (2x2).
    """
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
    rho = np.broadcast_to(RX90 @ rho0 @ RX90.conj().T,
                          (len(segment_phases[0]), 2, 2)).copy()
    for j, phis in enumerate(segment_phases):
        U = Uz(phis)
        rho = U @ rho @ U.conj().transpose(0, 2, 1)
        if j < len(segment_phases) - 1:           # pi pulse between segments
            rho = RX180 @ rho @ RX180.conj().T
    return rho.mean(axis=0)

def coherence_curve(sequence, n_pi, eval_idx):
    """W(T) for a sequence with n_pi refocusing pulses at CPMG positions
    t_j = (2j-1) T / (2 n_pi)  (Hahn = CPMG-1). n_pi = 0 -> Ramsey.
    eval_idx: array of sample indices defining total times T = k*DT."""
    W = np.empty(len(eval_idx))
    for m, k in enumerate(eval_idx):
        if n_pi == 0:
            segs = [C[:, k] - C[:, 0]]
        else:
            # pulse positions rounded to the sampling grid
            edges = np.round(np.arange(1, 2 * n_pi, 2) * k / (2 * n_pi)
                             ).astype(int)
            edges = np.concatenate([[0], edges, [k]])
            segs = [C[:, edges[j + 1]] - C[:, edges[j]]
                    for j in range(len(edges) - 1)]
        rho_avg = evolve_ensemble(segs)
        W[m] = 2 * np.abs(rho_avg[0, 1])
    return W

# ----------------------------------------------------------------------
# Run all sequences
# ----------------------------------------------------------------------
sequences = [("Ramsey", 0), ("Hahn echo", 1),
             ("CPMG-2", 2), ("CPMG-4", 4), ("CPMG-8", 8)]

results = {}
for name, n_pi in sequences:
    step = max(1, 2 * max(n_pi, 1))            # keep pulse grid reasonable
    eval_idx = np.arange(step, n_samples + 1, step)
    T = eval_idx * DT
    W = coherence_curve(name, n_pi, eval_idx)
    results[name] = (T, W)
    print(f"{name}: computed {len(T)} points up to {T[-1]*1e3:.2f} ms")

# ----------------------------------------------------------------------
# Fit W(T) = exp[-(T/T2)^p]  and extract T2 (1/e time)
# ----------------------------------------------------------------------
def decay(t, T2, p):
    return np.exp(-(t / T2) ** p)

FLOOR = 0.15   # statistical floor for 100 traces ~ sqrt(pi)/(2*sqrt(N)) ~ 0.09
fits = {}
print("\n--- Extracted coherence times, fit W = exp[-(t/T2)^p] ---")
for name, (T, W) in results.items():
    mask = W > FLOOR
    # keep only the initial decay (up to first crossing of the floor)
    first_below = np.argmax(~mask) if (~mask).any() else len(W)
    sel = slice(0, max(first_below, 5))
    T2_guess = T[sel][np.argmin(np.abs(W[sel] - np.exp(-1)))]
    try:
        popt, pcov = curve_fit(decay, T[sel], W[sel],
                               p0=[T2_guess, 2.0],
                               bounds=([1e-7, 0.5], [1.0, 6.0]))
        perr = np.sqrt(np.diag(pcov))
        fits[name] = popt
        print(f"{name:10s}: T2 = {popt[0]*1e6:8.1f} +/- {perr[0]*1e6:5.1f} us"
              f",  p = {popt[1]:.2f} +/- {perr[1]:.2f}")
    except RuntimeError:
        fits[name] = None
        print(f"{name:10s}: fit failed")

# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) example noise traces
ax = axes[0]
for i in range(3):
    ax.plot(time_axis * 1e3, B[i] * 1e9, lw=0.6, label=f"Trace {i+1}")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Effective field noise (nT)")
ax.set_title("Sample noise traces")
ax.legend(fontsize=8)

# (b) coherence decays + fits
ax = axes[1]
colors = plt.cm.viridis(np.linspace(0, 0.85, len(sequences)))
for (name, _), c in zip(sequences, colors):
    T, W = results[name]
    ax.plot(T * 1e6, W, ".", ms=3, color=c, alpha=0.6)
    if fits[name] is not None:
        T2, p = fits[name]
        tf = np.linspace(0, min(3.5 * T2, T[-1]), 400)
        ax.plot(tf * 1e6, decay(tf, T2, p), "-", color=c,
                label=f"{name}: $T_2$ = {T2*1e6:.0f} µs, p = {p:.2f}")
ax.axhline(np.exp(-1), color="gray", ls=":", lw=1)
ax.text(0.99, np.exp(-1) + 0.02, "1/e", color="gray",
        transform=ax.get_yaxis_transform(), ha="right")
ax.set_xscale("log")
ax.set_xlabel("Total free-evolution time T (µs)")
ax.set_ylabel(r"Coherence $W(T) = 2|\langle\rho_{01}\rangle|$")
ax.set_title("Ensemble coherence decay (100 spins)")
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("decoherence_decay.png", dpi=160)
print("\nSaved figure: decoherence_decay.png")
