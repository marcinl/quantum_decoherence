"""
Ensemble spin-dephasing simulation from measured magnetic-field noise traces,
WITH noise power spectral density (PSD) analysis.

Physics
-------
Each spin sees an effective field noise B_i(t) along z (pure dephasing):
    H_i(t) = -gamma * B_i(t) * S_z
Pulses are instantaneous rotations. Sequences (total sequence length tau):

  Ramsey : Rx(pi/2) - free(tau)                                   -> T2*
  Hahn   : Rx(pi/2) - free(tau/2) - Rx(pi) - free(tau/2)          -> T2
  CPMG-N : pi pulses at tau*(2j-1)/(2N), j = 1..N  (Hahn = CPMG-1)

The state is propagated as a 2x2 density matrix for EVERY noise trace using
explicit rotation operators and U(phi) = exp(-i phi sigma_z / 2),
phi = gamma * integral B dt.  Ensemble coherence: W = 2 |<rho_01>|.

PSD analysis
------------
The one-sided field PSD S_B(f) is estimated two independent ways:
  (a) Welch's method, averaged over the 100 traces;
  (b) Wiener-Khinchin: FFT of the ensemble-averaged autocorrelation.
S_B(f) is fitted with a Lorentzian (Ornstein-Uhlenbeck noise)
    S_B(f) = 4 sigma^2 tau_c / (1 + (2 pi f tau_c)^2)
and the fitted (sigma, tau_c) are fed into the Gaussian filter-function
prediction  W(tau) = exp[-chi(tau)],
    chi(tau) = (1/2pi) * int_0^inf  S_omega(w) |f~(w; tau)|^2 dw,
    S_omega(w) = gamma^2 * 2 sigma^2 tau_c / (1 + w^2 tau_c^2)   (two-sided),
where f~(w) is the Fourier transform of the +/-1 pulse toggling function.
These first-principles predictions are overlaid on the simulated decays.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import welch

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
GAMMA = 2 * np.pi * 28e9      # rad / (T s)
DT = 2.5e-6                   # s
CSV_FILE = "Qubit_Ensamble_NoiseSamples_CVS_short.csv"

# ----------------------------------------------------------------------
# Load noise traces
# ----------------------------------------------------------------------
with open(CSV_FILE, encoding="utf-8-sig") as f:
    rows = list(csv.reader(f))
time_axis = np.array([float(x) for x in rows[0][1:]])
B = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
n_traces, n_samples = B.shape
sigma_meas = B.std()
print(f"Loaded {n_traces} traces x {n_samples} samples, dt = {DT*1e6:.2f} us")
print(f"Noise: mean = {B.mean():.3e} T, std = {sigma_meas:.3e} T")
print(f"Quasi-static estimate T2* ~ sqrt(2)/(gamma*sigma) = "
      f"{np.sqrt(2)/(GAMMA*sigma_meas)*1e6:.1f} us")

# Cumulative phase integral per trace
C = GAMMA * DT * np.concatenate(
    [np.zeros((n_traces, 1)), np.cumsum(B, axis=1)], axis=1)

# ----------------------------------------------------------------------
# Operators and density-matrix propagation
# ----------------------------------------------------------------------
sx = np.array([[0, 1], [1, 0]], dtype=complex)

def Rx(theta):
    return np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * sx

def Uz(phi):
    U = np.zeros(phi.shape + (2, 2), dtype=complex)
    U[..., 0, 0] = np.exp(-1j * phi / 2)
    U[..., 1, 1] = np.exp(+1j * phi / 2)
    return U

RX90, RX180 = Rx(np.pi / 2), Rx(np.pi)

def evolve_ensemble(segment_phases):
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
    rho = np.broadcast_to(RX90 @ rho0 @ RX90.conj().T,
                          (len(segment_phases[0]), 2, 2)).copy()
    for j, phis in enumerate(segment_phases):
        U = Uz(phis)
        rho = U @ rho @ U.conj().transpose(0, 2, 1)
        if j < len(segment_phases) - 1:
            rho = RX180 @ rho @ RX180.conj().T
    return rho.mean(axis=0)

def coherence_curve(n_pi, eval_idx):
    W = np.empty(len(eval_idx))
    for m, k in enumerate(eval_idx):
        if n_pi == 0:
            segs = [C[:, k] - C[:, 0]]
        else:
            edges = np.round(np.arange(1, 2 * n_pi, 2) * k
                             / (2 * n_pi)).astype(int)
            edges = np.concatenate([[0], edges, [k]])
            segs = [C[:, edges[j + 1]] - C[:, edges[j]]
                    for j in range(len(edges) - 1)]
        W[m] = 2 * np.abs(evolve_ensemble(segs)[0, 1])
    return W

sequences = [("Ramsey", 0), ("Hahn echo", 1),
             ("CPMG-2", 2), ("CPMG-4", 4), ("CPMG-8", 8)]
results = {}
for name, n_pi in sequences:
    step = max(1, 2 * max(n_pi, 1))
    eval_idx = np.arange(step, n_samples + 1, step)
    results[name] = (eval_idx * DT, coherence_curve(n_pi, eval_idx))

# ----------------------------------------------------------------------
# Empirical fits  W = exp[-(tau/T2)^p]
# ----------------------------------------------------------------------
def decay(t, T2, p):
    return np.exp(-(t / T2) ** p)

FLOOR = 0.15
fits = {}
print("\n--- Empirical fits  W = exp[-(tau/T2)^p] ---")
for name, (T, W) in results.items():
    first_below = np.argmax(W < FLOOR) if (W < FLOOR).any() else len(W)
    sel = slice(0, max(first_below, 5))
    T2_guess = T[sel][np.argmin(np.abs(W[sel] - np.exp(-1)))]
    popt, pcov = curve_fit(decay, T[sel], W[sel], p0=[T2_guess, 2.0],
                           bounds=([1e-7, 0.5], [1.0, 6.0]))
    fits[name] = popt
    perr = np.sqrt(np.diag(pcov))
    print(f"{name:10s}: T2 = {popt[0]*1e6:8.1f} +/- {perr[0]*1e6:5.1f} us,"
          f"  p = {popt[1]:.2f} +/- {perr[1]:.2f}")

# ======================================================================
# NOISE POWER SPECTRAL DENSITY
# ======================================================================
fs = 1.0 / DT

# --- (a) Welch's method, averaged over traces (one-sided, T^2/Hz) ---
# One full-length segment per trace, ensemble-averaged over the 100 traces.
# detrend=False keeps the low-frequency/quasi-static power (most of the
# variance), which per-segment detrending would otherwise remove.
f_w, S_w = welch(B, fs=fs, nperseg=n_samples, detrend=False, axis=1)
S_welch = S_w.mean(axis=0)

# --- (b) Wiener-Khinchin: FFT of the averaged autocorrelation ---
# Autocorrelation via FFT (biased estimator), averaged over the ensemble.
# Per-trace means (quasi-static offsets, a delta-function at f=0) are
# subtracted. NOTE: this direct route uses an implicit rectangular data
# window whose sinc^2 sidelobes fall as f^-2 -- parallel to the 1/f^2
# spectrum here -- so leaked low-frequency power gives it a roughly
# constant ~1.7x upward bias across the band (verified against synthetic
# OU noise). The spectral SHAPE agrees with Welch; the Lorentzian fit
# below therefore uses the Welch estimate, whose Hann taper (sidelobes
# ~f^-3) suppresses this leakage.
nfft = 2 * n_samples
Bf = np.fft.rfft(B - B.mean(axis=1, keepdims=True), n=nfft, axis=1)
ac = np.fft.irfft(np.abs(Bf) ** 2, n=nfft, axis=1)[:, :n_samples] / n_samples
ac_mean = ac.mean(axis=0)                      # <B(t)B(t+s)>, s = lag
# One-sided PSD = 2*dt*FFT of the (even) autocorrelation. A Hann taper on the
# lag window (Blackman-Tukey) suppresses the noisy large-lag estimates.
n_lag = n_samples // 4
lag_win = np.hanning(2 * n_lag)[n_lag:]
ac_sym = np.concatenate([ac_mean[:n_lag] * lag_win,
                         np.zeros(1),
                         (ac_mean[:n_lag] * lag_win)[:0:-1]])
S_wk = 2 * DT * np.real(np.fft.rfft(ac_sym))
f_wk = np.fft.rfftfreq(len(ac_sym), DT)

# Correlation time directly from the autocorrelation (1/e crossing)
tau_c_ac = (np.argmax(ac_mean < ac_mean[0] / np.e) * DT
            if (ac_mean < ac_mean[0] / np.e).any() else np.nan)
print(f"\nAutocorrelation 1/e time: "
      f"{tau_c_ac*1e3:.2f} ms" if np.isfinite(tau_c_ac)
      else "\nAutocorrelation does not decay to 1/e within the record")

# --- Lorentzian (Ornstein-Uhlenbeck) fit to the Welch PSD ---
def lorentzian(f, sig2, tau_c):
    return 4 * sig2 * tau_c / (1 + (2 * np.pi * f * tau_c) ** 2)

mask = f_w > 0
popt_S, pcov_S = curve_fit(
    lambda f, s2, tc: np.log(lorentzian(f, s2, tc)),
    f_w[mask], np.log(S_welch[mask]),
    p0=[sigma_meas ** 2, 1e-3], bounds=([1e-18, 1e-6], [1e-10, 10.0]))
sig2_fit, tau_c_fit = popt_S
print(f"Lorentzian fit: sigma_B = {np.sqrt(sig2_fit)*1e9:.1f} nT "
      f"(measured {sigma_meas*1e9:.1f} nT), tau_c = {tau_c_fit*1e3:.2f} ms")
print(f"Check: integral of Welch PSD -> sigma_B = "
      f"{np.sqrt(np.trapezoid(S_welch, f_w))*1e9:.1f} nT")

# ======================================================================
# FILTER-FUNCTION PREDICTIONS  W(tau) = exp[-chi(tau)]
# ======================================================================
omega = np.logspace(0.5, 7.5, 4000)            # rad/s
S_omega = GAMMA**2 * 2 * sig2_fit * tau_c_fit / (1 + (omega * tau_c_fit)**2)

def chi_of_tau(tau, n_pi):
    """chi = (1/2pi) int_0^inf S_omega(w) |f~(w)|^2 dw, f~ from pulse edges."""
    if n_pi == 0:
        edges = np.array([0.0, tau])
    else:
        edges = np.concatenate([[0.0],
                                np.arange(1, 2 * n_pi, 2) * tau / (2 * n_pi),
                                [tau]])
    signs = (-1.0) ** np.arange(len(edges) - 1)
    e_iwt = np.exp(1j * np.outer(omega, edges))          # (n_w, n_edges)
    f_tilde = (e_iwt[:, 1:] - e_iwt[:, :-1]) @ signs / (1j * omega)
    return np.trapezoid(S_omega * np.abs(f_tilde) ** 2, omega) / (2 * np.pi)

pred = {}
print("\n--- Filter-function predictions from the fitted PSD ---")
for name, n_pi in sequences:
    T_sim, _ = results[name]
    T2_emp = fits[name][0]
    taus = np.logspace(np.log10(T_sim[0]), np.log10(min(4 * T2_emp,
                                                        T_sim[-1])), 60)
    Wp = np.exp(-np.array([chi_of_tau(t, n_pi) for t in taus]))
    pred[name] = (taus, Wp)
    T2_pred = np.interp(1.0, np.array([chi_of_tau(t, n_pi) for t in taus]),
                        taus)
    print(f"{name:10s}: predicted T2 = {T2_pred*1e6:8.1f} us "
          f"(simulated {T2_emp*1e6:8.1f} us)")

# ======================================================================
# Plots
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) sample noise traces
ax = axes[0]
for i in range(3):
    ax.plot(time_axis * 1e3, B[i] * 1e9, lw=0.6, label=f"Trace {i+1}")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Effective field noise $B_z$ (nT)")
ax.set_title("(a) Sample noise traces")
ax.legend(fontsize=8)

# (b) PSD
ax = axes[1]
ax.loglog(f_w[1:], S_welch[1:], label="Welch (trace-averaged)")
ax.loglog(f_wk[1:], np.abs(S_wk[1:]), alpha=0.5,
          label="FFT of autocorrelation")
ff = np.logspace(np.log10(f_w[1]), np.log10(f_w[-1]), 300)
ax.loglog(ff, lorentzian(ff, sig2_fit, tau_c_fit), "k--",
          label=(rf"Lorentzian fit: $\sigma_B$ = {np.sqrt(sig2_fit)*1e9:.0f} nT,"
                 rf" $\tau_c$ = {tau_c_fit*1e3:.1f} ms"))
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel(r"$S_B(f)$  (T$^2$/Hz)")
ax.set_title("(b) Noise power spectral density")
ax.legend(fontsize=8)

# (c) coherence decays: simulation, empirical fits, filter-function prediction
ax = axes[2]
colors = plt.cm.viridis(np.linspace(0, 0.85, len(sequences)))
for (name, _), c in zip(sequences, colors):
    T, W = results[name]
    T2, p = fits[name]
    ax.plot(T * 1e6, W, ".", ms=3, color=c, alpha=0.5)
    tf = np.linspace(T[0], min(3.5 * T2, T[-1]), 400)
    ax.plot(tf * 1e6, decay(tf, T2, p), "-", color=c, lw=1.5,
            label=f"{name}: $T_2$ = {T2*1e6:.0f} µs, p = {p:.2f}")
    taus, Wp = pred[name]
    ax.plot(taus * 1e6, Wp, "--", color="k", lw=1, alpha=0.7)
ax.plot([], [], "k--", lw=1, label="filter-function prediction")
ax.axhline(np.exp(-1), color="gray", ls=":", lw=1)
ax.text(0.99, np.exp(-1) + 0.02, "1/e", color="gray",
        transform=ax.get_yaxis_transform(), ha="right")
ax.set_xscale("log")
ax.set_xlabel(r"Total sequence length $\tau$ (µs)")
ax.set_ylabel(r"Coherence $W(\tau) = 2|\langle\rho_{01}\rangle|$")
ax.set_title("(c) Ensemble coherence decay (100 spins)")
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8, loc="lower left")

fig.tight_layout()
fig.savefig("decoherence_decay_psd.png", dpi=160)
print("\nSaved figure: decoherence_decay_psd.png")
