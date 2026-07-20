"""
Generate publication figures + results table for the IEEE paper.
Reuses the simulation machinery of decoherence_simulation_psd.py.
Outputs: fig_traces.pdf, fig_psd.pdf, fig_decay.pdf, fig_scaling.pdf,
         fig_sequences.pdf, results_table.tex, results_summary.txt
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from scipy.signal import welch

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 1.0,
})
COL = 3.4          # IEEE single column width (inches)
DBL = 7.1          # double column width

GAMMA = 2 * np.pi * 28e9
DT = 2.5e-6
CSV_FILE = "Assignment1_NoiseSamples_CVS_short.csv"

# ---------------- load ----------------
with open(CSV_FILE, encoding="utf-8-sig") as f:
    rows = list(csv.reader(f))
time_axis = np.array([float(x) for x in rows[0][1:]])
B = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
n_traces, n_samples = B.shape
sigma_meas = B.std()
C = GAMMA * DT * np.concatenate([np.zeros((n_traces, 1)),
                                 np.cumsum(B, axis=1)], axis=1)

# ---------------- propagation ----------------
sx = np.array([[0, 1], [1, 0]], dtype=complex)
Rx = lambda th: np.cos(th/2)*np.eye(2) - 1j*np.sin(th/2)*sx
RX90, RX180 = Rx(np.pi/2), Rx(np.pi)

def Uz(phi):
    U = np.zeros(phi.shape + (2, 2), dtype=complex)
    U[..., 0, 0] = np.exp(-1j*phi/2); U[..., 1, 1] = np.exp(1j*phi/2)
    return U

def evolve(segs):
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
    rho = np.broadcast_to(RX90 @ rho0 @ RX90.conj().T,
                          (len(segs[0]), 2, 2)).copy()
    for j, ph in enumerate(segs):
        U = Uz(ph)
        rho = U @ rho @ U.conj().transpose(0, 2, 1)
        if j < len(segs) - 1:
            rho = RX180 @ rho @ RX180.conj().T
    return rho.mean(axis=0)

def pulse_edges(tau_idx, n_pi):
    if n_pi == 0:
        return np.array([0, tau_idx])
    inner = np.round(np.arange(1, 2*n_pi, 2) * tau_idx / (2*n_pi)).astype(int)
    return np.concatenate([[0], inner, [tau_idx]])

def coherence(n_pi, idx):
    W = np.empty(len(idx))
    for m, k in enumerate(idx):
        e = pulse_edges(k, n_pi)
        segs = [C[:, e[j+1]] - C[:, e[j]] for j in range(len(e)-1)]
        W[m] = 2*np.abs(evolve(segs)[0, 1])
    return W

sequences = [("Ramsey", 0), ("Hahn echo", 1),
             ("CPMG-2", 2), ("CPMG-4", 4), ("CPMG-8", 8)]
results = {}
for name, n_pi in sequences:
    step = max(1, 2*max(n_pi, 1))
    idx = np.arange(step, n_samples+1, step)
    results[name] = (idx*DT, coherence(n_pi, idx))

decay = lambda t, T2, p: np.exp(-(t/T2)**p)
FLOOR = 0.15
fits, errs = {}, {}
for name, (T, W) in results.items():
    fb = np.argmax(W < FLOOR) if (W < FLOOR).any() else len(W)
    sel = slice(0, max(fb, 5))
    g = T[sel][np.argmin(np.abs(W[sel]-np.exp(-1)))]
    po, pc = curve_fit(decay, T[sel], W[sel], p0=[g, 2.0],
                       bounds=([1e-7, 0.5], [1.0, 6.0]))
    fits[name], errs[name] = po, np.sqrt(np.diag(pc))

# ---------------- PSD ----------------
fs = 1/DT
f_w, S_w = welch(B, fs=fs, nperseg=n_samples, detrend=False, axis=1)
S_welch = S_w.mean(axis=0)
nfft = 2*n_samples
Bf = np.fft.rfft(B - B.mean(axis=1, keepdims=True), n=nfft, axis=1)
ac = np.fft.irfft(np.abs(Bf)**2, n=nfft, axis=1)[:, :n_samples]/n_samples
acm = ac.mean(axis=0)
nl = n_samples//4
lw = np.hanning(2*nl)[nl:]
acs = np.concatenate([acm[:nl]*lw, np.zeros(1), (acm[:nl]*lw)[:0:-1]])
S_wk = 2*DT*np.real(np.fft.rfft(acs))
f_wk = np.fft.rfftfreq(len(acs), DT)

lorentz = lambda f, s2, tc: 4*s2*tc/(1 + (2*np.pi*f*tc)**2)
m = f_w > 0
popt_S, _ = curve_fit(lambda f, s2, tc: np.log(lorentz(f, s2, tc)),
                      f_w[m], np.log(S_welch[m]),
                      p0=[sigma_meas**2, 1e-3],
                      bounds=([1e-18, 1e-6], [1e-10, 10.0]))
sig2_fit, tau_c_fit = popt_S
S0_fit = 4*sig2_fit*tau_c_fit        # low-frequency plateau amplitude

# ---------------- filter-function predictions ----------------
omega = np.logspace(0.5, 7.5, 4000)
S_om = GAMMA**2 * 2*sig2_fit*tau_c_fit/(1 + (omega*tau_c_fit)**2)

def chi(tau, n_pi):
    if n_pi == 0:
        edges = np.array([0.0, tau])
    else:
        edges = np.concatenate([[0.0],
                                np.arange(1, 2*n_pi, 2)*tau/(2*n_pi), [tau]])
    signs = (-1.0)**np.arange(len(edges)-1)
    e = np.exp(1j*np.outer(omega, edges))
    ft = (e[:, 1:] - e[:, :-1]) @ signs / (1j*omega)
    return np.trapezoid(S_om*np.abs(ft)**2, omega)/(2*np.pi)

pred, T2_pred = {}, {}
for name, n_pi in sequences:
    T = results[name][0]
    taus = np.logspace(np.log10(T[0]),
                       np.log10(min(4*fits[name][0], T[-1])), 60)
    ch = np.array([chi(t, n_pi) for t in taus])
    pred[name] = (taus, np.exp(-ch))
    T2_pred[name] = np.interp(1.0, ch, taus)

# ================= FIGURES =================
colors = plt.cm.viridis(np.linspace(0, 0.85, len(sequences)))

# Fig 1: noise traces
fig, ax = plt.subplots(figsize=(COL, 2.1))
for i in range(4):
    ax.plot(time_axis*1e3, B[i]*1e9, lw=0.5, label=f"Trace {i+1}")
ax.set_xlabel("Time (ms)"); ax.set_ylabel(r"$B_z$ (nT)")
ax.legend(ncol=2, frameon=False)
fig.savefig("fig_traces.pdf"); plt.close(fig)

# Fig 2: pulse sequence schematic
fig, ax = plt.subplots(figsize=(COL, 1.9))
seq_draw = [("Ramsey", 0), ("Hahn echo", 1), ("CPMG-4", 4)]
for row, (name, n_pi) in enumerate(seq_draw):
    y = -row*1.0
    ax.plot([0, 1], [y, y], color="0.7", lw=0.8)
    ax.add_patch(Rectangle((-0.012, y-0.18), 0.024, 0.36,
                           color="tab:blue"))
    ax.add_patch(Rectangle((0.988, y-0.18), 0.024, 0.36,
                           color="tab:blue"))
    if n_pi:
        for j in range(1, 2*n_pi, 2):
            xp = j/(2*n_pi)
            ax.add_patch(Rectangle((xp-0.014, y-0.3), 0.028, 0.6,
                                   color="tab:red"))
    ax.text(-0.09, y, name, ha="right", va="center", fontsize=7)
ax.text(0.0, 0.55, r"$R_x(\pi/2)$", ha="center", fontsize=6.5,
        color="tab:blue")
ax.text(1.0, 0.55, r"$R_x(\pi/2)$", ha="center", fontsize=6.5,
        color="tab:blue")
ax.text(0.5, 0.55, r"$R_x(\pi)$", ha="center", fontsize=6.5, color="tab:red")
ax.annotate("", xy=(0, -2.65), xytext=(1, -2.65),
            arrowprops=dict(arrowstyle="<->", lw=0.7))
ax.text(0.5, -2.9, r"total sequence length $\tau$", ha="center", fontsize=7)
ax.set_xlim(-0.42, 1.12); ax.set_ylim(-3.1, 0.85); ax.axis("off")
fig.savefig("fig_sequences.pdf"); plt.close(fig)

# Fig 3: PSD
fig, ax = plt.subplots(figsize=(COL, 2.4))
ax.loglog(f_w[1:], S_welch[1:], lw=0.8, label="Welch (100-trace avg.)")
ax.loglog(f_wk[1:], np.abs(S_wk[1:]), lw=0.8, alpha=0.6,
          label="FFT of autocorrelation")
ff = np.logspace(np.log10(f_w[1]), np.log10(f_w[-1]), 300)
ax.loglog(ff, lorentz(ff, sig2_fit, tau_c_fit), "k--", lw=0.9,
          label="Lorentzian fit")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel(r"$S_B(f)$  (T$^2$/Hz)")
ax.legend(frameon=False, loc="lower left")
fig.savefig("fig_psd.pdf"); plt.close(fig)

# Fig 4: decay curves (double column)
fig, ax = plt.subplots(figsize=(DBL, 2.9))
for (name, _), c in zip(sequences, colors):
    T, W = results[name]; T2, p = fits[name]
    ax.plot(T*1e6, W, ".", ms=1.8, color=c, alpha=0.45)
    tf = np.linspace(T[0], min(3.5*T2, T[-1]), 400)
    ax.plot(tf*1e6, decay(tf, T2, p), "-", color=c, lw=1.3,
            label=rf"{name}: $T_2$={T2*1e6:.0f}$\,\mu$s, $p$={p:.2f}")
    ta, Wp = pred[name]
    ax.plot(ta*1e6, Wp, "--", color="k", lw=0.7, alpha=0.75)
ax.plot([], [], "k--", lw=0.7, label="filter-function prediction")
ax.axhline(np.exp(-1), color="gray", ls=":", lw=0.7)
ax.text(0.995, np.exp(-1)+0.03, r"$1/e$", color="gray", fontsize=6.5,
        transform=ax.get_yaxis_transform(), ha="right")
ax.set_xscale("log"); ax.set_ylim(-0.05, 1.06)
ax.set_xlabel(r"Total sequence length $\tau$ ($\mu$s)")
ax.set_ylabel(r"Coherence $W(\tau)=2|\langle\rho_{01}\rangle|$")
ax.legend(frameon=False, loc="lower left", ncol=2)
fig.savefig("fig_decay.pdf"); plt.close(fig)

# Fig 5: CPMG scaling
fig, ax = plt.subplots(figsize=(COL, 2.2))
Npi = np.array([1, 2, 4, 8])
T2N = np.array([fits[n][0] for n in ["Hahn echo", "CPMG-2", "CPMG-4",
                                     "CPMG-8"]])
sl, ic = np.polyfit(np.log(Npi), np.log(T2N), 1)
ax.loglog(Npi, T2N*1e6, "o", ms=4, color="tab:blue", label="simulation")
nn = np.logspace(0, np.log10(9), 50)
ax.loglog(nn, np.exp(ic)*nn**sl*1e6, "-", lw=0.9, color="tab:blue",
          label=rf"fit: $T_2 \propto N^{{{sl:.2f}}}$")
ax.loglog(nn, T2N[0]*nn**(2/3)*1e6, "k--", lw=0.8,
          label=r"$N^{2/3}$ (Lorentzian noise)")
ax.set_xlabel(r"Number of $\pi$ pulses $N$")
ax.set_ylabel(r"$T_2$ ($\mu$s)")
ax.legend(frameon=False, loc="upper left")
fig.savefig("fig_scaling.pdf"); plt.close(fig)

# ================= TABLE + SUMMARY =================
with open("results_table.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\hline\n")
    f.write("\\textbf{Sequence} & \\textbf{$N_\\pi$} & "
            "\\textbf{$T_2$ ($\\mu$s)} & \\textbf{$p$} & "
            "\\textbf{$T_2^{\\rm FF}$ ($\\mu$s)}\\\\\n\\hline\n")
    for name, n_pi in sequences:
        T2, p = fits[name]; eT2, ep = errs[name]
        f.write(f"{name} & {n_pi} & ${T2*1e6:.0f} \\pm {eT2*1e6:.0f}$ & "
                f"${p:.2f} \\pm {ep:.2f}$ & {T2_pred[name]*1e6:.0f}\\\\\n")
    f.write("\\hline\n\\end{tabular}\n")

with open("results_summary.txt", "w") as f:
    w = lambda s: (print(s), f.write(s+"\n"))
    w(f"traces = {n_traces}, samples = {n_samples}, dt = {DT*1e6} us")
    w(f"record length = {n_samples*DT*1e3:.2f} ms")
    w(f"sigma_B (measured) = {sigma_meas*1e9:.1f} nT")
    w(f"mean field offset  = {B.mean()*1e9:.2f} nT")
    w(f"quasi-static T2* estimate = "
      f"{np.sqrt(2)/(GAMMA*sigma_meas)*1e6:.1f} us")
    w(f"Lorentzian fit: sigma = {np.sqrt(sig2_fit)*1e9:.1f} nT, "
      f"tau_c = {tau_c_fit*1e3:.1f} ms, S0 = {S0_fit:.3e} T^2/Hz")
    w(f"sqrt(int S df) = {np.sqrt(np.trapezoid(S_welch, f_w))*1e9:.1f} nT")
    w(f"PSD band = {f_w[1]:.1f} Hz to {f_w[-1]/1e3:.1f} kHz")
    w(f"CPMG scaling exponent = {sl:.2f}")
    for name, _ in sequences:
        T2, p = fits[name]; eT2, ep = errs[name]
        w(f"{name:10s} T2 = {T2*1e6:8.1f} +/- {eT2*1e6:4.1f} us, "
          f"p = {p:.2f} +/- {ep:.2f}, FF pred = {T2_pred[name]*1e6:.1f} us")
print("\nFigures + results_table.tex written.")
