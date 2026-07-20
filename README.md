# Spin Dephasing from Measured Magnetic Field Noise

ELEC4606 postgraduate assignment — simulation of ensemble spin decoherence driven
by experimentally recorded magnetic field noise, with Ramsey, Hahn echo and CPMG
dynamical decoupling sequences.

An ensemble of 100 spin qubits is propagated as density matrices through explicit
rotation operators, each spin driven by its own recorded noise trace. The noise
power spectral density is estimated independently and used to predict the same
decay curves through the filter-function formalism, as a cross-check on the
trajectory-level simulation.

---

## Physics summary

Each recorded trace is taken to be the effective field `B_z,i(t)` seen by one
spin, giving the pure-dephasing Hamiltonian

```
H_i(t) = -γ B_z,i(t) Ŝ_z ,        γ = 2π × 28 GHz/T
```

Because `H_i(t)` commutes with itself at all times, the propagator needs no time
ordering:

```
U_i(t) = exp[ -i φ_i(t) σ_z / 2 ] ,   φ_i(t) = γ ∫₀ᵗ B_z,i(t') dt'
```

Starting from `|0⟩`, an instantaneous `R_x(π/2)` places each spin on the equator;
`R_x(π)` refocusing pulses are applied explicitly between free-evolution segments.
The observable is the ensemble-averaged coherence

```
W(τ) = 2 |⟨ρ₀₁⟩|
```

which decays not because any individual spin loses purity — each stays pure — but
because the phases `φ_i` fan out across the ensemble.

### Key results

| Sequence  | N<sub>π</sub> | T₂ (µs)      | p            | T₂ from PSD (µs) |
|-----------|---------------|--------------|--------------|------------------|
| Ramsey    | 0             | 71.0 ± 0.2   | 2.18 ± 0.02  | 63               |
| Hahn echo | 1             | 1094 ± 1     | 2.93 ± 0.01  | 1060             |
| CPMG-2    | 2             | 1738 ± 2     | 2.79 ± 0.01  | 1680             |
| CPMG-4    | 4             | 2781 ± 15    | 2.52 ± 0.05  | 2666             |
| CPMG-8    | 8             | 4407 ± 32    | 2.68 ± 0.08  | 4235             |

Fit function `W(τ) = exp[-(τ/T₂)^p]`. The last column is a parameter-free
prediction from the measured noise spectrum — it agrees with the simulation to
within 4–10%.

Noise statistics: σ_B = 95.1 nT, record length 19.01 ms, Δt = 2.5 µs.
Decoupling gain scales as T₂ ∝ N^0.67, matching the N^(2/3) law for a
Lorentzian (f⁻²) spectrum.

---

## Repository structure

```
.
├── Qubit_Ensamble_NoiseSamples_CVS_short.csv   input: 100 traces × 7603 samples (trunkated)
├── Qubit_Ensamble_NoiseSamples_CVS.zip        input: full length 100 spin traces
│
├── decoherence_simulation.py                v1: simulation + decay fits only
├── decoherence_simulation_psd.py            v2: + PSD estimation + filter functions
├── make_paper_figures.py                    publication figures + results table
├── build_preview.py                         compiles the LaTeX preview PDF
│
├── spin_decoherence_paper.tex               IEEE conference paper (IEEEtran)
├── results_table.tex                        auto-generated, \input by the paper
├── fig_*.pdf                                figures used by the paper
├── fig_*.png                                same figures, for this README
│
├── spin_decoherence_paper_overleaf.zip      upload-ready bundle for Overleaf
├── spin_decoherence_paper_preview.pdf       locally compiled preview
└── preview/                                 LaTeX build directory
```

### Which simulation script to use

`decoherence_simulation_psd.py` supersedes `decoherence_simulation.py`. The
earlier version is kept because it is the shorter read: it shows the density-matrix
propagation without the spectral analysis layered on top. Both produce identical
coherence times.

### Code layout (`decoherence_simulation_psd.py`)

| Section | What it does |
|---------|--------------|
| Load traces | Reads the CSV; builds `C[i,k] = φ_i(t_k)` as a cumulative sum so the phase in any interval is one subtraction |
| `Rx`, `Uz` | Rotation operator `exp(-iθσ_x/2)` and the diagonal free-evolution propagator |
| `evolve_ensemble` | Propagates all 100 density matrices through one pulse sequence, returns `⟨ρ⟩` |
| `coherence_curve` | Sweeps total sequence length τ, placing π pulses at CPMG positions |
| Empirical fits | Stretched exponential, restricted to `W > 0.15` (the 100-trace statistical floor is ≈0.09) |
| PSD | Welch (trace-averaged) and Wiener–Khinchin (FFT of autocorrelation), then a Lorentzian fit |
| `chi_of_tau` | Filter-function integral `χ(τ)`, giving the parameter-free prediction |

---

## Regenerating everything

Requires Python 3 with `numpy`, `scipy`, `matplotlib`:

```bash
pip install numpy scipy matplotlib
```

### 1. Run the simulation

```bash
python3 decoherence_simulation_psd.py
```

Prints the fitted coherence times, noise statistics and filter-function
predictions; writes `decoherence_decay_psd.png` (a quick 3-panel overview).
Runtime is a couple of seconds.

### 2. Regenerate the paper figures and results table

```bash
python3 make_paper_figures.py
```

Writes `fig_traces.pdf`, `fig_sequences.pdf`, `fig_psd.pdf`, `fig_decay.pdf`,
`fig_scaling.pdf` (sized for IEEE single/double column), plus
`results_table.tex` — which the paper pulls in via `\input`, so the table in the
PDF always matches the last simulation run — and `results_summary.txt`.

To regenerate the PNG copies used in this README:

```bash
for f in traces sequences psd decay scaling; do
  pdftoppm -r 150 -png -singlefile fig_$f.pdf fig_$f
done
```

### 3. Build the paper

**Locally**, if you have MacTeX or TeX Live with `IEEEtran.cls`:

```bash
pdflatex spin_decoherence_paper.tex
pdflatex spin_decoherence_paper.tex     # second pass resolves references
```

**Locally without IEEEtran** — the helper script detects this and substitutes an
article-class shim that approximates the two-column IEEE layout:

```bash
python3 build_preview.py                # → preview/paper_preview.pdf
```

Note: the shim cannot use IEEE's author macros, so it carries its own copy of the
title and author block in the `SHIM_TITLE` constant near the top of
`build_preview.py`. If the author details change in the `.tex`, update that
constant too. This does not apply to the real IEEEtran build.

**On Overleaf** (gives the true IEEE formatting):

1. Open the [IEEE Conference Template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn) → *Open as Template*
2. Upload the contents of `spin_decoherence_paper_overleaf.zip`
3. Set `spin_decoherence_paper.tex` as the main document and compile

The zip contains the `.tex`, `results_table.tex` and the five figure PDFs.

### Rebuilding the Overleaf bundle

```bash
zip spin_decoherence_paper_overleaf.zip \
    spin_decoherence_paper.tex results_table.tex \
    fig_traces.pdf fig_sequences.pdf fig_psd.pdf fig_decay.pdf fig_scaling.pdf
```

---

## Figures

### Fig. 1 — Sample noise traces

![Sample noise traces](fig_traces.png)

Four of the 100 recorded traces of effective field noise. Two features drive
everything that follows: the traces wander smoothly on millisecond timescales
(so the noise is nearly static over a 71 µs Ramsey measurement), and they sit at
markedly different mean offsets (inhomogeneous broadening across the ensemble —
exactly what a spin echo removes).

### Fig. 2 — Pulse sequences

![Pulse sequences](fig_sequences.png)

The three sequence families. Narrow bars are `R_x(π/2)` pulses, wide bars are
`R_x(π)` refocusing pulses; all are instantaneous. Ramsey accumulates phase
uninterrupted, so it is maximally sensitive to slow noise. Hahn echo inverts the
spin at τ/2 so the second half's phase subtracts, cancelling any static field.
CPMG-N generalises this with N pulses at `t_j = (2j−1)τ/2N`.

### Fig. 3 — Noise power spectral density

![Noise PSD](fig_psd.png)

The PSD by both required routes: Welch periodogram averaging across the ensemble
(legitimate here because the noise is stated to be ergodic), and the Fourier
transform of the ensemble-averaged autocorrelation. Both give S_B(f) ∝ f⁻²
across the whole accessible band, 52.6 Hz to 200 kHz, with no visible knee.

The constant ~1.7× offset of the autocorrelation route is spectral leakage, not
a physics difference: its implicit rectangular window has sidelobes falling as
f⁻², parallel to the signal spectrum, so low-frequency power leaks uniformly
across the band. Welch's Hann taper (f⁻³ sidelobes) suppresses this, so the
Welch estimate is used for all quantitative work.

Caveat worth knowing: since no knee is resolved, the Lorentzian fit constrains
only the product σ²τ_c, not τ_c alone. The honest statement is τ_c ≳ 19 ms. This
does not affect the coherence predictions, which probe τ ≪ τ_c and depend only
on that product.

### Fig. 4 — Coherence decay

![Coherence decay](fig_decay.png)

The main result. Points are the density-matrix simulation, coloured lines the
stretched-exponential fits, dashed black lines the parameter-free filter-function
predictions from the measured spectrum. The agreement between the two independent
routes is the central consistency check of the project.

Reading the exponents: Ramsey gives p ≈ 2.2 (Gaussian — the noise is frozen on
that timescale), while the echo gives p ≈ 2.9, the cubic exponent characteristic
of Ornstein–Uhlenbeck noise in the limit τ ≪ τ_c. The scatter below W ≈ 0.1 is
the statistical floor of a 100-spin ensemble, which is why fits are restricted to
points above 0.15.

### Fig. 5 — Decoupling scaling

![CPMG scaling](fig_scaling.png)

Coherence time against pulse number, log-log. The fitted exponent 0.67 is
indistinguishable from the 2/3 predicted for an f⁻² spectrum: with the filter
passband near f ≈ N/2τ, the decoherence integral scales as χ ∝ τ³/N², so setting
χ = 1 gives T₂ ∝ N^(2/3).

Practical reading: each doubling of the pulse count buys only 2^(2/3) ≈ 1.6× in
coherence time — diminishing returns, and in a real experiment the accumulating
pulse errors (assumed absent here) would eventually overwhelm that gain.

---

## Known limitations

- All pulses are instantaneous and error-free, so no error accumulates with pulse
  number. Real finite-duration pulses impose a ceiling on usable N.
- Pure dephasing only. The noise is classical and purely longitudinal, so there
  is no T₁ relaxation and no T₂ ≤ 2T₁ ceiling.
- The 100-trace ensemble sets a coherence floor of ≈0.09, limiting fits to
  roughly the first decade of decay.
- The 19 ms record bounds the lowest resolvable frequency at 52.6 Hz; the longest
  CPMG-8 coherence times reach a non-negligible fraction of the record length.
- Pulse positions are rounded to the 2.5 µs sampling grid.
