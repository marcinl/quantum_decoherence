#!/usr/bin/env python3
"""
Rebuild paper_preview.pdf from spin_decoherence_paper.tex.

The paper is written for IEEEtran (\\documentclass[conference]{IEEEtran}).
If IEEEtran.cls is available locally this script simply compiles the paper
as-is. If it is not, the script swaps in an article-class shim that
approximates the IEEE two-column layout so the content can still be read.

Usage:   python3 build_preview.py
Output:  preview/paper_preview.pdf
"""

import os
import shutil
import subprocess
import sys

SRC = "spin_decoherence_paper.tex"
OUTDIR = "preview"
ASSETS = ["results_table.tex", "fig_traces.pdf", "fig_sequences.pdf",
          "fig_psd.pdf", "fig_decay.pdf", "fig_scaling.pdf"]

SHIM = r"""\documentclass[10pt,twocolumn,a4paper]{article}
\usepackage[margin=0.7in,columnsep=0.25in]{geometry}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{url}
\usepackage{times}
\newenvironment{IEEEkeywords}%
  {\vspace{2pt}\par\noindent\small\textit{Index Terms}---}{\par}
\renewcommand{\thesection}{\Roman{section}}
\renewcommand{\thesubsection}{\Alph{subsection}}
"""

# Article-class \author cannot contain the IEEEauthorblock macros, so the
# whole title/author block is replaced when the shim is used. Edit here if
# the author details in the paper change.
SHIM_TITLE = r"""\title{\vspace{-2em}Simulating Spin Dephasing from Measured Magnetic Field Noise:\\
Ramsey, Hahn Echo and CPMG Dynamical Decoupling\thanks{Corrections to this report were prepared with the assistance of
Claude Opus 4.8 (Anthropic).}}
\author{Marcin Lubonski, \\
ELEC4606 Postgraduate Assignment Submission. \\ 
School of Electrical Engineering and Telecommunications, University of NSW, Sydney, Australia. \\
Email: marcin.lubonski@gmail.com}
\date{}
"""


def have_ieeetran():
    try:
        r = subprocess.run(["kpsewhich", "IEEEtran.cls"],
                           capture_output=True, text=True)
        return r.returncode == 0 and r.stdout.strip() != ""
    except FileNotFoundError:
        return False


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    if not os.path.exists(SRC):
        sys.exit(f"error: {SRC} not found in {here}")

    os.makedirs(OUTDIR, exist_ok=True)
    for a in ASSETS:
        if os.path.exists(a):
            shutil.copy(a, OUTDIR)

    src = open(SRC).read()

    if have_ieeetran():
        print("IEEEtran.cls found - compiling the paper directly.")
        tex = src
    else:
        print("IEEEtran.cls not found - using the article-class shim.")
        body = src[src.index("\\newcommand{\\ket}"):]
        body = (body[:body.index("\\title{")] + SHIM_TITLE
                + body[body.index("\\maketitle"):])
        tex = SHIM + body

    target = os.path.join(OUTDIR, "paper_preview.tex")
    with open(target, "w") as f:
        f.write(tex)

    # Three passes so cross-references and float placement settle.
    for i in range(3):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "paper_preview.tex"],
            cwd=OUTDIR, capture_output=True, text=True)
    errors = [l for l in r.stdout.splitlines() if l.startswith("!")]
    if errors:
        print("LaTeX reported errors:")
        print("\n".join(errors[:10]))
        print(f"see {OUTDIR}/paper_preview.log")
    pdf = os.path.join(OUTDIR, "paper_preview.pdf")
    print("wrote " + pdf if os.path.exists(pdf) else "no PDF produced")


if __name__ == "__main__":
    main()
