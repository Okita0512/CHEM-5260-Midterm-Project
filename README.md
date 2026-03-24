# CHEM 5260 Midterm Project

This repository contains a preliminary theoretical and numerical study of
quantum relaxation in two doorway-state models.

The main manuscript is [main.tex](./main.tex), which analyzes:
- Model A: a doorway state coupled to a finite bath manifold
- Model B: a two-step relaxation model with an upstream state and a doorway state
- wide-band approximation, Fermi's golden rule, damped two-level dynamics, and Poincar\'e recurrence

## Main files

- `main.tex`: project report
- `ref.bib`: bibliography database
- `criteria.tex`: criteria for a successful report
- `Model_A/`: numerical data, scripts, and figures for Model A
- `Model_B/`: numerical data, scripts, and figures for Model B

Generated simulation data under `**/data/` are intentionally excluded from
GitHub because some NumPy output files are too large for convenient upload.
These datasets can be regenerated locally by running `simulate.py` in the
corresponding model subdirectory.

## Build

Compile the manuscript with:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Compile the report criteria with:

```bash
pdflatex criteria.tex
```

## Data and code availability

The data and code supporting this study are available in this repository and at:

<https://github.com/Okita0512/CHEM-5260-Midterm-Project>
