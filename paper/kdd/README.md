# KDD Format Paper (ACM Style)

This directory contains the paper formatted for KDD conference using ACM acmart style.

## Files
- `main.tex` - Main paper source
- `figures/` - All figures

## Required Files (obtain from TeX distribution or CTAN)
- `acmart.cls` - ACM article class (https://ctan.org/pkg/acmart)

## Installation of acmart

### Option 1: TeX Live / MiKTeX
The acmart package is included in most TeX distributions. If not installed:
```bash
# TeX Live
tlmgr install acmart

# MiKTeX
mpm --install=acmart
```

### Option 2: Manual download
Download from CTAN: https://ctan.org/pkg/acmart
Place `acmart.cls` in this directory.

## Compilation

```bash
pdflatex main.tex
pdflatex main.tex
```

Or with bibliography:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Notes
- Uses `sigconf` format with anonymous submission
- CCS concepts and keywords are included
- The paper uses inline `thebibliography` environment
- ACM format for KDD: typically 9 pages main text + unlimited appendix
