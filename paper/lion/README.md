# LION Format Paper (LNCS Style)

This directory contains the paper formatted for LION conference using Springer LNCS style.

## Files
- `main.tex` - Main paper source
- `llncs.cls` - LNCS document class
- `splncs04.bst` - LNCS bibliography style
- `figures/` - All figures

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
- The paper uses inline `thebibliography` environment (no separate .bib file needed)
- All figures are referenced from the `figures/` directory
- LNCS format limits: typically 12-15 pages for full papers
