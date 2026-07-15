# Compilation

Compile from the directory containing `main.tex`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The document uses `biblatex` with the Biber backend. `latexmk` runs the required Biber and PDFLaTeX passes automatically.

Required source files:

- `main.tex`
- `references.bib`
- `figures/boxplot_visual_change_by_prompt.png`
- `figures/prompt_classifier_confusion_matrix_nonbaseline.png`
- `figures/bbox_style_vs_parseable.png`
