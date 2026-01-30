# Deep Learning Cheatsheet (Modified)

This repository contains a Deep Learning Cheatsheet, serving as a comprehensive reference for the "Deep Learning" course.

## Attribution

This project is based on the [original cheatsheet by Andreas Bloch](https://github.com/andbloch/eth-dl-cheat-sheet).

## Modifications

This version includes several modifications to the original work:
- **Refactored Architecture**: The monolithic document has been split into modular chapters for better maintainability.
- **Improved Formatting**: Math environments have been standardized (e.g., using `\(` and `\)` for inline math) and spaced for better readability.
- **Enhanced Preamble**: The preamble has been refactored into a custom package file (`cheatsheet.sty`) to keep the main document clean.
- **Updated Content**: Updated content to match the current version of the lecture (HS25).

## Building the PDF

To compile the cheatsheet, ensure you have a standard LaTeX distribution installed (e.g., TeX Live, MiKTeX) and run:

```bash
pdflatex document.tex
```

You may need to run the command multiple times to ensure all cross-references are resolved correctly.
