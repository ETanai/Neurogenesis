# Replication Protocol Draft

This directory contains a LaTeX protocol draft for the Neurogenesis Deep Learning replication and the dynamic-capacity ANN literature review.

## Files

- `main.tex`: report entrypoint.
- `sections/*.tex`: bullet-oriented protocol sections.
- `references.bib`: temporary bibliography scaffold. Replace or refresh this from Zotero/Better BibTeX before final submission.

## Intended Zotero Workflow

1. Use the Zotero group or collection `Neurogenesis` for the selected literature set.
2. Export the group with Better BibTeX to `docs/protocol/references.bib`.
3. Keep the citekeys already used in the section files stable, or update the section citekeys after export.
4. Use the `zotero-bib-sync` workflow in dry-run mode before importing missing entries into Zotero.

See `zotero_sync_status.md` for the current bridge status and the resume command.

If Zotero is running but the bridge cannot reach `http://127.0.0.1:23119`, run the diagnostic command in `zotero_sync_status.md`. On Zotero 7, the connector HTTP server settings may need to be explicitly enabled in Zotero's Config Editor.

## Compile Command

From this directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -synctex=0 main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error -synctex=0 main.tex
pdflatex -interaction=nonstopmode -halt-on-error -synctex=0 main.tex
```

This build path avoids the local `biber` setup issue by using `natbib` and BibTeX.
