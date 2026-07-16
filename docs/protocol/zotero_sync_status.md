# Zotero Sync Status

Target Zotero group: `Neurogenesis`

Target BibTeX export: `docs/protocol/references.bib`

Required citekeys are listed in `required_citekeys.txt`.

## Completed

- Added repo-local bridge script: `scripts/sync_zotero_bib_entries.py`.
- Verified that all LaTeX `\parencite{...}` keys are present in the current `references.bib`.
- Confirmed Zotero local API was initially reachable at `http://127.0.0.1:23119`.
- Dry-run against Zotero group `Neurogenesis` succeeded before import attempt.

## Last Successful Dry-Run Result

Existing in Zotero:

- `yoon2018den`

Missing from Zotero at that time:

- `chen2016net2net`
- `dai2018nest`
- `draelos2017neurogenesis`
- `evci2020rigging`
- `evci2022gradmax`
- `fahlman1990cascade`
- `fernando2017pathnet`
- `gordon2018morphnet`
- `guo2016dynamic`
- `han2015deep`
- `li2019learn`
- `liu2017network`
- `liu2019darts`
- `louizos2018learning`
- `maile2022north`
- `mocanu2018scalable`
- `rusu2016progressive`
- `wei2016network`
- `yan2021der`

## Citation Audit

The protocol citations were audited against primary paper pages where available. Two corrections were made:

- `draelos2017neurogenesis` now includes coauthor James B. Aimone.
- The orthogonality/neurogenesis claim now cites `maile2022north` instead of the original NDL paper.

The GradMax placeholder metadata was replaced with the full author list.

See `citation_audit.md` for the entry-level audit and Zotero coverage summary. Read-only Zotero database inspection found that most exact protocol citekeys already exist in the `Routing` group library; the `Neurogenesis` group is related but incomplete for this protocol bibliography.

## Current Status

Target Zotero group `Neurogenesis` is now complete for the protocol bibliography.

Final verified result:

- Zotero local API was restored on `http://127.0.0.1:23119`.
- The persistent Zotero profile override at `C:\Users\Admin\AppData\Roaming\Zotero\Zotero\Profiles\r1zhum3b.default\user.js` sets:
  - `extensions.zotero.httpServer.enabled = true`
  - `extensions.zotero.httpServer.port = 23119`
  - `extensions.zotero.httpServer.localAPI.enabled = true`
- Missing protocol entries were imported into Zotero group `Neurogenesis` (`group_id = 5485840`).
- A final dry-run found all 20 required protocol citekeys in `Neurogenesis` and no missing citekeys.
- Existing group `Routing` was also made complete for the same 20 citekeys.

The bridge script now has a diagnostic mode:

```powershell
python scripts\sync_zotero_bib_entries.py `
  --diagnose `
  --group-name Neurogenesis `
  --bib-file docs\protocol\references.bib `
  --import-bib docs\protocol\references.bib `
  --citekeys yoon2018den
```

Final diagnostic result:

- `extensions.zotero.httpServer.localAPI.enabled = true`
- `extensions.zotero.httpServer.enabled = true`
- `extensions.zotero.httpServer.port = 23119`

Recent Zotero 7 local-API guidance indicates that the connector HTTP server must also be enabled on port `23119`; the local API flag alone is not enough. The bridge diagnostic now reads both `prefs.js` and `user.js`, so it reports persistent overrides correctly.

`references.bib` is complete for the protocol citekeys and now matches a complete Zotero-backed source collection.

## Prepared Import Payload

The standalone NORTH* import payload is retained for traceability:

```text
docs/protocol/zotero_missing_import.bib
```

It has already been imported into both `Routing` and `Neurogenesis`.

## Verification Command

To re-check the complete `Neurogenesis` group:

```powershell
python scripts\sync_zotero_bib_entries.py `
  --dry-run `
  --group-name Neurogenesis `
  --bib-file docs\protocol\references.bib `
  --import-bib docs\protocol\references.bib `
  --citekeys (Get-Content docs\protocol\required_citekeys.txt)
```
