# Results and provenance

## Provenance labels

- `reported_in_article`: transcribed from the article-aligned numerical ledger;
- `generated_by_local_run`: produced by a reader's execution;
- `verified_local_artifact`: a local run whose declared files and hashes pass a
  local validation procedure.

A validator can check structure, hashes, arithmetic, and provenance labels. It
cannot turn a copied number into raw experimental evidence.

## Included reported values

The repository includes all 318 numeric entries in Tables V--XIII of the
article-aligned ledger and numeric coordinates explicitly used in Figures 2, 3,
5, 6, and 7. No projected or placeholder experiment appears.

## Excluded publication material

Article/supplement TeX and PDFs are absent. Figure 1's schematic and Figure 4's
qualitative montage are not duplicated. Readers use the journal PDFs for the
final visual presentation.

## New runs

A run writes to `runs/<id>/` and includes its rule, selection, certificate,
environment, and checksums. Experimental scripts must not edit
`results/reported/`.
