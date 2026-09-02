# Reported results and local runs

`reported/paper_results.csv` contains the numerical values stated in Tables
V--XIII of the associated article. Every row is labeled
`reported_in_article`. The values are not marked as locally regenerated or as
raw evidence.

`tables/` provides table-specific extracts of the same reported values.
`figure_data/` contains only numeric coordinates printed in article figures 2,
3, 5, 6, and 7. These files support reference-plot reconstruction; they do not
replace raw experiment records.

New experiments belong in `runs/` and must use a different provenance label.
The release checker rejects invalid provenance and projected, estimated, or placeholder result rows.
The article and supplementary source/PDF files are intentionally absent.
