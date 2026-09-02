# Article-reported result overview

All values on this page are labeled **reported in the associated article**.
They are not outputs of the synthetic smoke tests and are not represented as
locally regenerated evidence.

## Headline findings

- DIAL-Out improves CrowdHuman AP from 87.35 to 88.23 under the matched
  same-evidence comparison: +0.88 AP with a reported 95% hierarchical interval
  of [0.39, 1.37].
- On the predeclared high-pressure COCO slice, DIAL-Out improves AP from 34.18
  to 35.58: +1.40 AP with a reported interval of [0.91, 1.89].
- The production allocation audit reports a 0.42% median lower--upper gap,
  1.28% p95, 2.61% p99, and a 3.8% fallback rate.
- For Faster R-CNN-R50 at proposal budget M=150, DIAL-Prop reports an AP effect
  of +0.01 with interval [-0.05, 0.07] and total latency of 16.4 ms versus
  24.1 ms for NMS-300, corresponding to the reported 32.0% saving.

## Complete numerical records

- `reported/paper_results.csv`: 318 values from result Tables V--XIII.
- `tables/`: one CSV extract for each of those tables.
- `figure_data/`: 113 numeric coordinates used in Figures 2, 3, 5, 6, and 7.
- `reported/SOURCE.json`: provenance statement and ledger checksum.

The article and supplementary PDFs remain the source for narrative context,
final typography, schematic Figure 1, and qualitative Figure 4.
