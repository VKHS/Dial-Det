# Experiment protocol

## Article study matrix

The reported study covers COCO, CrowdHuman, CityPersons, and COCO-C. Detector
streams include YOLOv8n, YOLOv12-L, D-FINE-L, DEIM--D-FINE-L, DINO-R50,
Faster R-CNN-R50/R101, and Mask R-CNN-R50. The machine-readable matrix is in
`configs/paper_protocol/experiment_matrix.json`.

## Required workflow

1. Freeze dataset roles before calibration.
2. Train detector/evidence models without calibration, utility, or audit data.
3. Export immutable candidate streams.
4. Freeze every complete-rule field.
5. Evaluate every fixed rule on the identical calibration IDs.
6. Admit rules with the joint risk bounds.
7. Select utility only on the disjoint utility IDs.
8. Evaluate the chosen rule once on the audit role.
9. Run same-evidence and exact-cardinality attribution comparisons.
10. Report stage-wise timing and memory under the clean protocol.

## Data roles

The article states 5,000 calibration images and 5,000 disjoint utility images.
Actual sample IDs, not merely different split names, must be passed to
`SplitManifest`. A repeated ID causes validation failure.

## Five-seed inference

Retain all five training seeds rather than selecting one by test performance.
For AP intervals, a bootstrap replicate must rerun the pooled evaluator. Do not
average per-image AP because AP is not image-additive.

## Same-evidence controls

Candidate IDs, caps, boxes, class distributions, features, graph, values,
affinities, ranking evidence, and image-wise cardinality remain fixed. Only the
selection rule changes.

## Proposal controls

First compare selectors at equal `M`; then compare the chosen DIAL-Prop budget
against the operational NMS-300 system. Selector time, ROI/head time, total
time, proposal recall, miss UCB, and memory are separate measurements.
