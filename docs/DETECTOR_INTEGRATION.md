# Detector integration

DIAL-Det does not vendor detector frameworks. An adapter performs five explicit
steps:

1. run a fixed detector checkpoint and assign candidate IDs before selection;
2. export boxes, detector scores, full class probabilities, and selected
   detector feature taps;
3. build one paired directed typed graph and recompute geometry in each
   direction;
4. freeze the active relation subset and apply it while running the supplied
   role-routed evidence encoder;
5. save the strict encoded-stream NPZ and its SHA-256.

Do not refine boxes, fuse boxes, run a second NMS, or alter detector features
inside a same-evidence attribution run. Any such operation creates a different
candidate boundary and must be named separately.

## Recent comparator policy

CP-Cluster, OTP-NMS, Seq2Seq-NMS, and IoU-aware calibration should be run from
their official implementations. Convert their final predictions to the common
evaluator format. The repository intentionally does not publish approximate
stand-ins under those method names.

## Graph construction

`dial_det.data.geometry` supplies directionally correct features. A project
adapter may construct geometry, spatial-neighbor, appearance, and supported
semantic relations, but the relation definitions, thresholds, ordering, and
active mask belong in the complete rule.

The allocation stage verifies that the active relation set in the complete rule
matches the set used to encode the frozen stream. Identity affinities below the
registered support threshold are set to zero before component construction, so
the optimization support and the exposure audit use the same frozen boundary.
