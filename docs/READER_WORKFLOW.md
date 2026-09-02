# Reader implementation workflow

The repository is organized so that a reader can implement a complete local
DIAL-Det study without receiving the authors' article-production files or
private experiment cache.

## 1. Establish the environment

Create a clean Python environment and install the allocation, training,
evaluation, and plotting extras required by the intended study. Record the
Python, package, CUDA, driver, compiler, and hardware versions.

## 2. Obtain external assets

Download the selected detector framework, detector checkpoint, dataset, and
annotations from their original providers. Keep those assets outside this
repository and retain their licenses and checksums.

## 3. Freeze a detector candidate boundary

Assign candidate IDs before DIAL-Det processing. Export unchanged boxes,
detector scores, full class distributions, declared feature taps, and any
proposal costs. Do not silently refine, fuse, or suppress candidates in a
same-evidence comparison.

## 4. Construct the typed graph

Use `dial_det.adapters.graph_builder` or an equivalent registered adapter.
Geometry is recomputed independently in both directions. Freeze relation
definitions, neighbor counts, thresholds, priority, and the active relation
subset.

## 5. Prepare training graphs and train the evidence encoder

Follow `docs/TRAINING.md`. Every loss coefficient and training choice is
explicit configuration because values not enumerated in the article are not
invented in this repository. For a five-seed study, train and retain all five
predeclared seeds.

## 6. Encode and hash candidate streams

Apply the trained checkpoint with the active relation set, save the strict NPZ,
and record the stream SHA-256. The hash covers candidate data, features, graph,
evidence outputs, costs, checkpoints, preprocessing, and relation metadata.

## 7. Register complete rules

Create one rule JSON per detector/checkpoint/graph/value/objective/solver/budget/
ranking/fusion/fallback/matcher/risk combination. A calibration-visible change
creates a different rule and therefore a different multiplicity family.

## 8. Run allocation and baseline controls

Run DIAL-Out or DIAL-Prop through `dial-det run`. Use the same frozen stream for
same-evidence baselines. Run named external competitors from their official
implementations and convert only their outputs to the common evaluator format.

## 9. Calibrate and select

Supply actual sample IDs through the split manifest. Generate one bounded
image-loss vector for every fixed rule and risk, calibrate the whole family,
and choose utility only among jointly feasible rules on the disjoint utility
role.

## 10. Audit and report

Evaluate the selected rule once on the untouched audit role. Preserve run
manifests, stream/rule hashes, per-image loss vectors, solver certificates,
interval inputs, timing settings, and output checksums. Compare generated
results with `results/reported/` without overwriting those article-reference
files.
