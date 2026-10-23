# Implementation design

This document specifies the data contracts, model, objective, numerical checks
and experiment interfaces implemented by DIAL-Det. Executable workflows and
JSON schemas appear in [the experiment guide](docs/EXPERIMENTS.md); automated
COCO cross-fitting is specified in [COCO_STUDY.md](docs/COCO_STUDY.md).

## 1. Scientific specification and implementation scope

The mathematical specification is *DIAL-Det: Learning Identity Mass for Budgeted
Detection Allocation* and its supplementary appendices. Main Sections 1–3,
Appendix B, Appendix C.7–C.9 and Appendix D define the core method. Appendix E
defines matching metrics, data roles and comparison protocols. The exact source
documents are identified in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json).

The implementation exposes its concrete numerical and architectural choices so
experiments can be reproduced with fixed settings. The supported integration
scope is listed in [IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md).
Local computational checks are recorded in [VALIDATION_REPORT.md](VALIDATION_REPORT.md).
Benchmark measurements require completed runs on the specified real data and
checkpoints; generated result records identify their actual execution inputs.

## 2. Frozen candidate contract

Each image supplies up to 500 candidates. `CandidateRecord` stores:

| Field | Shape/type | Meaning |
|---|---|---|
| `image_id` | string | Stable image identity |
| `ids` | int64[N] | Unique source candidate IDs |
| `boxes` | float64[N,4] | Original-image xyxy coordinates |
| `classes` | int64[N] | Native nonnegative class IDs |
| `scores` | float64[N] | Immutable detector probabilities |
| `features` | float32[N,D] | Frozen detector appearance/query features |
| `image_size` | (H,W) | Original image dimensions |
| `gt_boxes`, `gt_classes`, `gt_ids`, `gt_ignore` | M ground-truth entries | Training/audit targets only |
| `metadata` | JSON object | Detector, preprocessing, score floor, annotation and split provenance |

NPZ contains no pickled objects. A hash covers the complete numerical interface, including native scores. Evaluation verifies that selected IDs, boxes, classes and scores match the frozen export exactly. Model inference consumes only boxes, classes, scores, appearance and image size. Ground truth never constructs the inference support or supplies the identity probabilities.

### Candidate, policy and population identity

The numerical `record_hash` identifies the frozen arrays. A separate semantic hash binds that numerical record to dataset and detector/export metadata. Transient role and local path fields are excluded from semantic identity so a role reassignment or relocation does not change the underlying export identity. Role still appears in the population descriptor. Content-identified checkpoints are preferred; an unverified checkpoint reference is retained as such rather than treated as a known detector identity.

`population_descriptor` hashes a sorted list of image IDs, roles and numerical/semantic hashes. A `dialdet-complete-policy-v1` descriptor separately hashes the policy definition: implementation source, evidence checkpoint, model configuration, identity temperature, fitting IDs, quality mode, detector/export contract, support construction, solver and fallback settings, fixed K, native-score ranking, deterministic tie handling, and matching/risk definitions. `binding_sha256` joins the policy identity and the particular population identity. Changing a dataset path is therefore different from changing its detector contract or image population.

The detector export contract is population-independent; development exports from out-of-fold detectors can be recorded separately from calibration exports of the deployed detector. A custom exporter must faithfully populate recognized metadata or provide its full contract under `metadata["export_contract"]`. A digest of incomplete metadata cannot recover omitted preprocessing or training history.

`validate-manifest` checks every file in the manifest, even when `--role` restricts the reported population. It verifies declared file/numerical/semantic hashes, candidate constraints, image identities and feature-dimension consistency, while reporting optional hashes that were not declared. It reads candidate files one at a time. This preflight checks the supplied data contract; it cannot determine whether the dataset annotation or detector-history declarations are true.

The implemented membership track restores native detector scores after every selector. The paper's optional complete-method track, where DIAL-Det reports q, can use `coco_predictions(...,own_scores=q)` through the Python API but is not silently mixed into CLI membership results.

### Dataset adapters

All three native exporters use the frozen torchvision Faster R-CNN R50-FPN interface before final RoI detection NMS; RPN proposal generation remains intact. This is a concrete starting detector adapter, not a claim that arbitrary YOLO, DETR or MMDetection checkpoints load into torchvision. Additional detector backends must export the same frozen candidate contract.

COCO reads its instances JSON, retains native category IDs and records the detector label-table size. CrowdHuman reads full-body `fbox` from ODGT, retains `mask`/`extra.ignore` regions and keeps body targets independent of `head_attr.ignore`. Native person-class filtering occurs before the score-sorted candidate cap. A compatible local checkpoint is required for the pedestrian exporters; a two-class detector is the default. Explicit multiclass checkpoints preserve their original person class and source candidate IDs.

CityPersons reads the official MATLAB `anno_<split>[_aligned]` structure. Source label 1 is a target; other labels are ignored regions associated with the detector's person class. Full-body and visible-body metadata are retained. The default preserves annotation coordinate values. Optional `coordinate_origin=1` subtracts one from x/y for an explicitly declared alternate source. The optional reasonable target filter ignores height<50 or visible/full area<0.65, records that choice, and does not replace official evaluation.

Pedestrian records namespace their internal image IDs by dataset and source split. Original dataset IDs are preserved in metadata for benchmark output. Manifests specify `num_classes`, `quality_mode`, annotation hashes, detector provenance and role. The training embedding table follows the declared label space even when a small fitting split omits some classes.

`--quality-mode auto` maps COCO to the mean of ten quality probabilities and CrowdHuman/CityPersons to p_0.50 throughout training, inference, tuning and calibration-policy collection. The explicit override is retained for ablations and recorded in outputs. An explicit CLI dataset inconsistent with the manifest is rejected before running the method.

## 3. Evidence and training

Overlap support joins same-class boxes at IoU >=0.25. Rescue support adds four nearest appearance neighbors under cosine similarity inside center distance / mean diagonal <=2. Both supports are symmetrized, deduplicated and contain self-edges. The implementation also applies same-class compatibility to rescue edges.

Frozen appearance is projected to 128 dimensions and concatenated with four normalized box coordinates, native score and a 16-dimensional class embedding. A learned projection produces a 160-dimensional hidden state. Two residual graph layers use four attention heads in each branch. Their scalar gate depends on the previous node state, both branch messages and logarithms of their degrees. Sparse message operations process only supported edges; N-by-N storage is used for affinities, never a full N-by-N-by-N transitivity tensor.

The quality head produces ten monotone probabilities for IoU thresholds 0.50 to 0.95. Each later logit subtracts cumulative nonnegative softplus increments. COCO consumes the mean probability; `iou50` consumes the 0.50 probability. A symmetric pair head receives absolute embedding differences, embedding products and symmetric geometry, so r_ij=r_ji. Diagonal affinities are exactly one and unsupported affinities are zero.

Training uses:

```text
L = L_quality + L_pair + 0.1 L_mass + 0.05 L_transitivity
L_mass = mean_u (log(sum_j r_ju^p) - log(d*_u))^2
L_transitivity = mean_(i,j,k) relu(r_ij+r_jk-1-r_ik)^2
```

Transitivity includes open wedges with a missing i-k edge, where r_ik=0. Mass supervision sees all support edges. Pair BCE includes every eligible unordered edge; therefore P=Q and importance weights equal one. There is no undocumented pair-sampling or clipping scheme.

Assignment chooses the highest-IoU same-class nonignored annotation, with annotation-ID tie handling and default admissibility IoU 0.5. Background candidates receive private identities. Ignore-only candidates are masked from supervised targets. Mass-supervised nodes adjacent to unknown ignored identities are masked because their full neighborhood target is unknown; inference still includes the full support.

Default optimization is AdamW, learning rate 3e-4, weight decay 1e-4, 30 epochs. One image is processed per optimizer step, with gradient clipping at 5. A minimum-development-loss checkpoint is selected, and a positive identity temperature is fitted only on development edges. The mass is recomputed after temperature fitting. The separate risk-calibration population is never used for this fitting.

Explicit implementation choices: hidden width 160; LeakyReLU 0.2 attention; GELU/LayerNorm residual transformations; symmetric relative geometry defined in `pair_geometry`; one-image batches; no learning-rate schedule; temperature range [0.05,20]. These settings define the reference implementation alongside the paper's mathematical specification. Epoch resume restores model/optimizer/RNG state and verifies candidate hashes and configuration. The tested CPU resumed run is exactly equal to uninterrupted training; CUDA reduction order can differ by platform.

## 4. Objective

For calibrated quality q and identity affinity r:

```text
s_ij = r_ij^p
d_u = sum_j s_ju                 # all support edges, BEFORE thresholding
r_retained = s * (s >= theta)
mu_u = 1 / d_u

F(S) = sum_i_in_S (q_i-rho)
     + eta * sum_u mu_u * max_i_in_S(q_i*r_retained_iu)
     - lam * sum_i<j_in_S r_retained_ij * min(q_i,q_j)
subject to |S| <= K
```

Paper defaults: rho 0.08, eta 0.2, lambda 1.25, p 2, theta 0.05. Mass stays fixed after thresholding, decomposition, splitting, insertions and swaps. The objective is nonmonotone submodular: negative marginals and the empty report are valid. Exactly-K selection would change it.

For ideal identity blocks, d equals block size, normalized block demand equals one, and every optimum keeps at most one candidate per block. Its objective is `(1+eta)*sum(q)-rho*|S|`, independent of how many duplicate proposals each object has. The duplicate penalty is a soft cost; no hard conflict/set-packing constraint forbids two distinct overlapping objects.

The generic marginal and exchange functions are exact on the coefficient grid. Closed-form isolated-pair admission (all five branches) and budget-two exchange boundaries are separately available and tested against direct evaluation. Those closed forms are not substituted for the full general objective.

## 5. Search and certificates

Coefficients a_i, b_iu and c_ij are rounded to the grid delta=2^-30. Grid values are represented as integers. `GridObjective.value`, `marginal`, and `exchange` return integer units; `float_value` returns the unrounded objective.

1. Decompose the retained-affinity graph. Split components above 256 using deterministic strongest-edge-first size-limited Kruskal merging, a documented realization of the unspecified weak-edge split rule.
2. Enumerate every feasible subset of components of size<=12. Otherwise run width 64 insertion beams, retaining the best distinct sets by exact grid value, followed by deterministic first-improvement single swaps for each cardinality.
3. Keep lower curves for every exact cardinality. Negative feasible values remain negative; infeasible states are distinct from zero.
4. Construct per-cardinality LP upper bounds with coverage assignments, McCormick product rows and Boolean-quadric clique inequalities. Up to three separation rounds are used; a configurable maximum of 2048 enumerated maximal cliques limits reference-runtime cost. Omitted cuts only loosen the bound.
5. Treat HiGHS multipliers as candidate dual witnesses. Independently evaluate the residual dual bound with exact dyadic/rational arithmetic, rounded upward to grid units. This replaces the paper's floating directed-rounding implementation with conservative exact arithmetic.
6. Combine lower and upper curves with exact integer max-plus DP over all sizes<=K. Restore removed cross-component coverage by `A_s=sum_u max_cut_i b_iu`; dropping cross penalties is already an upper relaxation.
7. Reevaluate the returned incumbent on the original unsplit coefficients. Replay the certificate independently, reconstructing LP rows, dual residuals, enumeration curves, split correction and the dynamic programs.

The transfer envelope is:

```text
epsilon_g = delta/2 * (K + N + K*(K-1)/2)
g_F = B_delta - L_delta + 2*epsilon_g
```

The solver internally caps K at N, which is equivalent for feasible reports and can tighten the quantization envelope. Status is `rounded_optimum` if the integer bound equals the incumbent; `certified` if g_F<=0.015*max(B_delta+epsilon_g,1); `uncertified` if the checked gap is larger; and `fallback` after numerical-integrity failure. The gate uses exact integer comparison.

Missing multipliers, rejected certificates and nonfinite evidence trigger an explicit local-rule fallback. Fallback sanitation keeps finite candidates and conservatively handles nonfinite pair evidence; this declared branch must remain in calibration losses. The local threshold defaults to 0.5. Tune it on development data before a scientific comparison. Fallback returns no objective-gap certificate.

The certificate proves the recorded grid problem and the stated coefficient quantization envelope. It does not prove calibration accuracy, AP, dataset independence or detector-training history. It is also not a formal proof of every Python/compiler/hardware operation. This is a correctness-oriented Python/SciPy implementation, not a reproduction of the paper's millisecond latency.

## 6. Matching, miss bounds and AP

Maximum bipartite class-consistent matching computes H for the complete candidate pool and C for the selected report; Q=min(K,H). The evaluator-independent terms are candidate absence `G-H`, capacity loss `H-Q`, and allocation shortfall `Q-C`.

`C_score_greedy` is an explicitly labeled diagnostic assignment. It is not substituted for official COCOeval's Cev, which includes evaluator-specific ignores, crowd rules and area handling. Official COCOeval is used when an annotation path is supplied. Matching counts are never relabeled AP. Ignored annotations are excluded from H/C; selected candidates occupying capacity still count in |S| for the allocation-risk definition.

The matching audit reconstructs ideal reference blocks from a maximum matching, forms F0, bounds its discrepancy from F using the checked grid certificate, and inverts positive block-value tail sums using exact fractions. This is the label-dependent, retrospective bound in Appendix C.8. It never supplies inference evidence. Missing/mismatched certificates yield the vacuous bound Q.

For policy-bound prediction files, `evaluate` requires exactly one row per requested image for each method/budget, including empty reports. It validates semantic and numerical candidate hashes, the complete policy's method/budget/population/matching settings, stored evidence and certificate hashes, and checked DIAL-Det certificate replay. A declared fallback must match its recorded local-rule branch. An invalid bound output is rejected rather than silently evaluated as a different policy. Predictions without policy bindings support numerical membership audits with an explicit unverified identity; bound and unbound rows cannot be mixed into one evaluation.

Pooled shortfall is `sum(Q-C)/sum(Q)`. Risk shortfall is the image mean of `(Q-C)/Q`, defined as zero when Q=0. They have different weighting and are reported separately.

The high-pressure summary selects `ceil(0.2*N)` images with largest full-pool H, breaking ties by the string image ID. All methods use that same subset. This is a label-dependent audit condition, computed without inspecting method performance; it is not an input to allocation. Paired statistics fix this subset once before bootstrap resampling.

### Dataset-specific benchmark metrics

COCO uses pycocotools with the global K cap imposed before the standard per-category cap. CrowdHuman outputs one ODGT row per original image ID, including empty reports, with full-body xywh boxes and native scores. A declared nondefault native person class is mapped only to the evaluator's person tag; the frozen candidate record is unchanged.

The CrowdHuman adapter invokes the original CrowdDetection `APMRToolkits` in a separate process, with `Database(..., mode=0)`, IoU threshold 0.5, upstream AP integration and `eval_MR(ref="CALTECH_-2")`. Separate `--development-annotations` and audit annotation paths support the focal protocol, where development is from the training population and audit is from validation. It preserves source annotation order for stable cross-image score ties, filters annotations to the exact requested population, enforces K, and records evaluator source hashes. Metrics are fractions. The upstream strict IoU comparison, ignore-region IoA and trapezoidal AP differ from COCOeval and are not replaced by it. Original head annotations remain required because the upstream loader reads them even in full-body mode; full-body matching still uses `fbox`.

Two upstream undefined/buggy edge cases are handled explicitly in the result: empty effective reports receive AP50=0/MR=1/Recall=0; empty-GT images remain negative images with their detections counted as false positives. A population with no nonignored targets has undefined metrics (`null`). JI is absent. Without a local upstream evaluator checkout, native prediction files and matching diagnostics are available but no CrowdHuman benchmark values are claimed.

CityPersons native export, evidence training and allocation are implemented. Official CityPersons MR remains an external evaluation step with its exact subset settings. The generic matching diagnostic is not a CityPersons MR estimate. Public-checkpoint folds are merged before dataset evaluation, and a populated merge destination is rejected to prevent stale evidence from a previous run.

## 7. Risk and statistics

The finite family is rho in {.02,.04,.06,.08,.12,.16} times lambda in {1,1.25,1.5,2}, giving 24 policies and 48 risk hypotheses. Every configuration must use the same calibration images, fixed K and per-image Q, including numerical fallback outputs.

For losses in [0,1], sample variance with ddof1 is used:

```text
h = log(2*48/0.05)
U_raw = mean + sqrt(2*sample_variance*h/n) + 7*h/(3*(n-1))
U = min(1,U_raw)
```

The selector uses independently fixed development utility among policies satisfying waste<=.05 and shortfall<=.10. No feasible policy means no selected risk-qualified configuration. The numerical bound is valid only under the theorem's assumptions; the code does not establish those assumptions by assertion.

### Complete-policy collection and development scoring

`collect-risk` produces the full 24-policy family, complete policy descriptors and digests over its actual calibration rows. It checks disjointness from checkpoint training/development IDs and a mass exponent consistent with evidence training. `rounded_optimum`, `certified`, `uncertified` and `fallback` outputs all contribute their actual matching losses; unsuccessful certification is not a reason to remove an image.

`score-family` loads that registered family and applies its unchanged solver, fallback, quality, K, checkpoint, temperature and implementation source to the checkpoint's exact reserved development population. The family may differ only in the predefined rho/lambda grid. Calibration losses are ignored by this command. Official COCO AP or CrowdHuman AP50 is evaluated for each policy; each score is bound to its policy and the actual development population. The explicit matching-proxy option uses `sum(C)/sum(Q)`, with value one when the denominator is zero, and is marked diagnostic. It cannot enable risk-certificate eligibility.

This file chain is a consistency mechanism. A registry produced while collecting calibration records does not by itself establish preregistration. The policy family, complete fallback, budget, evidence/detector choices and scoring rule must already have been fixed without inspecting calibration outcomes. Independent detector training, IID deployment sampling and truthful population roles require external records.

For conditional eligibility, the provenance JSON requires `protocol="independent_focal_detector"`; true declarations for `complete_policy_including_fallback`, `iid_deployment_population`, `encoder_fixed_before_calibration`, `detector_unseen_calibration`, `policy_family_fixed_before_calibration`, and `budget_fixed_before_calibration`; arrays `train_ids`, `dev_ids`, `audit_ids`; and genuine 64-character `encoder_sha256`/`detector_sha256` values. The encoder/detector identities must match the collected policy and export contract. The optional `calibration_population_sha256` is checked if supplied. The implementation also requires verified calibration and development bindings, an official development metric, and nonsynthetic inputs. Public-checkpoint cross-fitting is ineligible for that focal-detector guarantee. Synthetic files are never evidence of population risk control.

Legacy policy rows `{rho, lam, records}` and development rows `{rho, lam, score}` remain accepted for numerical bounds. They have `legacy_unsealed_records` identity status and cannot enable risk-certificate eligibility. Bound and unbound records cannot be mixed. The report distinguishes `no_feasible_policy`, `bounds_only_unverified_policy_binding`, `bounds_only_unverified_population`, and `conditional_on_declared_provenance`. The last remains conditional on the external assumptions; it is not an independently authenticated certificate of training history.

### Executable paired statistics

The `paired-statistics` command checks complete held-out audit populations and matching definitions across two methods and encoder seeds. It independently resamples images and seeds, using the same indices for both methods. It supports image-mean waste, image-mean shortfall, pooled shortfall, and official COCO AP/AP50/AP75. COCO resamples clone repeated images with fresh IDs, recompute dataset-level AP separately for each sampled seed, then average those dataset metrics across sampled seeds. Per-image AP is never averaged, and detections from different encoder seeds are not pooled into one prediction set.

The primary contrast is method A minus method B. The reported primary 95% interval is the observed contrast plus/minus 1.96 times the bootstrap standard error; a percentile interval is included as a diagnostic. The default is 2000 replicates. At least two unique encoder seeds are required; the report explicitly records whether five were supplied. All quantities are fractions. The exact input schema and JSON output fields are in `docs/EXPERIMENTS.md`.

The statistics command supports both all images and the fixed high-pressure audit subset. Official CrowdHuman/CityPersons metric bootstrapping remains external. Holm adjustment is available through the Python helper. A full multiplicity-adjusted study requires its declared comparison family and measured statistics.

### Optional DPP control helper

`dialdet.baselines.dpp_kernel` symmetrizes a similarity matrix, projects it onto the positive-semidefinite cone, adds a positive ridge, normalizes its diagonal and applies quality scaling. `dpp_map` greedily maximizes the log determinant up to K, stopping when no positive gain remains; the empty set is allowed. `scale`, `ridge` and `epsilon` are exposed with defaults 1, 1e-4 and 1e-8. The helper is deliberately absent from the CLI's default control results until its settings are chosen on development data and the intended comparison protocol is implemented. Availability of this helper is not a reproduction of the manuscript's tuned DPP-MAP* study.

### Run evidence and release verification

Successful CLI stages write `RUN_MANIFEST.json` with implementation version/source fingerprint, runtime versions, explicit input-file hashes, output-file hashes, command arguments and synthetic classification. `RUN_STATUS.json` records completion; failed commands write `FAILURE.json` and a failed status. A pipeline has a manifest for the complete pipeline output. Cluster jobs additionally preserve their source snapshot and the final Python argument vector in `000_MANIFEST/effective_launch.json`.

`verify-run` checks recorded artifacts and manifest consistency, and can optionally recheck external input paths. Unlisted files are reported separately; the run manifest is not an external signature. `compare-results` requires an explicit manuscript ledger and generated entries that point to numeric values in verified, nonsynthetic run artifacts under equal declared experiment context. It checks the recorded value and file identity; it does not automatically verify manuscript transcription or historical truth.

`scripts/release_check.py` checks the package hash manifest, imports and CLI, shell syntax, the test suite and actual temporary-directory smoke execution. Neural checks are opt-in. It records commands and measured outcomes, installs nothing, and does not submit jobs. Local validation therefore establishes the stated execution checks, not dataset-scale benchmark results or Arrhenius runtime measurements.

## 8. Remaining empirical work

Real experiments require dataset files, detector weights, correct out-of-fold/public-crossfit exports, retraining of the evidence network, development tuning, untouched audit evaluation, and appropriate dataset-specific evaluators. Reproduction of all manuscript claims additionally needs all detector settings, eleven external baseline implementations, ablation retraining, five seeds, the original measurement protocol and runtime profiling on the target hardware.

The local test reports substantiate code execution, arithmetic bounds and data-interface behavior. They do not substantiate the paper's AP gains, risk rates, certification rates, or 3.65 ms mean latency.

## 9. Primary implementation references

- DIAL-Det paper and supplement; exact file hashes in `SOURCE_PROVENANCE.json`.
- CrowdHuman native annotations: https://www.crowdhuman.org/download.html
- CrowdHuman upstream AP/MR: https://github.com/megvii-model/CrowdDetection/tree/master/evaluate/APMRToolkits
- CityPersons native annotations: https://github.com/cvgroup-njust/CityPersons/blob/master/annotations/README.txt
- CityPersons published evaluation annotations: https://github.com/cvgroup-njust/CityPersons/blob/master/evaluation/val_gt.json
- SciPy HiGHS LP interface: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html
- Torchvision final RoI postprocessing: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py
- Arrhenius hardware: https://www.naiss.se/resources/arrhenius-technical-description/
- NAISS module documentation: https://hpc.pages.naiss.se/training/selecting-modules/intro/
