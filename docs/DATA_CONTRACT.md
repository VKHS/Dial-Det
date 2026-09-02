# Encoded candidate-stream contract

The allocation boundary is a compressed NPZ loaded with `allow_pickle=False`.
The receiver/sender edge convention is `edge_index[0] <- edge_index[1]`.

## Required arrays

| Name | Shape | Meaning |
|---|---:|---|
| `candidate_ids` | `[N]` | stable identifiers assigned before allocation |
| `boxes_xyxy` | `[N,4]` | unchanged detector boxes |
| `detector_scores` | `[N]` | frozen reporting evidence |
| `class_probabilities` | `[N,C]` | full class distribution |
| `node_features` | `[N,D]` | frozen detector/crop evidence |
| `edge_index` | `[2,E]` | paired directed edges |
| `edge_type` | `[E]` | typed relation index |
| `edge_geometry` | `[E,G]` | recomputed directional geometry |
| `pressure_features` | `[P]` | one image-level routing vector |
| `raw_benefit_out` | `[N]` | DIAL-Out raw benefit head |
| `uncertainty_out` | `[N]` | nonnegative uncertainty head |
| `object_probability` | `[N]` | DIAL-Prop factor |
| `localization_quality` | `[N]` | DIAL-Prop factor |
| `coverage_probability` | `[N]` | DIAL-Prop factor |
| `roi_success_probability` | `[N]` | gated DIAL-Prop factor |
| `directional_identity_logits` | `[E]` | edge identity logits before symmetrization |
| `compatibility` | `[N,N]` | symmetric nonnegative compatibility gate |
| `graph_degree` | `[N]` | registered graph degree |
| `measured_cost` | `[N]` | measured or registered proposal cost |
| `eligible_identity_pairs` | `[N,N]` | independent exposure-audit denominator |
| `metadata_json` | scalar string | canonical metadata |

## Required metadata

- detector identifier;
- detector checkpoint SHA-256;
- evidence checkpoint SHA-256;
- candidate-interface identifier;
- candidate ordering;
- feature-tap list;
- preprocessing record;
- class mapping;
- ordered relation-type list;
- active relation subset used to encode the evidence;
- export-software version.

## Same-evidence hash

The SHA-256 includes every array above and canonical metadata. It therefore
changes when boxes, probabilities, features, graph structure, directional
geometry, values, uncertainty, identity evidence, compatibility, degree, cost,
checkpoint identity, or preprocessing changes.

## Exposure audit

`eligible_identity_pairs` is not inferred from the retained graph. It is an
independent denominator supplied by a labeled audit or a deliberately
conservative eligibility rule. This prevents the exposure check from becoming
tautological. A deployment stream without such an audit cannot issue an
exposure-dependent gap claim.
