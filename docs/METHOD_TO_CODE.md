# Method-to-code map

| Method element | Source |
|---|---|
| DIAL-Out value | `dial_det.method.values.dial_out_value` |
| DIAL-Prop value | `dial_det.method.values.dial_prop_value` |
| Directional-logit symmetrization | `dial_det.method.adapters.identity_matrix_from_directional_edges` |
| Frozen set evidence | `dial_det.method.objective.DIALObjectiveEvidence` |
| Quality--coverage--redundancy objective | `dial_det.method.objective` |
| Interaction components and additivity audit | `dial_det.method.objective` |
| Exact/MILP component curves | `dial_det.method.optimization.solve_component_curve` |
| Beam/two-swap feasible lower curve | `dial_det.method.optimization` |
| McCormick LP upper curve | `dial_det.method.optimization` |
| Global lower/upper DP | `dial_det.method.optimization` |
| DIAL-Prop measured-cost optimization | `dial_det.method.optimization.optimize_measured_cost_with_certificate` |
| Many-to-one output risks | `dial_det.method.risks` |
| Proposal-miss risk | `dial_det.method.risks.proposal_miss_loss` |
| Complete-rule empirical-Bernstein control | `dial_det.method.calibration` |
| Typed role-routed encoder | `dial_det.models.role_routed` |
| Unary, edge, rank, allocation, evidence losses | `dial_det.models.losses` |
| Directionally correct geometry | `dial_det.data.geometry` |
| Complete same-evidence hash | `dial_det.data.stream.EncodedCandidateStream.sha256` |
| Complete-rule validation | `dial_det.config.rule` |
| Actual split-ID disjointness | `dial_det.config.splits` |
| End-to-end contracts and fallback | `dial_det.pipeline` |
| Same-stream baselines | `dial_det.experiment.baselines` |
| Seed-aware and multiplicity statistics | `dial_det.experiment.statistics` |
| Timing protocol | `dial_det.experiment.timing` |
| Official COCO evaluation wrapper | `dial_det.experiment.coco` |
