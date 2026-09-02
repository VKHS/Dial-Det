# Complete-rule configuration

A complete rule fixes every calibration-visible choice: detector and evidence
checkpoint hashes, preprocessing/candidate boundary, feature taps, relation
types and active mask, value construction, objective coefficients, budget mode,
solver tolerances, ranking, no-fusion policy, named fallback, matcher, risk
family, multiplicity, alpha, and split-manifest hash.

Runnable examples are in `configs/smoke/`. The strict loader rejects unknown
top-level fields, empty scientific sections, malformed hashes, placeholders,
incompatible contracts and budgets, missing fallbacks, box fusion, and invalid
statistics.

`configs/paper_protocol/reported_settings.json` is not a complete rule. It
records only values explicitly stated in the article and names the missing
fields required for exact replication.
