# Troubleshooting

## Complete rule does not match stream

Compare detector and evidence checkpoint hashes, candidate interface, candidate
order, feature taps, and relation-type order. The mismatch is deliberate: a
rule must not be applied to a different frozen stream.

## Missing reverse edge

Every non-self directed edge must have exactly one reverse edge. Recompute
reverse geometry; never copy the forward feature vector.

## LP or MILP unavailable

Install SciPy. If an upper bound is unavailable, a finite-gap rule must use its
registered fallback rather than issue a near-optimality claim.

## Exposure gate fails

Provide an independent `eligible_identity_pairs` audit. The denominator cannot
be derived from the same retained graph being evaluated.

## Article value is not reproduced

Check dataset IDs, detector version/checkpoint, preprocessing, feature taps,
candidate cap, graph, evidence checkpoint, complete rule, seed, evaluator,
hardware, and statistical procedure. The reported ledger is a reference, not a
smoke-test fixture.
