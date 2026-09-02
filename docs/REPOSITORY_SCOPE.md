# Repository scope

## Purpose

This repository lets a technically capable reader implement and run DIAL-Det
without receiving a second copy of the journal article package. The source tree
contains implementation, schemas, protocols, examples, and reported numeric
references. The journal provides the article and supplementary PDFs.

## Meaning of “full code”

“Full code” means that the DIAL-Det-specific numerical and trainable components
are present: graph geometry, relation encoder, value heads, objective,
optimization, fallback gates, risk definitions, calibration, evaluation
helpers, manifests, and experiment utilities.

It does not mean that third-party detector frameworks, copyrighted datasets,
or trained study checkpoints are redistributed. A reader must obtain and
configure those assets and prepare the documented frozen candidate streams.

## Credibility boundary

The repository never treats a value copied from the article as a locally
recomputed result. It never calls a configuration “paper-exact” when the
article does not disclose every required field. It does not use a competitor's
name for a substitute implementation. It contains no artificial GPU workload
and no manuscript-production files.

## Publication-language consistency

A publication should describe this repository as a standalone implementation
and protocol release with article-reported reference values. It should not say
that all trained checkpoints, raw predictions, private timing traces, and every
cell-level source artifact are included unless those materials are separately
archived and linked.
