# Controlled ablations

These configurations make four component-removal experiments executable with the
same implementation. They define controlled comparisons to run on the chosen
candidate population and evaluation protocol. Their results come from the
completed training and evaluation runs.

| Configuration | Change from `full.json` | Question |
|---|---|---|
| `configs/ablations/full.json` | Explicit common model, loss and allocator settings | Reference for this controlled comparison |
| `configs/ablations/no_appearance.json` | `model.use_appearance=false` | How much does frozen appearance information contribute? |
| `configs/ablations/no_rescue.json` | `model.use_rescue=false` | How much does support beyond overlapping boxes contribute? |
| `configs/ablations/no_mass_loss.json` | `training.beta=0` | Does direct supervision of identity mass help? |
| `configs/ablations/no_transitivity.json` | `training.gamma=0` | Does identity consistency across wedges help? |

Each variant changes exactly the stated computational setting relative to the
explicit reference configuration; its descriptive `experiment` metadata also
changes. Feature dimension and class-table size still come from the frozen
candidate manifest. The reference includes documented implementation choices,
including hidden width 160 and gradient clipping at 5; see
[Implementation notes](../IMPLEMENTATION_NOTES.md).

## Run a controlled comparison

Start with the [COCO study specification](COCO_STUDY.md). Use the same frozen
candidate export, annotation file, split seed, development fraction, encoder
seeds, budgets and number of epochs for every variant. Change the specification's
`config` to the absolute path of the chosen configuration. A relative path is
resolved against the **specification file**, not the working directory. Declare
the comparison and its primary metric before viewing held-out results.

Create a separate specification and output directory for each variant. For
example, after saving the reference specification as `/data/full_study.json`:

```bash
python -m dialdet study --specification /data/full_study.json \
  --output /data/runs/full
python -m dialdet study --specification /data/full_study.json \
  --output /data/runs/full --execute
```

The first command validates inputs and writes a plan. The second executes that
plan. Repeat with a specification pointing to `no_mass_loss.json` and a different
output directory, then for each remaining variant. Every variant requires fresh
evidence training and development temperature fitting. Do not change a trained
checkpoint's metadata to simulate an ablation. The detector remains frozen.

The pipeline tunes the local-rule/fallback threshold on its own reserved
development images. Report that policy: the threshold is development-selected
under each variant, not selected on audit performance. Keep the declared grid,
selection metric and budget fixed across variants.

## What each switch actually does

**No appearance:** zeros frozen appearance before both the input projection and
rescue-neighbor ranking. Learned projection biases remain. Geometry-gated stable
ties still determine rescue neighbors, so this removes visual information and
also changes the support selected by the appearance ranking. It is not an
experiment that holds the full model's rescue graph fixed.

**No rescue:** disables the rescue branch and its message path. The model still
receives appearance in node features. Overlap support and self-edges remain.

**No mass loss:** retains the same identity-affinity computation and full,
unthresholded inference-mass definition; only the coefficient of direct log-mass
supervision becomes zero. Learned values can change after retraining. Coverage
is still normalized by identity mass.

**No transitivity:** sets the wedge consistency term's coefficient to zero. The
identity head remains symmetric; self-affinities, pair supervision and mass
supervision retain their reference behavior.

## Reporting and limits

Compare the completed held-out union for each encoder seed. Report mean AP and
matching diagnostics, the signed contrast, its interval, seed count and exact
protocol. Use `paired-statistics` with the reference run's and variant's *same*
images and seeds when constructing a cross-variant analysis. The bundled command
compares two method groups within each input run; it does not
automatically merge separate variant runs into that format. Preserve each
variant's complete run records when preparing such an analysis, and do not pool
predictions from different encoder seeds as one detector.

These four removals cover a useful subset of the appendix's experiments. They do
not implement the entire quadratic/full objective × greedy/structured search ×
mass-loss factorial, the unnormalized-coverage variant, every external decoder,
or the complete mechanism-diagnostic study. The within-run `greedy` and `local`
controls are emitted alongside DIAL-Det; they are distinct from retrained
encoder ablations. Scope is tabulated in
[Implementation status](IMPLEMENTATION_STATUS.md).
