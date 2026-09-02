# Training

The trainable component is `RoleRoutedDIAL`. It receives node features, paired
directed typed edges, directional geometry, and image-level pressure features.
It produces:

- DIAL-Out raw benefit and uncertainty;
- DIAL-Prop object, localization, coverage, and ROI-success factors;
- directional identity logits;
- relation gates and embeddings.

## Prepared graph sample

Each training NPZ contains:

```text
node_features              [N,D]
edge_index                 [2,E]
edge_type                  [E]
edge_geometry              [E,G]
pressure_features          [P]
directional_identity       [E]
compatibility              [N,N]
graph_degree               [N]
ranking_target             [N]
value_out                  [N]
uncertainty_discount       scalar or broadcastable
prop_factors               [N,4]
beta_roi                   scalar or broadcastable
```

## Losses

The supplied loss composes unary supervision, edge identity, ranking,
uncertainty/evidence terms, and a soft-budget allocation objective. Identity
matrices are detached inside the allocation term so the allocation loss cannot
improve by erasing duplicate evidence. The temperature remains at 1.0 until the
final 40% of training and then decays to 0.05.

The article does not list every numerical loss coefficient. Therefore the
training runner requires explicit coefficients and stores them in the
checkpoint. `configs/smoke/training.json` is a runnable software example, not a
claim about the article's undisclosed weights.

## Command

```bash
python scripts/train_relation_encoder.py \
  --data-directory prepared/train \
  --configuration training.json \
  --output-directory checkpoints/run-001
```

The output contains the checkpoint, exact training configuration, source file
list, epoch history, and checkpoint SHA-256.
