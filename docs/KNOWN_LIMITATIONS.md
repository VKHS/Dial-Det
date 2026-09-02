# Known limitations

- The repository provides DIAL-Det-specific source code and protocol, but it
  does not redistribute datasets, third-party detector source trees or weights,
  trained article checkpoints, candidate caches, or raw prediction archives.
- Exact article-level reproduction requires settings and asset identifiers that
  are not all enumerated in the article. The repository records the disclosed
  settings and does not invent the missing ones.
- Official external comparator implementations are not reimplemented under
  their names. Their outputs must be imported through the common schema.
- The large-component LP can dominate runtime and memory. A failed or loose
  bound triggers the registered fallback and removes the optimization-gap
  claim.
- Exposure validity depends on an independently constructed eligible-pair
  denominator. It cannot be inferred from the retained graph itself.
- Complete-rule calibration controls the stated expected bounded risks under
  its assumptions. It does not certify average precision, every individual
  image, an altered checkpoint, or an unregistered distribution shift.
- The 48-dimensional crop descriptor is a fully specified reference
  realization of the article's stated dimension partition. Exact replication
  must record the descriptor implementation used for the trained checkpoint.
