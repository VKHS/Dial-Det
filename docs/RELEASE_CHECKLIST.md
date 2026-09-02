# Public release checklist

Before tagging a release:

1. Run `pytest -q` and `python scripts/release_check.py`.
2. Run all four deterministic smoke contracts.
3. Validate every complete rule and split manifest used by a new experiment.
4. Confirm that `results/reported/` is unchanged and still labeled
   `reported_in_article`.
5. Remove local runs, datasets, weights, caches, build directories, secrets,
   machine paths, article source, and publication PDFs.
6. Generate `FILES.sha256.json`, build the source archive, and record the
   archive SHA-256.
7. Create an immutable Git tag matching the package version.
8. Make the code-availability wording in the submitted article agree exactly
   with `docs/CODE_AVAILABILITY_TEXT.md`.
