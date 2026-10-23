# GitHub setup

## Repository layout

Place the contents of the `DIAL-Det` directory at the repository root. The root
contains `README.md`, `pyproject.toml`, `dialdet/`, `configs/`, `tests/`, `scripts/`,
`docs/`, `validation/` and the launch scripts. Include `.github/`, `.gitignore`
and `.gitattributes`; GitHub discovers the workflow under `.github/workflows/`.

Keep datasets, candidate exports, checkpoints, generated runs and local settings
outside the tracked source. The example configurations describe their expected
locations. For Arrhenius, install the repository contents at the code path in
[README_ARRHENIUS.txt](../README_ARRHENIUS.txt).

## Local validation

Run from the repository root in an environment with the required dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install 'setuptools>=68' wheel
python scripts/release_check.py --wheel --output /absolute/path/dialdet-check
```

For neural training and official COCO evaluation, install a compatible
PyTorch/torchvision build for the host, then run:

```bash
python -m pip install -r requirements_optional.txt
python scripts/release_check.py --neural --wheel --output /absolute/path/dialdet-neural-check
```

The gate checks package hashes and required files, imports, CLI availability,
shell syntax, the test suite, and a synthetic smoke pipeline. `--wheel` also
builds a source copy, installs it into a temporary target, verifies import
locations, and invokes the installed console entry point. Those wheel stages
use existing dependencies with package-index access and dependency resolution
disabled.

`RELEASE_CHECK.json` records commands, exit statuses and the overall outcome.
`TEST_REPORT.json`, `SMOKE_REPORT.json` and the complete logs record the checks
performed. Read skipped-test counts and reasons. The external CrowdHuman
integration tests require `DIAL_CROWDHUMAN_EVALUATOR` to point to an upstream
CrowdDetection checkout. Use a validation output directory outside the source.

## Create the GitHub repository

From inside the `DIAL-Det` directory:

```bash
git init -b main
git add .
git status --short
git diff --cached --stat
git commit -m "DIAL-Det implementation and experiment workflows"
```

Create an empty repository under the intended GitHub account, then use its
actual URL:

```bash
git remote add origin "REPLACE_WITH_YOUR_REPOSITORY_URL"
git push -u origin main
```

Review the staged source before committing. GitHub's
[local-source import guide](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
describes the import options.

## Continuous integration

[The workflow](../.github/workflows/ci.yml) runs on pushes, pull requests and manual
dispatch. It defines core jobs for Python 3.11 and 3.12, and a Python 3.12 CPU
neural job using PyTorch 2.5.1, torchvision 0.20.1 and the official COCO package.
Each job executes the release gate and retains its reports, logs, dependency
versions and built wheel. Its summary states test counts, failures, errors and
skips.

The workflow requests `contents: read` and disables persisted checkout
credentials. It uses [checkout](https://github.com/actions/checkout),
[setup-python](https://github.com/actions/setup-python), and
[upload-artifact](https://github.com/actions/upload-artifact). The CPU package
pairing follows the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v251).

The external CrowdHuman checkout is not provisioned in CI; dependent tests report
that limitation as a skip. The neural job exercises the study runner using a
small synthetic COCO-format fixture. Benchmark datasets and detector checkpoints
are external inputs to the scientific experiment workflow. Inspect the
repository's **Actions** tab for the result associated with each commit.

## Maintain package integrity

`PACKAGE_SHA256.json` identifies the tracked distribution. After reviewing and
staging an intentional source change, refresh its hashes and rerun the relevant
checks:

```bash
python - <<'PY'
import hashlib, json, subprocess
from pathlib import Path
from dialdet import __version__

root = Path.cwd().resolve()
tracked = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
hashes = {}
for name in sorted(set(tracked) - {"", "PACKAGE_SHA256.json"}):
    path = root / name
    if path.is_symlink() or not path.is_file() or root not in path.resolve().parents:
        raise SystemExit(f"Expected a tracked local regular file: {name}")
    hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
if not hashes:
    raise SystemExit("Stage the intended source files before generating hashes.")
(root / "PACKAGE_SHA256.json").write_text(
    json.dumps({"version": __version__, "sha256": hashes}, indent=2, sort_keys=True) + "\n"
)
PY
git add PACKAGE_SHA256.json
python scripts/release_check.py --neural --wheel --output /absolute/path/dialdet-check
```

Preserve the validation report with its source commit. Experimental outputs keep
their own input, source and artifact identities; updating package hashes does not
change the evidence associated with an experiment.

## Publish experimental evidence

Use the [implementation map](IMPLEMENTATION_STATUS.md),
[reviewer guide](REVIEWER_GUIDE.md) and [experiment guide](EXPERIMENTS.md) to connect
a result to its executed protocol. Include the study specification, detector and
candidate identities, data roles, completed task records, per-seed evaluation,
paired-statistics output and runtime environment. State the dataset, split,
budget and evaluator alongside each measurement.

The [validation report](../VALIDATION_REPORT.md) describes the executed software
checks. A benchmark results table should identify actual benchmark artifacts and
its statistical analysis. For a source download, retain the tests, configurations,
launch scripts and documentation alongside the Python package.
