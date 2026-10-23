#!/bin/bash -l
# Run INSIDE an Arrhenius GPU allocation, whose CPU architecture is aarch64.
# Default is read-only validation; --install explicitly installs only the
# project's non-PyTorch packages into the selected existing venv.
set -Eeuo pipefail

case "${1:-check}" in
  help|-h|--help)
    cat <<'EOF'
Usage (inside an Arrhenius GPU allocation):
  bash setup_dialdet_arrhenius.sh              # validate only
  bash setup_dialdet_arrhenius.sh --install    # install NumPy/SciPy/Pillow/pycocotools

DIAL_ENV_FILE optionally selects a trusted shell configuration. DIAL_BASE,
DIAL_VENV_DIR and DIAL_PYTHON select the existing environment.
Default: /nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls/code/venv_arrhenius_gh200
PyTorch/torchvision are never installed or upgraded by this script: use the
site-supported compatible ARM64 CUDA builds. The detector exporter requires
torchvision; the allocator and GAT require no PyG or torch-scatter extension.
EOF
    exit 0 ;;
  check|--check) install=0 ;;
  --install) install=1 ;;
  *) printf '%s\n' 'Use check, --install or --help.' >&2; exit 64 ;;
esac
if [[ -n "${DIAL_ENV_FILE:-}" ]]; then
  [[ -r "${DIAL_ENV_FILE}" ]] || { printf '[ERROR] Cannot read DIAL_ENV_FILE: %s\n' "${DIAL_ENV_FILE}" >&2; exit 66; }
  DIAL_ENV_FILE="$(readlink -f -- "${DIAL_ENV_FILE}")"
  export DIAL_ENV_FILE
  set -a
  source "${DIAL_ENV_FILE}"
  set +a
fi
BASE="${DIAL_BASE:-${CM_BASE:-/nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls}}"
VENV_DIR="${DIAL_VENV_DIR:-${BASE}/code/venv_arrhenius_gh200}"
PYTHON="${DIAL_PYTHON:-${VENV_DIR}/bin/python}"
type module >/dev/null 2>&1 || { printf '%s\n' '[ERROR] Run inside an Arrhenius GPU allocation with modules available.' >&2; exit 69; }
module purge
module load "${DIAL_GPU_MODULE:-GPU/buildenv-nvhpc/25.9-cu13.0}"
module load "${DIAL_PYTHON_MODULE:-GPU/Python/3.13.5-bare-gcc-2025b-eb}"
[[ -x "${PYTHON}" ]] || { printf '[ERROR] Existing Python environment not found: %s\nCreate an ARM64 environment inside a GPU allocation and install the supported PyTorch stack first.\n' "${PYTHON}" >&2; exit 66; }
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
"${PYTHON}" - <<'PY'
import platform, sys
if sys.version_info < (3, 10):
    raise SystemExit('DIAL-Det requires Python 3.10 or later.')
if platform.machine() not in ('aarch64', 'arm64'):
    raise SystemExit('This is not an ARM64 GPU-node interpreter: ' + platform.machine())
print('[PYTHON]', sys.executable, platform.python_version(), platform.machine())
PY
if (( install )); then
  mkdir -p "${BASE}/cache/dialdet" "${BASE}/cache/pip"
  export PIP_CACHE_DIR="${BASE}/cache/pip"
  exec 9>"${BASE}/cache/dialdet/environment.lock"
  flock -x 9
  "${PYTHON}" -m pip install 'numpy>=1.26,<3' 'scipy>=1.11,<2' 'Pillow>=10' 'pycocotools>=2.0.7'
  flock -u 9
fi
"${PYTHON}" - <<'PY'
import importlib, json
missing, report = [], {}
for name in ('numpy', 'scipy', 'PIL', 'torch', 'torchvision', 'pycocotools'):
    try:
        m = importlib.import_module(name)
        report[name] = getattr(m, '__version__', 'installed')
    except Exception as exc:
        report[name] = f'UNAVAILABLE: {type(exc).__name__}: {exc}'
        if name in ('numpy', 'scipy', 'torch'):
            missing.append(name)
print(json.dumps(report, indent=2))
if missing:
    raise SystemExit('Required modules unavailable: ' + ', '.join(missing))
import torch
print('[CUDA]', torch.version.cuda, 'available=', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable; inspect the allocation and installed ARM64 PyTorch build.')
print('[GPU]', torch.cuda.get_device_name(0))
print('Core environment validated. torchvision and pycocotools are additional exporter/evaluator requirements.')
PY
