#!/bin/bash -l
#SBATCH -A naiss2025-5-643-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH -t 04:00:00
#SBATCH -J dialdet
#SBATCH -o /nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls/logs/DIALDET_%j.out
#SBATCH -e /nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls/logs/DIALDET_%j.err

# Use submit_dialdet_arrhenius.sh when changing DIAL_BASE: Slurm does not
# expand shell variables in #SBATCH directives and opens logs before execution.
set -Eeuo pipefail
umask 002

usage() {
  cat <<'EOF'
DIAL-Det Arrhenius job wrapper
  bash submit_dialdet_arrhenius.sh smoke
  DIAL_DATASET=coco bash submit_dialdet_arrhenius.sh export --images /data/val2017 --annotations /data/instances_val2017.json --weights /models/coco.pt
  DIAL_DATASET=crowdhuman bash submit_dialdet_arrhenius.sh export --images /data/Images --annotations /data/annotation_val.odgt --weights /models/crowdhuman.pt
  bash submit_dialdet_arrhenius.sh pipeline --manifest /absolute/manifest.json
  bash submit_dialdet_arrhenius.sh train --manifest /absolute/manifest.json --epochs 30
  bash submit_dialdet_arrhenius.sh infer --manifest /absolute/manifest.json --checkpoint /absolute/model.pt
  bash submit_dialdet_arrhenius.sh tune-local --manifest /absolute/manifest.json --checkpoint /absolute/model.pt --budget 100
  bash submit_dialdet_arrhenius.sh collect-risk --manifest /absolute/focal_manifest.json --checkpoint /absolute/model.pt --budget 100
  bash submit_dialdet_arrhenius.sh score-family --manifest /absolute/focal_manifest.json --checkpoint /absolute/model.pt --policy-records /absolute/risk_collection/policy_records.json
  bash submit_dialdet_arrhenius.sh paired-statistics --specification /absolute/paired_runs.json
  bash submit_dialdet_arrhenius.sh study --specification /absolute/coco_study.json --execute

The default smoke run performs a small real neural optimization on the GPU.
full maps to pipeline (30 epochs); pilot maps to pipeline (2 epochs).
study creates a COCO study plan by default; --execute explicitly runs it.
Its specification owns device, epochs, seeds and budgets. DIAL_STUDY_SPEC
sets the specification path; --resume explicitly resumes verified tasks.
Datasets: coco (default), crowdhuman, citypersons, generic.
export dispatches to a native exporter for the selected dataset.
paired-statistics runs on the CPU; collect-risk, score-family and tune-local run evidence
inference on DIAL_DEVICE. Single-budget stages use DIAL_BUDGET, not DIAL_BUDGETS.
All additional arguments are passed as separate arguments, never through eval.
Set DIAL_ENV_FILE to a trusted shell settings file, or set DIAL_* variables.
DIAL_PLAN=1 prints the effective Python command without accessing the GPU.
Use DIAL_OUTPUT=/absolute/run to set a run directory. Runtime validates the
existing environment; no packages are installed automatically. See
README_ARRHENIUS.txt and configs/arrhenius.env.example for the settings.
EOF
}
case "${1:-smoke}" in help|-h|--help) usage; exit 0 ;; esac
if [[ -n "${DIAL_ENV_FILE:-}" ]]; then
  [[ -r "${DIAL_ENV_FILE}" ]] || { printf '[ERROR] Cannot read DIAL_ENV_FILE: %s\n' "${DIAL_ENV_FILE}" >&2; exit 66; }
  DIAL_ENV_FILE="$(readlink -f -- "${DIAL_ENV_FILE}")"
  export DIAL_ENV_FILE
  set -a
  # Explicitly selected, trusted Bash configuration; paths remain quoted below.
  source "${DIAL_ENV_FILE}"
  set +a
fi
mode="${1:-smoke}"
if (( $# )); then shift; fi
case "${mode}" in smoke|train|infer|evaluate|export|export-coco|export-crowdhuman|export-citypersons|pipeline|full|pilot|collect-risk|score-family|tune-local|paired-statistics|study) ;; *) usage >&2; exit 64 ;; esac
for arg in "$@"; do
  case "${arg}" in
    --output|--output=*)
      printf '%s\n' '[ERROR] Use DIAL_OUTPUT to set the complete run directory.' >&2; exit 64 ;;
  esac
done
if [[ "${mode}" == study ]]; then
  # The specification is the single authority for scientific settings. Reject
  # unrelated stage flags even in command-printing mode, before any GPU work.
  study_needs_value=0
  for arg in "$@"; do
    if (( study_needs_value )); then
      [[ "${arg}" != --* && -n "${arg}" ]] || { printf '%s\n' '[ERROR] study --specification needs a path.' >&2; exit 64; }
      study_needs_value=0
      continue
    fi
    case "${arg}" in
      --specification) study_needs_value=1 ;;
      --specification=*|--execute|--resume) ;;
      *) printf '[ERROR] study accepts --specification, --execute and --resume; set device, epochs, seeds and budgets in its specification (received %s).\n' "${arg}" >&2; exit 64 ;;
    esac
  done
  (( ! study_needs_value )) || { printf '%s\n' '[ERROR] study --specification needs a path.' >&2; exit 64; }
fi

# Determine the dataset before constructing options. Explicit CLI choices take
# precedence over defaults, including the exporter and quality-target routing.
dataset="${DIAL_DATASET:-}"
for (( i=1; i<=$#; i++ )); do
  arg="${!i}"
  case "${arg}" in
    --dataset=*) dataset="${arg#--dataset=}" ;;
    --dataset)
      j=$((i+1))
      (( j <= $# )) || { printf '%s\n' '[ERROR] --dataset needs a value.' >&2; exit 64; }
      dataset="${!j}" ;;
  esac
done
case "${mode}" in
  study) dataset=coco ;;
  export-coco) dataset="${dataset:-coco}" ;;
  export-crowdhuman) dataset="${dataset:-crowdhuman}" ;;
  export-citypersons) dataset="${dataset:-citypersons}" ;;
  *) dataset="${dataset:-coco}" ;;
esac
case "${dataset}" in coco|crowdhuman|citypersons|generic) ;; *) printf '[ERROR] Unsupported detection dataset: %s\n' "${dataset}" >&2; exit 64 ;; esac
command_mode="${mode}"
case "${mode}" in
  full|pilot) command_mode=pipeline ;;
  export)
    case "${dataset}" in
      coco|crowdhuman|citypersons) command_mode="export-${dataset}" ;;
      *) printf '[ERROR] Dataset %s currently uses a prepared detection manifest; no native exporter is registered.\n' "${dataset}" >&2; exit 64 ;;
    esac ;;
esac
case "${command_mode}" in
  export-coco|export-crowdhuman|export-citypersons)
    [[ "${command_mode}" == "export-${dataset}" ]] || { printf '[ERROR] %s conflicts with DIAL_DATASET=%s.\n' "${command_mode}" "${dataset}" >&2; exit 64; } ;;
esac

BASE="${DIAL_BASE:-${CM_BASE:-/nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls}}"
CODE_DIR="${DIAL_CODE_DIR:-${BASE}/code/dialdet_reconstruction}"
VENV_DIR="${DIAL_VENV_DIR:-${BASE}/code/venv_arrhenius_gh200}"
PYTHON="${DIAL_PYTHON:-${VENV_DIR}/bin/python}"
LOGS_DIR="${BASE}/logs"
CACHE_DIR="${BASE}/cache"
OUTPUTS_DIR="${BASE}/outputs"
run_id="${SLURM_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
SUITE="${DIAL_OUTPUT:-${OUTPUTS_DIR}/DIALDET_${dataset}_${mode}_${run_id}}"
case "${SUITE}" in /*) ;; *) printf '%s\n' '[ERROR] DIAL_OUTPUT must be an absolute path.' >&2; exit 64 ;; esac
device="${DIAL_DEVICE:-cuda}"
# Statistics and evaluation do not run the evidence network. Keep their runtime
# preflight independent of a CUDA context; the GPU partition still allocates the
# requested node when these commands are submitted through this helper.
case "${command_mode}" in evaluate|paired-statistics|study) device=cpu ;; esac
export DIAL_DATASET="${dataset}" DIAL_DEVICE="${device}" DIAL_COMMAND_MODE="${command_mode}"
args=("${command_mode}" --output "${SUITE}")
add_env_option() {
  local option="$1" variable="$2"
  [[ -z "${!variable:-}" ]] || args+=("${option}" "${!variable}")
}
has_option() {
  local option="$1" value
  shift
  for value in "$@"; do [[ "${value}" == "${option}" || "${value}" == "${option}="* ]] && return 0; done
  return 1
}
case "${command_mode}" in
  train|infer|evaluate|pipeline|collect-risk|score-family|tune-local)
    args+=(--dataset "${dataset}")
    add_env_option --manifest DIAL_MANIFEST ;;
esac
case "${command_mode}" in
  smoke) args+=(--neural --device "${device}") ;;
  train|infer|pipeline|collect-risk|score-family|tune-local|export-coco|export-crowdhuman|export-citypersons) args+=(--device "${device}") ;;
esac
case "${command_mode}" in
  train|pipeline)
    quality_mode=auto
    case "${dataset}" in coco) quality_mode=coco ;; crowdhuman|citypersons) quality_mode=iou50 ;; esac
    default_epochs=30
    [[ "${mode}" != pilot ]] || default_epochs=2
    args+=(--quality-mode "${DIAL_QUALITY_MODE:-${quality_mode}}" --epochs "${DIAL_EPOCHS:-${default_epochs}}" --seed "${DIAL_SEED:-1}")
    add_env_option --resume DIAL_RESUME ;;
esac
case "${command_mode}" in
  infer|collect-risk|score-family|tune-local) add_env_option --checkpoint DIAL_CHECKPOINT ;;
  evaluate) add_env_option --predictions DIAL_PREDICTIONS ;;
  paired-statistics) add_env_option --specification DIAL_STATISTICS_SPEC ;;
  study) add_env_option --specification DIAL_STUDY_SPEC ;;
esac
if [[ "${command_mode}" == score-family ]]; then
  add_env_option --policy-records DIAL_POLICY_RECORDS
fi
case "${command_mode}" in
  train|infer|pipeline|collect-risk) add_env_option --config DIAL_CONFIG ;;
esac
case "${command_mode}" in
  collect-risk|tune-local)
    args+=(--budget "${DIAL_BUDGET:-100}")
    add_env_option --quality-mode DIAL_QUALITY_MODE ;;
esac
case "${command_mode}" in
  infer|pipeline)
    if [[ -n "${DIAL_BUDGETS:-}" ]]; then
      read -r -a budgets <<< "${DIAL_BUDGETS}"
      args+=(--budgets "${budgets[@]}")
    fi ;;
esac
case "${command_mode}" in
  pipeline|evaluate|tune-local|score-family)
    eval_annotations="${DIAL_EVAL_ANNOTATIONS:-${DIAL_ANNOTATIONS:-}}"
    case "${command_mode}" in tune-local|score-family) eval_annotations="${DIAL_DEV_ANNOTATIONS:-${eval_annotations}}" ;; esac
    if [[ -n "${eval_annotations}" ]]; then
      case "${dataset}" in
        coco) args+=(--coco-annotations "${eval_annotations}") ;;
        crowdhuman) args+=(--crowdhuman-annotations "${eval_annotations}") ;;
      esac
    fi
    if [[ "${dataset}" == crowdhuman ]]; then add_env_option --crowdhuman-evaluator DIAL_CROWDHUMAN_EVALUATOR; fi ;;
  export-coco|export-crowdhuman|export-citypersons)
    add_env_option --images DIAL_IMAGES
    add_env_option --annotations DIAL_ANNOTATIONS
    if ! has_option --download-weights "$@" && ! has_option --weights "$@"; then add_env_option --weights DIAL_DETECTOR_WEIGHTS; fi
    default_classes=91
    [[ "${dataset}" == coco ]] || default_classes=2
    args+=(--num-classes "${DIAL_NUM_CLASSES:-${default_classes}}")
    add_env_option --max-images DIAL_MAX_IMAGES
    add_env_option --score-floor DIAL_SCORE_FLOOR
    add_env_option --role DIAL_ROLE
    if [[ "${dataset}" == crowdhuman || "${dataset}" == citypersons ]]; then
      add_env_option --split DIAL_DATASET_SPLIT
      add_env_option --person-class-id DIAL_PERSON_CLASS_ID
    fi
    if [[ "${dataset}" == citypersons ]]; then
      args+=(--coordinate-origin "${DIAL_COORDINATE_ORIGIN:-0}" --citypersons-subset "${DIAL_CITYPERSONS_SUBSET:-all}")
    fi ;;
esac
if [[ "${command_mode}" == pipeline ]]; then
  add_env_option --development-annotations DIAL_DEV_ANNOTATIONS
fi
args+=("$@")
require_option() {
  local option="$1" setting="$2" found=0 value="" index token
  # Inspect the final occurrence, matching argparse's explicit-CLI precedence.
  # A present option followed by another flag is not a supplied path/value.
  for (( index=0; index<${#args[@]}; index++ )); do
    token="${args[index]}"
    if [[ "${token}" == "${option}="* ]]; then
      found=1; value="${token#*=}"
    elif [[ "${token}" == "${option}" ]]; then
      found=1; value="${args[index+1]:-}"
      [[ "${value}" != --* ]] || value=""
    fi
  done
  if (( ! found )) || [[ -z "${value}" ]]; then
    printf '[ERROR] %s requires %s (or %s in the settings file).\n' "${command_mode}" "${option}" "${setting}" >&2
    exit 64
  fi
}
case "${command_mode}" in
  train|infer|evaluate|pipeline|collect-risk|score-family|tune-local) require_option --manifest DIAL_MANIFEST ;;
esac
case "${command_mode}" in
  infer|collect-risk|score-family|tune-local) require_option --checkpoint DIAL_CHECKPOINT ;;
  evaluate) require_option --predictions DIAL_PREDICTIONS ;;
  paired-statistics) require_option --specification DIAL_STATISTICS_SPEC ;;
  study) require_option --specification DIAL_STUDY_SPEC ;;
  export-coco|export-crowdhuman|export-citypersons)
    require_option --images DIAL_IMAGES
    require_option --annotations DIAL_ANNOTATIONS
    if [[ "${command_mode}" != export-coco ]] || ! has_option --download-weights "${args[@]}"; then
      require_option --weights DIAL_DETECTOR_WEIGHTS
    fi ;;
esac
if [[ "${command_mode}" == score-family ]]; then
  require_option --policy-records DIAL_POLICY_RECORDS
fi
if [[ "${command_mode}" == tune-local || "${command_mode}" == score-family ]] && ! has_option --allow-matching-proxy "${args[@]}"; then
  case "${dataset}" in
    coco) require_option --coco-annotations DIAL_DEV_ANNOTATIONS ;;
    crowdhuman)
      require_option --crowdhuman-annotations DIAL_DEV_ANNOTATIONS
      require_option --crowdhuman-evaluator DIAL_CROWDHUMAN_EVALUATOR ;;
    *)
      printf '%s\n' '[ERROR] Development scoring for this dataset needs an implemented official evaluator, or the explicit diagnostic --allow-matching-proxy option.' >&2
      exit 64 ;;
  esac
fi
if [[ "${DIAL_PLAN:-0}" == 1 ]]; then
  printf '[PLAN] dataset=%s mode=%s\n' "${dataset}" "${command_mode}"
  printf '[COMMAND]'; printf ' %q' "${PYTHON}" "${CODE_DIR}/run_dialdet_arrhenius.py" "${args[@]}"; printf '\n'
  printf '%s\n' '[PLAN] Execution will use the recorded source snapshot. No job or GPU work was started.'
  exit 0
fi
required_modules="numpy scipy"
case "${command_mode}" in
  smoke|train|infer|pipeline|collect-risk|score-family|tune-local) required_modules+=" torch" ;;
  study)
    if has_option --execute "${args[@]}"; then required_modules+=" torch pycocotools"; fi ;;
  export-coco|export-crowdhuman|export-citypersons) required_modules+=" torch torchvision PIL" ;;
esac
if has_option --coco-annotations "${args[@]}"; then required_modules+=" pycocotools"; fi
export DIAL_REQUIRED_MODULES="${required_modules}"
mkdir -p "${LOGS_DIR}" "${CACHE_DIR}/dialdet" "${SUITE}/000_MANIFEST"
exact_out="${LOGS_DIR}/DIALDET_${run_id}_${mode}.out"
exact_err="${LOGS_DIR}/DIALDET_${run_id}_${mode}.err"
exec 7>&1 8>&2
exec > >(tee -a "${exact_out}" >&7) 2> >(tee -a "${exact_err}" >&8)
ln -sfn "${exact_out}" "${LOGS_DIR}/DIALDET_LIVE.out"
ln -sfn "${exact_err}" "${LOGS_DIR}/DIALDET_LIVE.err"
printf '[OUTPUT] %s\n[LOGS] %s %s\n' "${SUITE}" "${exact_out}" "${exact_err}"

export DIAL_RUN_DIR="${SUITE}" DIAL_RUN_MODE="${mode}" DIAL_CODE_DIR="${CODE_DIR}"
export DIAL_LOG_OUT="${exact_out}" DIAL_LOG_ERR="${exact_err}"
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export STARTED_UTC
write_status() {
  local rc="$1"
  export DIAL_EXIT_CODE="${rc}"
  if [[ -x "${PYTHON}" ]]; then
    "${PYTHON}" - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
p = Path(os.environ['DIAL_RUN_DIR']) / 'JOB_STATUS.json'
record = {
    'exit_code': int(os.environ['DIAL_EXIT_CODE']),
    'status': 'COMPLETED' if os.environ['DIAL_EXIT_CODE'] == '0' else 'FAILED',
    'mode': os.environ['DIAL_RUN_MODE'],
    'command_mode': os.environ['DIAL_COMMAND_MODE'],
    'dataset': os.environ['DIAL_DATASET'],
    'job_id': os.environ.get('SLURM_JOB_ID'),
    'restart_count': int(os.environ.get('SLURM_RESTART_COUNT', '0')),
    'started_utc': os.environ['STARTED_UTC'],
    'finished_utc': datetime.now(timezone.utc).isoformat(),
    'log_out': os.environ['DIAL_LOG_OUT'], 'log_err': os.environ['DIAL_LOG_ERR'],
    'note': 'Job completion is not evidence of manuscript accuracy or reproduced performance.'
}
tmp = p.with_suffix('.json.tmp')
tmp.write_text(json.dumps(record, indent=2) + '\n')
tmp.replace(p)
PY
  else
    printf 'exit_code=%s\nstatus=FAILED_RUNTIME\n' "${rc}" > "${SUITE}/JOB_STATUS.txt"
  fi
}
on_exit() {
  local rc=$?
  trap - EXIT
  write_status "${rc}" || true
  if (( rc != 0 )); then printf '[FAILED] exit=%s; inspect %s\n' "${rc}" "${exact_err}" >&2; fi
  exit "${rc}"
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

type module >/dev/null 2>&1 || { printf '%s\n' '[ERROR] Environment modules are unavailable. Use the Arrhenius login shell and GPU allocation.' >&2; exit 69; }
module purge
module load "${DIAL_GPU_MODULE:-GPU/buildenv-nvhpc/25.9-cu13.0}"
module load "${DIAL_PYTHON_MODULE:-GPU/Python/3.13.5-bare-gcc-2025b-eb}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${OMP_NUM_THREADS}" NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export TORCH_HOME="${CACHE_DIR}/torch" XDG_CACHE_HOME="${CACHE_DIR}/xdg" PIP_CACHE_DIR="${CACHE_DIR}/pip"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
mkdir -p "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}"
[[ -x "${PYTHON}" ]] || { printf '[ERROR] Python is missing: %s. See setup_dialdet_arrhenius.sh.\n' "${PYTHON}" >&2; exit 66; }
[[ -f "${CODE_DIR}/run_dialdet_arrhenius.py" ]] || { printf '[ERROR] Missing canonical runner in %s\n' "${CODE_DIR}" >&2; exit 66; }

if [[ "${command_mode}" == study ]] && has_option --execute "${args[@]}"; then
  study_specification=""
  for (( i=0; i<${#args[@]}; i++ )); do
    case "${args[i]}" in
      --specification) study_specification="${args[i+1]}" ;;
      --specification=*) study_specification="${args[i]#*=}" ;;
    esac
  done
  # Derive the preflight device from the final specification argument. Never
  # inject DIAL_DEVICE or another stage's training defaults into the study.
  device="$("${PYTHON}" - "${study_specification}" <<'PY'
import json, sys
from pathlib import Path
try:
    specification = json.loads(Path(sys.argv[1]).read_text())
    if not isinstance(specification, dict):
        raise ValueError('study specification must be a JSON object')
    device = specification.get('device', 'cpu')
    if device not in ('cpu', 'cuda'):
        raise ValueError('study device must be cpu or cuda')
except (OSError, ValueError) as exc:
    raise SystemExit(f'Cannot read the study runtime settings: {exc}') from exc
print(device)
PY
)"
  export DIAL_DEVICE="${device}"
fi

"${PYTHON}" - <<'PY'
import importlib, json, os, platform, sys
from pathlib import Path
if sys.version_info < (3, 10):
    raise SystemExit('DIAL-Det requires Python 3.10 or later.')
if platform.machine() not in ('aarch64', 'arm64'):
    raise SystemExit('Arrhenius GH200 jobs require the ARM64 Python environment; this interpreter is ' + platform.machine())
versions = {}
for name in os.environ['DIAL_REQUIRED_MODULES'].split():
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        raise SystemExit(f'Cannot load required {name}: {type(exc).__name__}: {exc}. Inspect the existing environment with setup_dialdet_arrhenius.sh. No automatic install was attempted.') from exc
    versions[name] = getattr(mod, '__version__', 'unknown')
record = {'machine': platform.machine(), 'python': platform.python_version(),
          'versions': versions, 'device': os.environ['DIAL_DEVICE'],
          'required_modules': os.environ['DIAL_REQUIRED_MODULES'].split(),
          'cuda_runtime': None, 'cuda_available': None}
if 'torch' in versions:
    import torch
    record.update(cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
if os.environ['DIAL_DEVICE'] == 'cuda':
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable in the allocated GPU environment.')
    # One finite correctness check; no background utilization/keepalive loop.
    x = torch.arange(16, dtype=torch.float32, device='cuda')
    if x.sum().item() != 120.0:
        raise SystemExit('CUDA tensor check failed.')
    record['gpu_name'] = torch.cuda.get_device_name(0)
    torch.cuda.synchronize()
Path(os.environ['DIAL_RUN_DIR'], '000_MANIFEST', 'runtime.json').write_text(json.dumps(record, indent=2) + '\n')
print('[RUNTIME]', json.dumps(record, sort_keys=True))
PY

# Snapshot the complete small source tree once. Requeues retain its identity.
"${PYTHON}" - <<'PY'
import hashlib, json, os, shutil
from pathlib import Path
source = Path(os.environ['DIAL_CODE_DIR']).resolve()
manifest = Path(os.environ['DIAL_RUN_DIR'], '000_MANIFEST')
dest = manifest / 'source'
hashes_path = manifest / 'source_sha256.json'
if dest.exists():
    if not hashes_path.is_file():
        raise SystemExit('Incomplete source snapshot; use a new DIAL_OUTPUT directory.')
    expected = json.loads(hashes_path.read_text())
    observed = {str(p.relative_to(dest)) for p in dest.rglob('*')
                if p.is_file() and '__pycache__' not in p.relative_to(dest).parts
                and p.suffix != '.pyc'}
    if observed != set(expected):
        raise SystemExit('Source snapshot contains missing or unrecorded files; use a new DIAL_OUTPUT directory.')
    for rel, digest in expected.items():
        p = dest / rel
        if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise SystemExit(f'Changed or missing snapshot file: {rel}')
else:
    temporary = manifest / ('source.building.' + str(os.getpid()))
    temporary.mkdir()
    ignore = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'output', 'outputs', 'validation', 'test_outputs'}
    suffixes = {'.py', '.sh', '.json', '.yaml', '.yml', '.toml', '.txt', '.md', '.ini', '.example'}
    hashes = {}
    for p in sorted(source.rglob('*')):
        rel = p.relative_to(source)
        if not p.is_file() or p.is_symlink() or any(part in ignore for part in rel.parts):
            continue
        if p.suffix not in suffixes or p.stat().st_size > 4 * 1024 * 1024:
            continue
        target = temporary / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        hashes[str(rel)] = hashlib.sha256(target.read_bytes()).hexdigest()
    if 'run_dialdet_arrhenius.py' not in hashes:
        raise SystemExit('Source snapshot is missing the runner.')
    temporary.rename(dest)
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + '\n')
print('[SOURCE]', str(dest))
PY
module list > "${SUITE}/000_MANIFEST/modules.txt" 2>&1
"${PYTHON}" -m pip freeze > "${SUITE}/000_MANIFEST/pip_freeze.txt"
nvidia-smi > "${SUITE}/000_MANIFEST/nvidia_smi.txt" 2>&1 || true
uname -a > "${SUITE}/000_MANIFEST/uname.txt"

runner="${SUITE}/000_MANIFEST/source/run_dialdet_arrhenius.py"
"${PYTHON}" - "${SUITE}/000_MANIFEST/command.json" "${PYTHON}" "${runner}" "${args[@]}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
command_path = Path(sys.argv[1])
command_path.write_text(json.dumps(sys.argv[2:], indent=2) + '\n')
# Explicit non-secret settings only. Never serialize the full environment or
# copy the contents of the user's trusted Bash settings file.
keys = ('DIAL_DATASET', 'DIAL_DEVICE', 'DIAL_COMMAND_MODE', 'DIAL_BASE',
        'DIAL_CODE_DIR', 'DIAL_VENV_DIR', 'DIAL_GPU_MODULE', 'DIAL_PYTHON_MODULE',
        'SLURM_JOB_ID', 'SLURM_JOB_ACCOUNT', 'SLURM_JOB_PARTITION',
        'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE', 'SLURM_RESTART_COUNT')
config = os.environ.get('DIAL_ENV_FILE')
record = {'argv': sys.argv[2:],
          'settings': {k: os.environ[k] for k in keys if k in os.environ},
          'settings_file': config,
          'settings_file_sha256': hashlib.sha256(Path(config).read_bytes()).hexdigest() if config else None,
          'device': os.environ['DIAL_DEVICE'],
          'modules': [os.environ.get('DIAL_GPU_MODULE', 'GPU/buildenv-nvhpc/25.9-cu13.0'),
                      os.environ.get('DIAL_PYTHON_MODULE', 'GPU/Python/3.13.5-bare-gcc-2025b-eb')],
          'note': 'argv is authoritative when explicit command-line options override environment defaults.'}
command_path.with_name('effective_launch.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
PY
printf '[COMMAND]'; printf ' %q' "${PYTHON}" "${runner}" "${args[@]}"; printf '\n'
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --ntasks=1 "${PYTHON}" -u "${runner}" "${args[@]}"
else
  "${PYTHON}" -u "${runner}" "${args[@]}"
fi
printf '[COMPLETE] %s\n' "${SUITE}"
