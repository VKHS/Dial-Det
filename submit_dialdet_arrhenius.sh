#!/usr/bin/env bash
# Run this small submission helper on the Arrhenius login node.
# It creates log directories BEFORE Slurm opens the job's stdout/stderr.
set -Eeuo pipefail
umask 002

usage() {
  cat <<'EOF'
Usage: bash submit_dialdet_arrhenius.sh MODE [Python runner options]

Modes: smoke (default), train, infer, evaluate, export, export-coco,
       export-crowdhuman, export-citypersons, pipeline, collect-risk,
       score-family, tune-local, paired-statistics, study,
       full (= pipeline, 30 epochs), pilot (= pipeline, 2 epochs).

Examples:
  bash submit_dialdet_arrhenius.sh smoke
  DIAL_ENV_FILE=/absolute/coco.env bash submit_dialdet_arrhenius.sh export
  DIAL_ENV_FILE=/absolute/coco.env bash submit_dialdet_arrhenius.sh full
  DIAL_DATASET=crowdhuman bash submit_dialdet_arrhenius.sh pipeline --manifest /absolute/crowdhuman_manifest.json
  DIAL_STATISTICS_SPEC=/absolute/paired_runs.json bash submit_dialdet_arrhenius.sh paired-statistics
  DIAL_STUDY_SPEC=/absolute/coco_study.json bash submit_dialdet_arrhenius.sh study --execute

Overrides: DIAL_BASE, DIAL_CODE_DIR, DIAL_PYTHON, DIAL_VENV_DIR,
  DIAL_ACCOUNT, DIAL_PARTITION, DIAL_CPUS, DIAL_MEM, DIAL_TIME,
  DIAL_OUTPUT, DIAL_MANIFEST, DIAL_DEVICE, DIAL_DATASET,
  DIAL_IMAGES, DIAL_ANNOTATIONS, DIAL_DETECTOR_WEIGHTS, DIAL_CHECKPOINT,
  DIAL_BUDGET, DIAL_CONFIG, DIAL_POLICY_RECORDS, DIAL_STATISTICS_SPEC, DIAL_STUDY_SPEC,
  DIAL_ENV_FILE, DIAL_DRY_RUN=1.
Use DIAL_OUTPUT for the run directory, not the runner's --output option.
DIAL_DRY_RUN prints both the effective Python command and the sbatch command.
study defaults to planning; use --execute for computation and --resume to
reuse verified completed tasks. Set its scientific options in the JSON spec.
See configs/arrhenius.env.example for complete detection settings.
The supplied account and storage IDs are retained; their current allocation
validity is checked by Slurm when a job is submitted.
EOF
}

case "${1:-smoke}" in help|-h|--help) usage; exit 0 ;; esac
if [[ -n "${DIAL_ENV_FILE:-}" ]]; then
  [[ -r "${DIAL_ENV_FILE}" ]] || { printf '[ERROR] Cannot read DIAL_ENV_FILE: %s\n' "${DIAL_ENV_FILE}" >&2; exit 66; }
  DIAL_ENV_FILE="$(readlink -f -- "${DIAL_ENV_FILE}")"
  export DIAL_ENV_FILE
  set -a
  # The user explicitly selects this trusted Bash configuration.
  source "${DIAL_ENV_FILE}"
  set +a
fi
mode="${1:-smoke}"
if (( $# )); then shift; fi
case "${mode}" in
  smoke|train|infer|evaluate|export|export-coco|export-crowdhuman|export-citypersons|pipeline|full|pilot|collect-risk|score-family|tune-local|paired-statistics|study) ;;
  *) printf '[ERROR] Unsupported mode: %s\n' "${mode}" >&2; usage >&2; exit 64 ;;
esac
for arg in "$@"; do
  case "${arg}" in
    --output|--output=*)
      printf '%s\n' '[ERROR] Set DIAL_OUTPUT=/absolute/run/directory so logs, snapshots and Python outputs share one run directory.' >&2
      exit 64 ;;
  esac
done

DIAL_BASE="${DIAL_BASE:-${CM_BASE:-/nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls}}"
DIAL_CODE_DIR="${DIAL_CODE_DIR:-${DIAL_BASE}/code/dialdet_reconstruction}"
export DIAL_BASE DIAL_CODE_DIR
job_script="${DIAL_CODE_DIR}/run_dialdet_arrhenius.sh"
logs="${DIAL_BASE}/logs"
[[ -f "${job_script}" ]] || { printf '[ERROR] Missing job script: %s\n' "${job_script}" >&2; exit 66; }
[[ -f "${DIAL_CODE_DIR}/run_dialdet_arrhenius.py" ]] || { printf '[ERROR] Missing Python runner in %s\n' "${DIAL_CODE_DIR}" >&2; exit 66; }
# Validate routing and print the exact detection-stage options before submitting.
# This path requires neither modules nor a GPU and creates no run directory.
DIAL_PLAN=1 bash "${job_script}" "${mode}" "$@"
mkdir -p "${logs}" "${DIAL_BASE}/outputs" "${DIAL_BASE}/cache/dialdet"

default_time=04:00:00
[[ "${mode}" == smoke ]] && default_time=00:20:00
cmd=(sbatch --parsable
  --account="${DIAL_ACCOUNT:-naiss2025-5-643-gpu}"
  --partition="${DIAL_PARTITION:-gpu}" --nodes=1 --ntasks=1 --gpus=1
  --cpus-per-task="${DIAL_CPUS:-4}" --mem="${DIAL_MEM:-32G}"
  --time="${DIAL_TIME:-${default_time}}" --requeue
  --job-name="dialdet_${mode}" --chdir="${PWD}"
  --output="${logs}/DIALDET_%j.out" --error="${logs}/DIALDET_%j.err"
  --export=ALL "${job_script}" "${mode}" "$@")
printf '[SUBMIT]'; printf ' %q' "${cmd[@]}"; printf '\n'
if [[ "${DIAL_DRY_RUN:-0}" == 1 ]]; then
  printf '%s\n' '[DRY_RUN] No job submitted.'
  exit 0
fi
command -v sbatch >/dev/null 2>&1 || { printf '%s\n' '[ERROR] sbatch is unavailable. Submit from Arrhenius, or use DIAL_DRY_RUN=1 to inspect the command.' >&2; exit 69; }
job_reply="$("${cmd[@]}")"
job_id="${job_reply%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || { printf '[ERROR] Unexpected Slurm reply: %s\n' "${job_reply}" >&2; exit 70; }
printf '[JOB] %s\n[STDOUT] %s/DIALDET_%s.out\n[STDERR] %s/DIALDET_%s.err\n' "${job_id}" "${logs}" "${job_id}" "${logs}" "${job_id}"
