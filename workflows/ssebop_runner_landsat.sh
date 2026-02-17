#!/bin/bash

set -euo pipefail

# Terminal style helpers for an HPC-like startup badge and log lines.
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_CYAN=$'\033[36m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
else
  C_RESET=""
  C_BOLD=""
  C_CYAN=""
  C_GREEN=""
  C_YELLOW=""
fi

print_badge() {
  cat <<EOF
${C_CYAN}${C_BOLD}+----------------------------------------------+
|                 SSEBop RUNNER                |
|              Landsat + SILO workflow         |
+----------------------------------------------+${C_RESET}
EOF
}

print_status() {
  local label="$1"
  shift
  echo "${C_GREEN}${C_BOLD}[${label}]${C_RESET} $*"
}

print_skip() {
  local label="$1"
  shift
  echo "${C_YELLOW}${C_BOLD}[${label}]${C_RESET} $*"
}

format_seconds() {
  local total="$1"
  printf '%02dm%02ds' "$((total / 60))" "$((total % 60))"
}

run_with_progress() {
  local est_seconds="$1"
  shift

  if [[ ! -t 1 ]]; then
    "$@"
    return $?
  fi

  local width=30
  local elapsed=0
  local percent=0
  local filled=0
  local bar=""

  "$@" &
  local cmd_pid=$!

  while kill -0 "${cmd_pid}" 2>/dev/null; do
    if (( est_seconds > 0 )); then
      percent=$(( elapsed * 100 / est_seconds ))
      if (( percent > 99 )); then
        percent=99
      fi
    else
      percent=0
    fi

    filled=$(( percent * width / 100 ))
    bar="$(printf '%*s' "${filled}" '' | tr ' ' '#')$(printf '%*s' "$((width - filled))" '' | tr ' ' '-')"
    printf '\r%s%s[STEP 2/2]%s Progress [%s] %3d%% | elapsed %s | est %s' \
      "${C_CYAN}" "${C_BOLD}" "${C_RESET}" "${bar}" "${percent}" \
      "$(format_seconds "${elapsed}")" "$(format_seconds "${est_seconds}")"

    sleep 1
    elapsed=$((elapsed + 1))
  done

  if wait "${cmd_pid}"; then
    :
  else
    local status=$?
    printf '\n'
    return "${status}"
  fi

  printf '\r\033[K'
}

# Project paths.
PROJECT_DIR="/g/data/ym05/sweb_model"

# Shared spatial/temporal setup for both GEE download and SSEBop run.
DATE_RANGE="2020-11-01 to 2026-02-11"

#EXTENT="147.48, -34.79, 147.52, -34.75" # Summer Hill
#EXTENT="148.62, -33.51, 148.66, -33.47" # The Pines
EXTENT="147.20, -35.10, 147.30, -35.00" # North Wagga
#EXTENT="149.98, -28.80, 150.18, -28.60" # Warrakirri

# Input/output locations.
INPUT_DIR="${PROJECT_DIR}/1_ssebop_inputs"
SILO_DIR="/g/data/yx97/EO_collections/SILO"
OUTPUT_DIR="${PROJECT_DIR}/2_ssebop_outputs"

# Landsat/SSEBop settings for band mapping and ET interpolation.
LANDSAT_PATTERN="*.tif"
RED_BAND="SR_B4"
NIR_BAND="SR_B5"
LST_BAND="ST_B10"
MAX_GAP_DAYS="32"
APPLY_WATER_MASK="false"
SILO_TEMP_UNITS="celsius"
GAPFILL_ETF="true"
GAPFILL_WINDOW_DAYS="32"
GAPFILL_MIN_SAMPLES="5"
N_WORKERS="${N_WORKERS:-${PBS_NCPUS:-4}}"
DEM_PATH="/g/data/yx97/EO_collections/GA/DEM_30m_v01/dems1sv1_0/w001000.adf"
LANDCOVER_PATH="/g/data/yx97/EO_collections/ESA/WorldCover/ESA_WorldCover_100m_v200.tif"

# Step toggles (set via CLI flags).
RUN_DOWNLOAD="true"
RUN_SSEBOP="true"
RUN_SUBDIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mute-download)
      RUN_DOWNLOAD="false"
      shift
      ;;
    --mute-run)
      RUN_SSEBOP="false"
      shift
      ;;
    --workers)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --workers" >&2
        exit 1
      fi
      N_WORKERS="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: ssebop_runner_landsat.sh <run_subdir> [--mute-download] [--mute-run] [--workers N]

  run_subdir     Subdirectory under 1_input_data for downloaded data.
  --mute-download  Skip Step 1 (GEE download/config prep).
  --mute-run       Skip Step 2 (SSEBop model run).
  --workers N      Number of parallel scene workers for Step 2.
USAGE
      exit 0
      ;;
    *)
      if [[ -z "${RUN_SUBDIR}" ]]; then
        RUN_SUBDIR="$1"
        shift
      else
        echo "Unknown option: $1" >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "${RUN_SUBDIR}" ]]; then
  echo "Missing required run_subdir argument. See --help for usage." >&2
  exit 1
fi
if [[ ! "${N_WORKERS}" =~ ^[0-9]+$ ]] || (( N_WORKERS < 1 )); then
  echo "Invalid --workers value '${N_WORKERS}'. Must be an integer >= 1." >&2
  exit 1
fi

RUN_INPUT_DIR="${INPUT_DIR}/${RUN_SUBDIR}"
mkdir -p "${RUN_INPUT_DIR}"
RUN_OUTPUT_DIR="${OUTPUT_DIR}/${RUN_SUBDIR}"
mkdir -p "${RUN_OUTPUT_DIR}"

print_badge
print_status "CONFIG" "Date range : ${DATE_RANGE}"
print_status "CONFIG" "Extent     : ${EXTENT}"
print_status "CONFIG" "Input dir  : ${RUN_INPUT_DIR}"
print_status "CONFIG" "Output dir : ${RUN_OUTPUT_DIR}"
echo

# GEE download config is written to a temporary YAML file so the downloader
# can reuse its existing config loader.
GEE_CONFIG="$(mktemp)"
cleanup() {
  rm -f "${GEE_CONFIG}"
}
trap cleanup EXIT

# GEE parameters for Landsat C2 L2 downloads.
cat > "${GEE_CONFIG}" <<'YAML'
filename_prefix: "Landsat_30m"
# Preferred multi-collection setting (Landsat 8 + Landsat 9).
collections:
  - "LANDSAT/LC08/C02/T1_L2"
  - "LANDSAT/LC09/C02/T1_L2"
# Backward-compatible fallback for single-collection consumers.
collection: "LANDSAT/LC08/C02/T1_L2"
bands: ["SR_B4", "SR_B5", "ST_B10"]
scale: 30
out_format: "tif"
auth_mode: "browser"
max_images: null
cloud_mask:
  enabled: true
  band: "QA_PIXEL"
  type: "bits_any"
  bits: [0,1,2,3,4,5]
  keep: false
postprocess:
  maskval_to_na: true
  enforce_float32: false
YAML

# Step 1: download Landsat (and optionally write a per-run GEE config).
if [[ "${RUN_DOWNLOAD}" == "true" ]]; then
  print_status "STEP 1/2" "Running SSEBop input preparation (GEE download)..."
  python "1_ssebop_prepare_inputs.py" \
    --date-range "${DATE_RANGE}" \
    --extent "${EXTENT}" \
    --gee-config "${GEE_CONFIG}" \
    --out-dir "${RUN_INPUT_DIR}"
else
  print_skip "STEP 1/2" "Skipping SSEBop input preparation (GEE download)."
fi

# Step 2: run SSEBop using local Landsat + pre-downloaded SILO data.
if [[ "${RUN_SSEBOP}" == "true" ]]; then
  MODEL_START_TS="$(date +%s)"
  LANDSAT_COUNT="$(find "${RUN_INPUT_DIR}" -maxdepth 1 -type f -name "${LANDSAT_PATTERN}" | wc -l | tr -d ' ')"
  STEP2_EST_SECONDS="${STEP2_EST_SECONDS:-}"
  if [[ -z "${STEP2_EST_SECONDS}" ]]; then
    if (( LANDSAT_COUNT > 0 )); then
      STEP2_EST_SECONDS=$((LANDSAT_COUNT * 12))
    else
      START_DATE="${DATE_RANGE%% to *}"
      END_DATE="${DATE_RANGE##* to }"
      if START_TS="$(date -d "${START_DATE}" +%s 2>/dev/null)" && END_TS="$(date -d "${END_DATE}" +%s 2>/dev/null)"; then
        DATE_SPAN_DAYS=$(( (END_TS - START_TS) / 86400 + 1 ))
        if (( DATE_SPAN_DAYS > 0 )); then
          STEP2_EST_SECONDS=$((DATE_SPAN_DAYS * 6))
        fi
      fi
    fi
  fi
  if [[ ! "${STEP2_EST_SECONDS}" =~ ^[0-9]+$ ]]; then
    STEP2_EST_SECONDS="600"
  fi

  print_status "STEP 2/2" "Running SSEBop model..."
  print_status "STEP 2/2" "Date range        : ${DATE_RANGE}"
  print_status "STEP 2/2" "Landsat input dir : ${RUN_INPUT_DIR}"
  print_status "STEP 2/2" "Matched files     : ${LANDSAT_COUNT} (${LANDSAT_PATTERN})"
  print_status "STEP 2/2" "SILO source dir   : ${SILO_DIR}"
  print_status "STEP 2/2" "Band mapping      : red=${RED_BAND}, nir=${NIR_BAND}, lst=${LST_BAND}"
  print_status "STEP 2/2" "Gap settings      : max_gap_days=${MAX_GAP_DAYS}, temp_units=${SILO_TEMP_UNITS}"
  print_status "STEP 2/2" "Water mask        : ${APPLY_WATER_MASK}"
  print_status "STEP 2/2" "ETF gapfill       : ${GAPFILL_ETF} (window=${GAPFILL_WINDOW_DAYS}, min_samples=${GAPFILL_MIN_SAMPLES})"
  print_status "STEP 2/2" "Workers           : ${N_WORKERS}"
  print_status "STEP 2/2" "Progress estimate : $(format_seconds "${STEP2_EST_SECONDS}") (set STEP2_EST_SECONDS to override)"
  print_status "STEP 2/2" "Output dir        : ${RUN_OUTPUT_DIR}"

  RUN_ARGS=(
    --date-range "${DATE_RANGE}"
    --silo-dir "${SILO_DIR}"
    --landsat-dir "${RUN_INPUT_DIR}"
    --landsat-pattern "${LANDSAT_PATTERN}"
    --red-band "${RED_BAND}"
    --nir-band "${NIR_BAND}"
    --lst-band "${LST_BAND}"
    --dem "${DEM_PATH}"
    --landcover "${LANDCOVER_PATH}"
    --output-dir "${RUN_OUTPUT_DIR}"
    --max-gap-days "${MAX_GAP_DAYS}"
    --silo-temp-units "${SILO_TEMP_UNITS}"
    --workers "${N_WORKERS}"
  )
  if [[ "${APPLY_WATER_MASK}" == "true" ]]; then
    RUN_ARGS+=(--apply-water-mask)
  fi
  if [[ "${GAPFILL_ETF}" == "true" ]]; then
    RUN_ARGS+=(--gapfill-etf)
    RUN_ARGS+=(--gapfill-window-days "${GAPFILL_WINDOW_DAYS}")
    RUN_ARGS+=(--gapfill-min-samples "${GAPFILL_MIN_SAMPLES}")
  fi

  print_status "STEP 2/2" "Resolved command:"
  printf '  %q' env \
    "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}" \
    "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}" \
    "MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}" \
    python "2_ssebop_run_model.py" "${RUN_ARGS[@]}"
  printf '\n'

  run_with_progress "${STEP2_EST_SECONDS}" env \
    "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}" \
    "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}" \
    "MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}" \
    python "2_ssebop_run_model.py" "${RUN_ARGS[@]}"
  MODEL_END_TS="$(date +%s)"
  print_status "STEP 2/2" "Completed in $((MODEL_END_TS - MODEL_START_TS))s."
else
  print_skip "STEP 2/2" "Skipping SSEBop model run."
fi
