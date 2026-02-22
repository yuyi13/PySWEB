#!/bin/bash
# Script: sweb_mlcons_runner.sh
# Objective: Run MLConstraints SWEB preprocessing, domain calibration, and model simulation workflows for a selected run subdirectory.
# Author: Yi Yu
# Created: 2026-02-22
# Last updated: 2026-02-22
# Inputs: ET source NetCDFs, SILO rainfall inputs, MLConstraints soil rasters, SMAP-DS rasters, and CLI flags.
# Outputs: Preprocessed NetCDF inputs, calibration CSV, and SWEB RZSM NetCDF outputs.
# Usage: bash sweb_mlcons_runner.sh <run_subdir> [--burn-in-end YYYY-MM-DD] [--workers N] [--uncalibrated] [--mute-preprocess] [--mute-calib] [--mute-run]
# Requirements: bash, date, python, xarray-compatible Python environment, project scripts under /g/data/ym05/sweb_model/code/spec
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
|                 SWEB RUNNER                  |
|            Domain preprocess/calib/run       |
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

print_config() {
  local key="$1"
  shift
  local value="$*"
  local message=""
  printf -v message "%-17s: %s" "${key}" "${value}"
  print_status "CONFIG" "${message}"
}

# Project paths.
PROJECT_DIR="/g/data/ym05/sweb_model"
CODE_DIR="${PROJECT_DIR}/code/spec"
PREPROCESS_SCRIPT="${CODE_DIR}/3_sweb_preprocess_inputs.py"
CALIB_SCRIPT="${CODE_DIR}/4_sweb_calib_domain.py"
RUNNER_SCRIPT="${CODE_DIR}/5_sweb_run_model.py"

# Shared spatial/temporal setup.
MODEL_RUN_PERIOD="2024-01-01 to 2026-01-31"
BURN_IN_END="" # Optional CLI/default burn-in end override (YYYY-MM-DD).
CALIB_PERIOD="2021-01-01 to 2021-12-31"

#EXTENT="147.48, -34.79, 147.52, -34.75" # Summer Hill
EXTENT="148.62, -33.51, 148.66, -33.47" # The Pines
#EXTENT="147.20, -35.10, 147.30, -35.00" # North Wagga
#EXTENT="149.98, -28.80, 150.18, -28.60" # Warrakirri

SM_RES="0.00025"
DEFAULT_DIFF_FACTOR="1000.0"
DEFAULT_SM_MAX_FACTOR="1.0"
DEFAULT_SM_MIN_FACTOR="1.0"
DEFAULT_ROOT_BETA="0.96"
CALIB_DIFF_MIN="0.0"
CALIB_DIFF_MAX="10000.0"

# Forcing and reference inputs.
RAIN_ROOT="/g/data/yx97/EO_collections/SILO"
RAIN_PATTERN="{year}.daily_rain.nc"
RAIN_VAR="daily_rain"
ET_DIR="/g/data/ym05/sweb_model/2_ssebop_outputs"
ET_PATTERN="et_daily_ssebop_*.nc"
E_VAR="E"
ET_VAR="ET"
T_VAR="T"
SOIL_MLCONS_DIR="/g/data/yx97/EO_collections/USYD/MLConstraints"
SMAP_DIR="/g/data/yx97/EO_collections/NASA/SMAP/SMAP-DS"

# Output locations.
PREPROCESS_OUT_DIR="${PROJECT_DIR}/3_sweb+mlcons_inputs"
MODEL_OUT_DIR="${PROJECT_DIR}/4_sweb+mlcons_outputs"

# Step toggles (set via CLI flags).
RUN_PREPROCESS="true"
RUN_CALIB="true"
RUN_SWB="true"
UNCALIBRATED_MODE="false"
N_WORKERS="${N_WORKERS:-${PBS_NCPUS:-4}}"
RUN_SUBDIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mute-preprocess)
      RUN_PREPROCESS="false"
      shift
      ;;
    --mute-calib)
      RUN_CALIB="false"
      shift
      ;;
    --mute-run)
      RUN_SWB="false"
      shift
      ;;
    --uncalibrated)
      UNCALIBRATED_MODE="true"
      shift
      ;;
    --burn-in-end)
      if [[ $# -lt 2 ]]; then
        echo "--burn-in-end requires a date value (YYYY-MM-DD)." >&2
        exit 1
      fi
      BURN_IN_END="$2"
      shift 2
      ;;
    --workers)
      if [[ $# -lt 2 ]]; then
        echo "--workers requires an integer value." >&2
        exit 1
      fi
      N_WORKERS="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: sweb_domain_runner.sh <run_subdir> [--burn-in-end YYYY-MM-DD] [--workers N] [--uncalibrated] [--mute-preprocess] [--mute-calib] [--mute-run]

  run_subdir        Subdirectory under 2_ssebop_outputs/3_sweb+mlcons_inputs/4_sweb+mlcons_outputs.
  MODEL_RUN_PERIOD/CALIB_PERIOD are configured near the top of this script.
                    If CALIB_PERIOD is outside MODEL_RUN_PERIOD, Step 1 preprocesses both periods.
  --burn-in-end     Burn-in end date (YYYY-MM-DD). Burn-in start is fixed to MODEL_RUN_PERIOD start.
                    Default: 1 year from model start (clamped to model end - 1 day).
  --workers N       Number of parallel workers for preprocessing, calibration, and SWEB run steps.
  --uncalibrated    Run SWEB in uncalibrated mode (skip calibration and use default parameters).
  --mute-preprocess  Skip Step 1 (spatial preprocessing).
  --mute-calib       Skip Step 2 (domain-wide calibration).
  --mute-run         Skip Step 3 (SWEB model run).
USAGE
      exit 0
      ;;
    *)
      if [[ "$1" == --* ]]; then
        echo "Unknown option: $1" >&2
        exit 1
      fi
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

if [[ "${MODEL_RUN_PERIOD}" != *" to "* ]]; then
  echo "MODEL_RUN_PERIOD must use 'YYYY-MM-DD to YYYY-MM-DD': ${MODEL_RUN_PERIOD}" >&2
  exit 1
fi
MODEL_START_DATE="${MODEL_RUN_PERIOD%% to *}"
MODEL_END_DATE="${MODEL_RUN_PERIOD##* to }"

if ! MODEL_START_TS="$(date -d "${MODEL_START_DATE}" +%s 2>/dev/null)"; then
  echo "Invalid MODEL_RUN_PERIOD start date: ${MODEL_START_DATE}" >&2
  exit 1
fi
if ! MODEL_END_TS="$(date -d "${MODEL_END_DATE}" +%s 2>/dev/null)"; then
  echo "Invalid MODEL_RUN_PERIOD end date: ${MODEL_END_DATE}" >&2
  exit 1
fi
if (( MODEL_END_TS < MODEL_START_TS )); then
  echo "MODEL_RUN_PERIOD end date precedes start date: ${MODEL_RUN_PERIOD}" >&2
  exit 1
fi
if (( MODEL_END_TS == MODEL_START_TS )); then
  echo "MODEL_RUN_PERIOD must span at least two days to produce burn-in and post-burn outputs." >&2
  exit 1
fi

if [[ "${CALIB_PERIOD}" != *" to "* ]]; then
  echo "CALIB_PERIOD must use 'YYYY-MM-DD to YYYY-MM-DD': ${CALIB_PERIOD}" >&2
  exit 1
fi
CALIB_START_DATE="${CALIB_PERIOD%% to *}"
CALIB_END_DATE="${CALIB_PERIOD##* to }"

if ! CALIB_START_TS="$(date -d "${CALIB_START_DATE}" +%s 2>/dev/null)"; then
  echo "Invalid CALIB_PERIOD start date: ${CALIB_START_DATE}" >&2
  exit 1
fi
if ! CALIB_END_TS="$(date -d "${CALIB_END_DATE}" +%s 2>/dev/null)"; then
  echo "Invalid CALIB_PERIOD end date: ${CALIB_END_DATE}" >&2
  exit 1
fi
if (( CALIB_END_TS < CALIB_START_TS )); then
  echo "CALIB_PERIOD end date precedes start date: ${CALIB_PERIOD}" >&2
  exit 1
fi

CALIB_WITHIN_MODEL="false"
if (( CALIB_START_TS >= MODEL_START_TS )) && (( CALIB_END_TS <= MODEL_END_TS )); then
  CALIB_WITHIN_MODEL="true"
fi

IFS=',' read -r EXTENT_MIN_LON EXTENT_MIN_LAT EXTENT_MAX_LON EXTENT_MAX_LAT <<< "${EXTENT}"
EXTENT_MIN_LON="${EXTENT_MIN_LON//[[:space:]]/}"
EXTENT_MIN_LAT="${EXTENT_MIN_LAT//[[:space:]]/}"
EXTENT_MAX_LON="${EXTENT_MAX_LON//[[:space:]]/}"
EXTENT_MAX_LAT="${EXTENT_MAX_LAT//[[:space:]]/}"
if [[ -z "${EXTENT_MIN_LON}" || -z "${EXTENT_MIN_LAT}" || -z "${EXTENT_MAX_LON}" || -z "${EXTENT_MAX_LAT}" ]]; then
  echo "EXTENT must use 'MIN_LON, MIN_LAT, MAX_LON, MAX_LAT': ${EXTENT}" >&2
  exit 1
fi
EXTENT_ARGS=("${EXTENT_MIN_LON}" "${EXTENT_MIN_LAT}" "${EXTENT_MAX_LON}" "${EXTENT_MAX_LAT}")

if ! DEFAULT_BURN_IN_END="$(date -d "${MODEL_START_DATE} +1 year -1 day" +%Y-%m-%d 2>/dev/null)"; then
  echo "Failed to compute default burn-in end date from ${MODEL_START_DATE}" >&2
  exit 1
fi
if ! DEFAULT_BURN_IN_END_TS="$(date -d "${DEFAULT_BURN_IN_END}" +%s 2>/dev/null)"; then
  echo "Failed to parse computed burn-in end date: ${DEFAULT_BURN_IN_END}" >&2
  exit 1
fi
if ! MODEL_END_MINUS_ONE="$(date -d "${MODEL_END_DATE} -1 day" +%Y-%m-%d 2>/dev/null)"; then
  echo "Failed to compute model end minus one day from ${MODEL_END_DATE}" >&2
  exit 1
fi
if (( DEFAULT_BURN_IN_END_TS >= MODEL_END_TS )); then
  DEFAULT_BURN_IN_END="${MODEL_END_MINUS_ONE}"
fi
if [[ -z "${BURN_IN_END}" ]]; then
  BURN_IN_END="${DEFAULT_BURN_IN_END}"
fi
if ! BURN_IN_END_TS="$(date -d "${BURN_IN_END}" +%s 2>/dev/null)"; then
  echo "Invalid --burn-in-end date: ${BURN_IN_END}" >&2
  exit 1
fi
if (( BURN_IN_END_TS < MODEL_START_TS || BURN_IN_END_TS >= MODEL_END_TS )); then
  echo "Burn-in end must satisfy ${MODEL_START_DATE} <= burn-in-end < ${MODEL_END_DATE}." >&2
  echo "Provided burn-in end: ${BURN_IN_END} ; MODEL_RUN_PERIOD: ${MODEL_START_DATE} to ${MODEL_END_DATE}" >&2
  exit 1
fi
BURN_IN_START="${MODEL_START_DATE}"
POST_BURN_START=""
if ! POST_BURN_START="$(date -d "${BURN_IN_END} +1 day" +%Y-%m-%d 2>/dev/null)"; then
  echo "Failed to compute post burn-in start date from ${BURN_IN_END}" >&2
  exit 1
fi

ET_DIR="${ET_DIR}/${RUN_SUBDIR}"
PREPROCESS_OUT_DIR="${PREPROCESS_OUT_DIR}/${RUN_SUBDIR}"
MODEL_OUT_DIR="${MODEL_OUT_DIR}/${RUN_SUBDIR}"
CALIB_OUTPUT="${PREPROCESS_OUT_DIR}/domain_calibration.csv"
AUTO_UNCALIBRATED_REASON=""
if [[ "${RUN_CALIB}" == "false" && "${UNCALIBRATED_MODE}" == "false" && ! -f "${CALIB_OUTPUT}" ]]; then
  UNCALIBRATED_MODE="true"
  AUTO_UNCALIBRATED_REASON="--mute-calib was used and calibration CSV was not found at ${CALIB_OUTPUT}"
fi
mkdir -p "${PREPROCESS_OUT_DIR}" "${MODEL_OUT_DIR}"

print_badge
print_config "Model period" "${MODEL_RUN_PERIOD}"
print_config "Calib period" "${CALIB_PERIOD}"
print_config "Extent" "${EXTENT}"
print_config "Burn-in period" "${BURN_IN_START} to ${BURN_IN_END}"
print_config "Post-burn range" "${POST_BURN_START} to ${MODEL_END_DATE}"
if [[ "${CALIB_WITHIN_MODEL}" == "true" ]]; then
  print_config "Preprocess plan" "single run (CALIB_PERIOD within MODEL_RUN_PERIOD)"
else
  print_config "Preprocess plan" "dual runs (CALIB_PERIOD outside MODEL_RUN_PERIOD)"
fi
print_config "ET source dir" "${ET_DIR}"
print_config "Soil source" "${SOIL_MLCONS_DIR}"
print_config "Input dir" "${PREPROCESS_OUT_DIR}"
print_config "Output dir" "${MODEL_OUT_DIR}"
print_config "Calib CSV path" "${CALIB_OUTPUT}"
if [[ -f "${CALIB_OUTPUT}" ]]; then
  print_config "Calib CSV" "found"
else
  print_config "Calib CSV" "missing"
fi
print_config "Workers" "${N_WORKERS}"
print_config "Uncalibrated mode" "${UNCALIBRATED_MODE}"
if [[ -n "${AUTO_UNCALIBRATED_REASON}" ]]; then
  print_skip "CONFIG" "${AUTO_UNCALIBRATED_REASON}; forcing uncalibrated mode."
fi
echo

find_et_source_file() {
  local target_start="$1"
  local target_end="$2"
  local target_start_ts="$3"
  local target_end_ts="$4"
  local candidate=""
  local et_start=""
  local et_end=""
  local et_start_ts=""
  local et_end_ts=""

  for candidate in "${ET_DIR}"/${ET_PATTERN}; do
    [[ -f "${candidate}" ]] || continue
    if [[ "${candidate}" =~ et_daily_ssebop_([0-9]{4}-[0-9]{2}-[0-9]{2})_([0-9]{4}-[0-9]{2}-[0-9]{2})\.nc$ ]]; then
      et_start="${BASH_REMATCH[1]}"
      et_end="${BASH_REMATCH[2]}"
      if et_start_ts="$(date -d "${et_start}" +%s 2>/dev/null)" && \
         et_end_ts="$(date -d "${et_end}" +%s 2>/dev/null)"; then
        if (( target_start_ts >= et_start_ts )) && (( target_end_ts <= et_end_ts )); then
          echo "${candidate}"
          return 0
        fi
      fi
    fi
  done

  echo "No ET file in ${ET_DIR} matches ${ET_PATTERN} covering ${target_start} to ${target_end}." >&2
  return 1
}

run_preprocess_for_window() {
  local window_label="$1"
  local window_start="$2"
  local window_end="$3"
  local window_start_ts="$4"
  local window_end_ts="$5"
  local et_source_file=""

  if ! et_source_file="$(find_et_source_file "${window_start}" "${window_end}" "${window_start_ts}" "${window_end_ts}")"; then
    return 1
  fi

  print_status "STEP 1/3" "Preprocessing ${window_label}: ${window_start} to ${window_end}"
  print_status "STEP 1/3" "Using ET file: ${et_source_file}"
  python "${PREPROCESS_SCRIPT}" \
      --date-range "${window_start}" "${window_end}" \
      --extent "${EXTENT_ARGS[@]}" \
      --sm-res "${SM_RES}" \
      --workers "${N_WORKERS}" \
      --rain-root "${RAIN_ROOT}" \
      --rain-filename-pattern "${RAIN_PATTERN}" \
      --rain-var "${RAIN_VAR}" \
      --et-file "${et_source_file}" \
      --e-var "${E_VAR}" \
      --et-var "${ET_VAR}" \
      --t-var "${T_VAR}" \
      --soil-mlcons-dir "${SOIL_MLCONS_DIR}" \
      --output-dir "${PREPROCESS_OUT_DIR}"
}

# Step 1: preprocess forcing + soil + SMAP SSM to common grid.
if [[ "${RUN_PREPROCESS}" == "true" ]]; then
  if [[ "${CALIB_WITHIN_MODEL}" == "true" ]]; then
    print_status "STEP 1/3" "Running one preprocessing pass for MODEL_RUN_PERIOD."
    run_preprocess_for_window "MODEL_RUN_PERIOD" "${MODEL_START_DATE}" "${MODEL_END_DATE}" "${MODEL_START_TS}" "${MODEL_END_TS}"
    print_status "STEP 1/3" "CALIB_PERIOD is within MODEL_RUN_PERIOD; calibration will reuse model-period preprocessing outputs."
  else
    print_status "STEP 1/3" "Running separate preprocessing for MODEL_RUN_PERIOD and CALIB_PERIOD."
    run_preprocess_for_window "MODEL_RUN_PERIOD" "${MODEL_START_DATE}" "${MODEL_END_DATE}" "${MODEL_START_TS}" "${MODEL_END_TS}"
    run_preprocess_for_window "CALIB_PERIOD" "${CALIB_START_DATE}" "${CALIB_END_DATE}" "${CALIB_START_TS}" "${CALIB_END_TS}"
  fi
else
  if [[ "${CALIB_WITHIN_MODEL}" == "true" ]]; then
    print_skip "STEP 1/3" "Skipping preprocessing; expecting model-period preprocessed files to exist."
  else
    print_skip "STEP 1/3" "Skipping preprocessing; expecting model-period and calibration-period preprocessed files to exist."
  fi
fi

# Derived filenames for model and calibration windows.
MODEL_START_FMT=$(date -d "${MODEL_START_DATE}" +%Y%m%d)
MODEL_END_FMT=$(date -d "${MODEL_END_DATE}" +%Y%m%d)
MODEL_PRECIP_FILE="${PREPROCESS_OUT_DIR}/rain_daily_${MODEL_START_FMT}_${MODEL_END_FMT}.nc"
MODEL_EFFECTIVE_PRECIP_FILE="${PREPROCESS_OUT_DIR}/effective_precip_daily_${MODEL_START_FMT}_${MODEL_END_FMT}.nc"
MODEL_ET_FILE="${PREPROCESS_OUT_DIR}/et_daily_${MODEL_START_FMT}_${MODEL_END_FMT}.nc"
MODEL_T_FILE="${PREPROCESS_OUT_DIR}/t_daily_${MODEL_START_FMT}_${MODEL_END_FMT}.nc"
MODEL_SMAP_SSM_FILE="${PREPROCESS_OUT_DIR}/smap_ssm_daily_${MODEL_START_FMT}_${MODEL_END_FMT}.nc"

CALIB_START_FMT=$(date -d "${CALIB_START_DATE}" +%Y%m%d)
CALIB_END_FMT=$(date -d "${CALIB_END_DATE}" +%Y%m%d)
CALIB_PERIOD_PRECIP_FILE="${PREPROCESS_OUT_DIR}/rain_daily_${CALIB_START_FMT}_${CALIB_END_FMT}.nc"
CALIB_PERIOD_EFFECTIVE_PRECIP_FILE="${PREPROCESS_OUT_DIR}/effective_precip_daily_${CALIB_START_FMT}_${CALIB_END_FMT}.nc"
CALIB_PERIOD_ET_FILE="${PREPROCESS_OUT_DIR}/et_daily_${CALIB_START_FMT}_${CALIB_END_FMT}.nc"
CALIB_PERIOD_T_FILE="${PREPROCESS_OUT_DIR}/t_daily_${CALIB_START_FMT}_${CALIB_END_FMT}.nc"
CALIB_PERIOD_SMAP_SSM_FILE="${PREPROCESS_OUT_DIR}/smap_ssm_daily_${CALIB_START_FMT}_${CALIB_END_FMT}.nc"

if [[ "${CALIB_WITHIN_MODEL}" == "true" ]]; then
  CALIB_PRECIP_FILE="${MODEL_PRECIP_FILE}"
  CALIB_EFFECTIVE_PRECIP_FILE="${MODEL_EFFECTIVE_PRECIP_FILE}"
  CALIB_ET_FILE="${MODEL_ET_FILE}"
  CALIB_T_FILE="${MODEL_T_FILE}"
  CALIB_SMAP_SSM_FILE="${MODEL_SMAP_SSM_FILE}"
else
  CALIB_PRECIP_FILE="${CALIB_PERIOD_PRECIP_FILE}"
  CALIB_EFFECTIVE_PRECIP_FILE="${CALIB_PERIOD_EFFECTIVE_PRECIP_FILE}"
  CALIB_ET_FILE="${CALIB_PERIOD_ET_FILE}"
  CALIB_T_FILE="${CALIB_PERIOD_T_FILE}"
  CALIB_SMAP_SSM_FILE="${CALIB_PERIOD_SMAP_SSM_FILE}"
fi

# Step 2: domain-wide calibration (single parameter set).
if [[ "${RUN_CALIB}" == "true" ]]; then
  if [[ "${UNCALIBRATED_MODE}" == "true" ]]; then
    print_skip "STEP 2/3" "Uncalibrated mode enabled; skipping domain calibration and using default parameters."
  else
    print_status "STEP 2/3" "Running domain-wide calibration against SMAP-DS..."
    print_status "STEP 2/3" "Calibration range: ${CALIB_START_DATE} to ${CALIB_END_DATE}"
    print_status "STEP 2/3" "Calibration diff bounds (mm): ${CALIB_DIFF_MIN} to ${CALIB_DIFF_MAX}"
    print_status "STEP 2/3" "Workers: ${N_WORKERS}"
    if [[ "${CALIB_WITHIN_MODEL}" == "true" ]]; then
      print_status "STEP 2/3" "Calibration inputs are sourced from model-period preprocessing."
    else
      print_status "STEP 2/3" "Calibration inputs are sourced from calibration-period preprocessing."
    fi
    env \
        "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}" \
        "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}" \
        "MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}" \
        python "${CALIB_SCRIPT}" \
        --effective-precip "${CALIB_EFFECTIVE_PRECIP_FILE}" \
        --effective-precip-var "effective_precipitation" \
        --et "${CALIB_ET_FILE}" \
        --et-var "et" \
        --t "${CALIB_T_FILE}" \
        --t-var "t" \
        --soil-dir "${PREPROCESS_OUT_DIR}" \
        --smap-ssm "${CALIB_SMAP_SSM_FILE}" \
        --date-range "${CALIB_START_DATE}" "${CALIB_END_DATE}" \
        --layer-bottoms-mm 150 300 600 1000 \
        --diff-bounds "${CALIB_DIFF_MIN}" "${CALIB_DIFF_MAX}" \
        --workers "${N_WORKERS}" \
        --output "${CALIB_OUTPUT}"
  fi
else
  print_skip "STEP 2/3" "Skipping domain calibration (--mute-calib)."
fi

# Step 3: run SWEB using calibrated domain-wide parameters.
if [[ "${RUN_SWB}" == "true" ]]; then
  DIFF_FACTOR="${DEFAULT_DIFF_FACTOR}"
  SM_MAX_FACTOR="${DEFAULT_SM_MAX_FACTOR}"
  SM_MIN_FACTOR="${DEFAULT_SM_MIN_FACTOR}"
  ROOT_BETA="${DEFAULT_ROOT_BETA}"

  if [[ "${UNCALIBRATED_MODE}" == "false" && ! -f "${CALIB_OUTPUT}" ]]; then
    UNCALIBRATED_MODE="true"
    print_skip "STEP 3/3" "Calibration CSV not found at ${CALIB_OUTPUT}; running in uncalibrated mode with default parameters."
  fi

  if [[ "${UNCALIBRATED_MODE}" == "true" ]]; then
    print_status "STEP 3/3" "Uncalibrated mode active: forcing default parameters (no calibration CSV required)."
  else
    read -r DIFF_FACTOR SM_MAX_FACTOR SM_MIN_FACTOR ROOT_BETA < <(
      python - "${CALIB_OUTPUT}" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="") as handle:
    reader = csv.DictReader(handle)
    row = next(reader)
print(row["diff_factor"], row.get("sm_max_factor", ""), row.get("sm_min_factor", ""), row.get("root_beta", ""))
PY
    )
    if [[ -z "${DIFF_FACTOR}" ]]; then
      echo "Failed to read calibration parameters from ${CALIB_OUTPUT}" >&2
      exit 1
    fi
    if [[ -z "${ROOT_BETA}" ]]; then
      ROOT_BETA="${DEFAULT_ROOT_BETA}"
      print_skip "STEP 3/3" "No root_beta column in calibration CSV; using default beta=${ROOT_BETA}."
    fi
    print_status "STEP 3/3" "Using parameters from calibration CSV: ${CALIB_OUTPUT}"
  fi

  MODEL_OUTPUT_TMP="$(mktemp "${MODEL_OUT_DIR}/.SWEB_RZSM_FULL_${MODEL_START_DATE}_${MODEL_END_DATE}_XXXXXX.nc")"
  MODEL_OUTPUT_BURN="${MODEL_OUT_DIR}/SWEB_RZSM_Burn_In_${BURN_IN_START}_${BURN_IN_END}.nc"
  MODEL_OUTPUT_POST="${MODEL_OUT_DIR}/SWEB_RZSM_${POST_BURN_START}_${MODEL_END_DATE}.nc"

  print_status "STEP 3/3" "Running soil water balance model with domain parameters..."
  print_status "STEP 3/3" "Simulation range : ${MODEL_START_DATE} to ${MODEL_END_DATE}"
  print_status "STEP 3/3" "Burn-in period   : ${BURN_IN_START} to ${BURN_IN_END}"
  print_status "STEP 3/3" "Post-burn range  : ${POST_BURN_START} to ${MODEL_END_DATE}"
  print_status "STEP 3/3" "Workers          : ${N_WORKERS}"
  print_status "STEP 3/3" "Parameters       : diff=${DIFF_FACTOR}, sm_max=${SM_MAX_FACTOR:-${DEFAULT_SM_MAX_FACTOR}}, sm_min=${SM_MIN_FACTOR:-${DEFAULT_SM_MIN_FACTOR}}, beta=${ROOT_BETA}"
  env \
      "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}" \
      "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}" \
      "MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}" \
      python "${RUNNER_SCRIPT}" \
      --precip "${MODEL_PRECIP_FILE}" \
      --precip-var "precipitation" \
      --effective-precip "${MODEL_EFFECTIVE_PRECIP_FILE}" \
      --effective-precip-var "effective_precipitation" \
      --et "${MODEL_ET_FILE}" \
      --et-var "et" \
      --t "${MODEL_T_FILE}" \
      --t-var "t" \
      --date-range "${MODEL_START_DATE}" "${MODEL_END_DATE}" \
      --soil-dir "${PREPROCESS_OUT_DIR}" \
      --output-dir "${MODEL_OUT_DIR}" \
      --output-file "${MODEL_OUTPUT_TMP}" \
      --sm-res "${SM_RES}" \
      --layer-bottoms-mm 150 300 600 1000 \
      --diff-factor "${DIFF_FACTOR}" \
      --root-beta "${ROOT_BETA}" \
      --sm-max-factor "${SM_MAX_FACTOR:-1.0}" \
      --sm-min-factor "${SM_MIN_FACTOR:-1.0}" \
      --workers "${N_WORKERS}" \
      --nan-to-zero

  print_status "STEP 3/3" "Writing burn-in output  : ${MODEL_OUTPUT_BURN}"
  print_status "STEP 3/3" "Writing post-burn output: ${MODEL_OUTPUT_POST}"
  python - "${MODEL_OUTPUT_TMP}" "${MODEL_OUTPUT_BURN}" "${BURN_IN_START}" "${BURN_IN_END}" "${MODEL_OUTPUT_POST}" "${POST_BURN_START}" "${MODEL_END_DATE}" <<'PY'
import sys
import xarray as xr

src, burn_out, burn_start, burn_end, post_out, post_start, post_end = sys.argv[1:8]
with xr.open_dataset(src) as ds:
    burn_subset = ds.sel(time=slice(burn_start, burn_end))
    post_subset = ds.sel(time=slice(post_start, post_end))
    if burn_subset.sizes.get("time", 0) == 0:
        raise ValueError(f"No timesteps available for burn-in range: {burn_start} to {burn_end}")
    if post_subset.sizes.get("time", 0) == 0:
        raise ValueError(f"No timesteps available for post-burn range: {post_start} to {post_end}")
    burn_subset.to_netcdf(burn_out)
    post_subset.to_netcdf(post_out)
print(f"Wrote {burn_out}")
print(f"Wrote {post_out}")
PY
  rm -f "${MODEL_OUTPUT_TMP}"
else
  print_skip "STEP 3/3" "Skipping SWEB model run."
fi
