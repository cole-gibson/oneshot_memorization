#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/submit_slurm_array.sh CONFIG_DIR [-- extra sbatch args...]

Submits a Slurm job array with one task per config.yaml under CONFIG_DIR.

Environment:
  TRAIN_SEEDS        Optional seed list/ranges, e.g. 0-1 or 0-3,7.
  TRAIN_COMMAND      Command used to launch training. Default: uv run python -m base.train
  SLURM_ARRAY_LIMIT  Optional max concurrently running array tasks.
  SLURM_JOB_NAME     Optional job name. Default: train-configs
  SLURM_PENDING_DIR  Optional temporary log directory. Default: logs/slurm/pending

Examples:
  scripts/submit_slurm_array.sh configs
  TRAIN_SEEDS=0-1 scripts/submit_slurm_array.sh configs
  TRAIN_SEEDS=0-3 SLURM_ARRAY_LIMIT=2 scripts/submit_slurm_array.sh configs -- --partition=gpu
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

expand_train_seeds() {
  local spec="${TRAIN_SEEDS:-}"
  if [[ -z "$spec" ]]; then
    printf '\n'
    return
  fi

  local chunk start end seed
  local -a chunks
  IFS=',' read -r -a chunks <<< "$spec"
  for chunk in "${chunks[@]}"; do
    chunk="${chunk//[[:space:]]/}"
    if [[ "$chunk" =~ ^[0-9]+-[0-9]+$ ]]; then
      start="${chunk%-*}"
      end="${chunk#*-}"
      (( start <= end )) || die "invalid TRAIN_SEEDS range '$chunk'"
      for ((seed = start; seed <= end; seed++)); do
        printf '%s\n' "$seed"
      done
    elif [[ "$chunk" =~ ^[0-9]+$ ]]; then
      printf '%s\n' "$chunk"
    else
      die "invalid TRAIN_SEEDS entry '$chunk'"
    fi
  done
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ $# -ge 1 ]] || {
  usage
  exit 1
}

config_root="$1"
shift

if [[ "${1:-}" == "--" ]]; then
  shift
fi

[[ -d "$config_root" ]] || die "CONFIG_DIR does not exist: $config_root"
command -v sbatch >/dev/null 2>&1 || die "sbatch was not found on PATH"

project_dir="$(pwd)"
config_root="$(cd "$config_root" && pwd)"
manifest_dir="$project_dir/.slurm_manifests"
pending_log_dir="${SLURM_PENDING_DIR:-$project_dir/logs/slurm/pending}"
run_dir_marker_dir="$manifest_dir/run-dirs"
timestamp="$(date +%Y%m%d-%H%M%S)"
manifest="$manifest_dir/config-array-$timestamp.tsv"

mkdir -p "$manifest_dir" "$pending_log_dir" "$run_dir_marker_dir"

configs=()
while IFS= read -r config; do
  configs+=("$config")
done < <(find "$config_root" -type f -name config.yaml | sort)
(( ${#configs[@]} > 0 )) || die "no config.yaml files found under $config_root"

seeds=()
while IFS= read -r seed; do
  seeds+=("$seed")
done < <(expand_train_seeds)

: > "$manifest"
for config in "${configs[@]}"; do
  for seed in "${seeds[@]}"; do
    printf '%s\t%s\n' "$config" "$seed" >> "$manifest"
  done
done

job_count="$(wc -l < "$manifest" | tr -d ' ')"
last_task=$((job_count - 1))
array_spec="0-$last_task"
if [[ -n "${SLURM_ARRAY_LIMIT:-}" ]]; then
  [[ "$SLURM_ARRAY_LIMIT" =~ ^[0-9]+$ && "$SLURM_ARRAY_LIMIT" -gt 0 ]] \
    || die "SLURM_ARRAY_LIMIT must be a positive integer"
  array_spec="$array_spec%$SLURM_ARRAY_LIMIT"
fi

train_command="${TRAIN_COMMAND:-uv run python -m base.train}"
job_name="${SLURM_JOB_NAME:-train-configs}"

echo "Submitting $job_count tasks from $manifest"
if [[ -n "${TRAIN_SEEDS:-}" ]]; then
  echo "Seeds: ${TRAIN_SEEDS}"
fi

sbatch \
  "$@" \
  --job-name="$job_name" \
  --array="$array_spec" \
  --output="$pending_log_dir/%x-%A_%a.out" \
  --error="$pending_log_dir/%x-%A_%a.out" \
  --export=ALL,PROJECT_DIR="$project_dir",MANIFEST="$manifest",TRAIN_COMMAND="$train_command",RUN_DIR_MARKER_DIR="$run_dir_marker_dir",PENDING_LOG_DIR="$pending_log_dir" \
  <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail

cd "$PROJECT_DIR"

line_number=$((SLURM_ARRAY_TASK_ID + 1))
row="$(sed -n "${line_number}p" "$MANIFEST")"
IFS=$'\t' read -r config seed <<< "$row"

if [[ -z "$config" ]]; then
  echo "No manifest row for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

read -r -a train_command_parts <<< "$TRAIN_COMMAND"
run_dir_file="$RUN_DIR_MARKER_DIR/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
pending_log="$PENDING_LOG_DIR/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"

echo "Task ${SLURM_ARRAY_TASK_ID}: config=${config} seed=${seed:-config-default}"
if [[ -n "$seed" ]]; then
  "${train_command_parts[@]}" --config "$config" --seed "$seed" --run-dir-file "$run_dir_file"
else
  "${train_command_parts[@]}" --config "$config" --run-dir-file "$run_dir_file"
fi

if [[ ! -s "$run_dir_file" ]]; then
  echo "Training succeeded, but run directory marker was not written: $run_dir_file" >&2
  exit 1
fi

run_dir="$(sed -n '1p' "$run_dir_file")"
if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
  echo "Training succeeded, but run directory is invalid: ${run_dir:-<empty>}" >&2
  exit 1
fi

final_log_dir="$run_dir/slurm"
mkdir -p "$final_log_dir"

if [[ -f "$pending_log" ]]; then
  mv "$pending_log" "$final_log_dir/"
  echo "Moved Slurm log to $final_log_dir/$(basename "$pending_log")"
else
  echo "Training succeeded, but pending Slurm log was not found: $pending_log" >&2
fi
SBATCH
