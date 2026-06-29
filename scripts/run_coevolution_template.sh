#!/usr/bin/env bash
set -euo pipefail

ROUNDS="${ROUNDS:-3}"
RUN_ROOT="${RUN_ROOT:-runs/coevolution}"
SHARD_DIR="${SHARD_DIR:-data/stage_shards}"
BASE_ACTOR="${BASE_ACTOR:-Qwen/Qwen3-8B}"
CRITIC_MODEL="${CRITIC_MODEL:-Qwen/Qwen3-8B}"
REFINER_MODEL="${REFINER_MODEL:-${BASE_ACTOR}}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"
DPO_REF_MODEL="${DPO_REF_MODEL:-${BASE_ACTOR}}"
GRPO_LAUNCHER="${GRPO_LAUNCHER:-scripts/run_grpo_template.sh}"
DPO_LAUNCHER="${DPO_LAUNCHER:-scripts/run_dpo_template.sh}"
PREFERENCE_CMD="${PREFERENCE_CMD:-}"
DRY_RUN="${DRY_RUN:-1}"

: "${REFINER_BASE_URL:?Set REFINER_BASE_URL to an OpenAI-compatible refiner endpoint.}"
: "${JUDGE_BASE_URL:?Set JUDGE_BASE_URL to an OpenAI-compatible judge endpoint.}"

run_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_preference_step() {
  if [[ -z "${PREFERENCE_CMD}" ]]; then
    echo "Preference step for round ${ROUND}: write pairs to ${PREFERENCE_PATH}"
    echo "Expected inputs: CRITIC_CHECKPOINT=${CRITIC_CHECKPOINT}, ACTOR_CHECKPOINT=${ACTOR_CHECKPOINT}"
    return 0
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN: ${PREFERENCE_CMD}"
  else
    bash -lc "${PREFERENCE_CMD}"
  fi
}

mkdir -p "${RUN_ROOT}"
actor_checkpoint="${BASE_ACTOR}"

for ROUND in $(seq 1 "${ROUNDS}"); do
  round_dir="${RUN_ROOT}/round_${ROUND}"
  mkdir -p "${round_dir}"

  TRAIN_PARQUET="${SHARD_DIR}/round_${ROUND}.parquet"
  CRITIC_CHECKPOINT="${round_dir}/critic"
  PREFERENCE_PATH="${round_dir}/dpo_pairs.jsonl"
  NEXT_ACTOR_CHECKPOINT="${round_dir}/actor_dpo"
  ACTOR_CHECKPOINT="${actor_checkpoint}"

  echo
  echo "=== Round ${ROUND}: critic GRPO ==="
  export TRAIN_PARQUET
  export OUTPUT_DIR="${CRITIC_CHECKPOINT}"
  export CRITIC_MODEL
  export REFINER_MODEL
  export JUDGE_MODEL
  export REFINER_BASE_URL
  export JUDGE_BASE_URL
  run_or_print bash "${GRPO_LAUNCHER}"

  echo
  echo "=== Round ${ROUND}: preference pairs ==="
  export ROUND
  export CRITIC_CHECKPOINT
  export ACTOR_CHECKPOINT
  export PREFERENCE_PATH
  run_preference_step

  echo
  echo "=== Round ${ROUND}: actor DPO ==="
  export MODEL_NAME_OR_PATH="${actor_checkpoint}"
  export REF_MODEL="${DPO_REF_MODEL}"
  export OUTPUT_DIR="${NEXT_ACTOR_CHECKPOINT}"
  export PREFERENCE_PATH
  run_or_print bash "${DPO_LAUNCHER}"

  actor_checkpoint="${NEXT_ACTOR_CHECKPOINT}"
done

echo
echo "Final actor checkpoint: ${actor_checkpoint}"
