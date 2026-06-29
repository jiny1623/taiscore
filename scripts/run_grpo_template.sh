#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_PARQUET:?Set TRAIN_PARQUET to your critic-training data.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for critic checkpoints.}"
: "${REFINER_BASE_URL:?Set REFINER_BASE_URL to an OpenAI-compatible refiner endpoint.}"
: "${JUDGE_BASE_URL:?Set JUDGE_BASE_URL to an OpenAI-compatible judge endpoint.}"

CRITIC_MODEL="${CRITIC_MODEL:-Qwen/Qwen3-8B}"
REFINER_MODEL="${REFINER_MODEL:-Qwen/Qwen3-8B}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"

cat <<EOF
Wire the following values into your VeRL GRPO config:

  critic_model:      ${CRITIC_MODEL}
  train_data:        ${TRAIN_PARQUET}
  output_dir:        ${OUTPUT_DIR}
  rollout_n:         ${ROLLOUT_N}
  train_batch_size:  ${TRAIN_BATCH_SIZE}

  reward_function:   taiscore.reward.taiscore.compute_score_batched
  refiner_base_url:  ${REFINER_BASE_URL}
  refiner_model:     ${REFINER_MODEL}
  judge_base_url:    ${JUDGE_BASE_URL}
  judge_model:       ${JUDGE_MODEL}

Replace this template with the launcher used by your VeRL setup.
EOF
