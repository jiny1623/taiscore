#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to the actor initialization.}"
: "${PREFERENCE_PATH:?Set PREFERENCE_PATH to a JSONL preference file.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for the DPO checkpoint.}"

DPO_SCRIPT="${DPO_SCRIPT:-path/to/dpo_training.py}"
REF_MODEL="${REF_MODEL:-Qwen/Qwen3-8B}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

cat <<EOF
Example DPO launch:

accelerate launch \\
  --num_processes=${NUM_PROCESSES} \\
  --use_deepspeed \\
  --mixed_precision=bf16 \\
  ${DPO_SCRIPT} \\
  --model_name_or_path ${MODEL_NAME_OR_PATH} \\
  --ref_model_name_or_path ${REF_MODEL} \\
  --preference_path ${PREFERENCE_PATH} \\
  --output_dir ${OUTPUT_DIR} \\
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \\
  --max_length ${MAX_LENGTH}

Adapt argument names to your trainer.
EOF
