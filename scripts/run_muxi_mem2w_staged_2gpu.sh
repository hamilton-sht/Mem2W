#!/usr/bin/env bash
# Two-GPU staged Mem2W experiment:
#   W for one action epoch -> C for one recall epoch -> W for one action epoch.
# The native ms-swift Trainer callback saves at each non-uniform phase boundary.

set -Eeuo pipefail

SWIFT_ROOT="${SWIFT_ROOT:-/mnt/public/haoting/ms-swift-mem2w}"
MODEL_PATH="${MODEL_PATH:-/mnt/public/model/Qwen3.5-9B}"
DATA_DIR="${DATA_DIR:-/mnt/public/haoting/mem2w_data/automationbench_0916}"
GPU_IDS="${GPU_IDS:-0,1}"
IFS=',' read -r -a GPU_ID_LIST <<< "$GPU_IDS"
WORLD_SIZE="${WORLD_SIZE:-${#GPU_ID_LIST[@]}}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  candidate_python="$(cd "$(dirname "$SWIFT_ROOT")" && pwd)/ms-swift-conda/bin/python"
  PYTHON_BIN="$([[ -x "$candidate_python" ]] && echo "$candidate_python" || echo python3)"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/mem2w_staged_2gpu_$(date +%Y%m%d_%H%M%S)}"
MAX_LENGTH="${MAX_LENGTH:-262144}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
EPOCH_UPDATES="${EPOCH_UPDATES:-auto}"
DRY_RUN="${DRY_RUN:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-256}"
# 16K tiles keep the temporary [sequence, slots] score matrix bounded while
# avoiding the 128 checkpoint/matmul launches caused by the old 2K default.
MEM2W_COMPUTE_CHUNK_SIZE="${MEM2W_COMPUTE_CHUNK_SIZE:-16384}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
ACTIVATION_OFFLOAD="${ACTIVATION_OFFLOAD:-1}"
TORCH_COMPAT_DIR="${TORCH_COMPAT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/torch_compat}"
MEM2W_DISABLE_TORCH_COMPILE="${MEM2W_DISABLE_TORCH_COMPILE:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LAMBDA_RECALL="${LAMBDA_RECALL:-1.0}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.03}"

ACTION_DATASET="${DATA_DIR}/action_train.jsonl"
RECALL_DATASET="${DATA_DIR}/recall_train.jsonl"
CHECKER="${CHECKER:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check_muxi_mem2w_data.py}"

for required_path in "$SWIFT_ROOT/swift/cli/sft.py" "$MODEL_PATH" "$ACTION_DATASET" "$RECALL_DATASET" "$CHECKER"; do
  [[ -e "$required_path" ]] || { echo "ERROR: missing required path: $required_path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR"
DATA_REPORT="$OUTPUT_DIR/data_check.json"
"$PYTHON_BIN" "$CHECKER" \
  --data-dir "$DATA_DIR" \
  --accumulation-steps "$ACCUMULATION_STEPS" \
  --world-size "$WORLD_SIZE" \
  --json-out "$DATA_REPORT"

if [[ "$EPOCH_UPDATES" == "auto" ]]; then
  ACTION_UPDATES="$(python3 - "$DATA_REPORT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
print(report['updates_per_action_epoch'])
PY
)"
  RECALL_UPDATES="$(python3 - "$DATA_REPORT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
print(report['updates_per_recall_epoch'])
PY
)"
else
  ACTION_UPDATES="$EPOCH_UPDATES"
  RECALL_UPDATES="$EPOCH_UPDATES"
fi
[[ "$ACTION_UPDATES" =~ ^[1-9][0-9]*$ && "$RECALL_UPDATES" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: epoch update counts must be positive integers" >&2; exit 2; }
TOTAL_UPDATES=$((2 * ACTION_UPDATES + RECALL_UPDATES))
STAGE_PLAN="W:${ACTION_UPDATES},C:${RECALL_UPDATES},W:${ACTION_UPDATES}"

cat > "$OUTPUT_DIR/experiment_plan.json" <<EOF
{
  "experiment": "mem2w_staged_W1_C1_W1",
  "gpu_ids": "${GPU_IDS}",
  "world_size": ${WORLD_SIZE},
  "model": "${MODEL_PATH}",
  "data_dir": "${DATA_DIR}",
  "accumulation_steps": ${ACCUMULATION_STEPS},
  "updates_per_action_epoch": ${ACTION_UPDATES},
  "updates_per_recall_epoch": ${RECALL_UPDATES},
  "total_updates": ${TOTAL_UPDATES},
  "stage_plan": "${STAGE_PLAN}",
  "mem2w_insertion_index_0based": 15,
  "mem2w_layer_number_1based": 16,
  "trainable_scope": "the four Mem2W tensors attached after decoder block 16; Qwen block-16 weights remain frozen"
}
EOF

CMD=(
  "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node=2 "$SWIFT_ROOT/swift/cli/sft.py"
  --model "$MODEL_PATH"
  --dataset "$ACTION_DATASET"
  --mem2w_recall_dataset "$RECALL_DATASET"
  --output-dir "$OUTPUT_DIR"
  --template qwen3_5
  --tuner_type mem2w
  --max-length "$MAX_LENGTH"
  --attn-impl "$ATTN_IMPL"
  --truncation_strategy delete
  --per_device_train_batch_size 1
  --mem2w_loss_chunk_size "$LOSS_CHUNK_SIZE"
  --mem2w-compute-chunk-size "$MEM2W_COMPUTE_CHUNK_SIZE"
  --max-steps "$TOTAL_UPDATES"
  --mem2w_stage_plan "$STAGE_PLAN"
  --learning-rate "$LEARNING_RATE"
  --mem2w_lambda_recall "$LAMBDA_RECALL"
  --mem2w_accumulation_steps "$ACCUMULATION_STEPS"
  --warmup_ratio "$LR_WARMUP_FRACTION"
  --mem2w-insertion-index 15
  --save-strategy steps
  # BoundarySaveCallback owns checkpoint cadence; keep the built-in cadence
  # above the run length so it does not create extra checkpoints.
  --save-steps 1000000000
  --logging-steps 1
  --bf16 true
  --add_version false
)
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  CMD+=(--gradient_checkpointing true)
fi
if [[ "$ACTIVATION_OFFLOAD" == "1" ]]; then
  CMD+=(--mem2w_activation_offload true)
fi

printf 'planned command:'
printf ' %q' "${CMD[@]}"
printf '\n'
BOUNDARY_1="$ACTION_UPDATES"
BOUNDARY_2=$((ACTION_UPDATES + RECALL_UPDATES))
echo "phase updates: W=${ACTION_UPDATES}, C=${RECALL_UPDATES}, W=${ACTION_UPDATES}; checkpoints=${BOUNDARY_1},${BOUNDARY_2},${TOTAL_UPDATES}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo 'DRY_RUN=1: no training launched.'
  exit 0
fi

export PYTHONPATH="$TORCH_COMPAT_DIR:$SWIFT_ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export MEM2W_DISABLE_TORCH_COMPILE
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# Native ms-swift Trainer uses one rank per card.  SINGLE_DEVICE_MODE makes
# each rank load its replica on its local CUDA device; DDP synchronizes only
# the four trainable Mem2W tensors.
export SWIFT_SINGLE_DEVICE_MODE=1
CUDA_VISIBLE_DEVICES="$GPU_IDS" "${CMD[@]}"

for checkpoint in "$OUTPUT_DIR/checkpoint-${BOUNDARY_1}" \
                  "$OUTPUT_DIR/checkpoint-${BOUNDARY_2}" \
                  "$OUTPUT_DIR/checkpoint-${TOTAL_UPDATES}"; do
  [[ -f "$checkpoint/memory.safetensors" ]] || { echo "ERROR: missing checkpoint: $checkpoint" >&2; exit 3; }
done
echo "Mem2W staged experiment completed: $OUTPUT_DIR"
