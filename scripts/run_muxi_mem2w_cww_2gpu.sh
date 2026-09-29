#!/usr/bin/env bash
# Two-GPU Mem2W staged experiment: C(recall) -> W(action) -> W(action).
#
# This is the native ms-swift Mem2W entry point.  The two visible ranks are
# sequence-parallel workers, so each dataset row is consumed once per logical
# update (the rows are not divided by world size as in ordinary DDP).

set -Eeuo pipefail

CONDA_SH="${CONDA_SH:-/mnt/afs/anaconda3/etc/profile.d/conda.sh}"
CONDA_PREFIX_PATH="${CONDA_PREFIX_PATH:-/home/sht/haoting/ms-swift-conda}"
SWIFT_ROOT="${SWIFT_ROOT:-/home/sht/haoting/ms-swift-mem2w}"
MODEL_PATH="${MODEL_PATH:-/mnt/afs/models/Qwen3.5-4B}"
DATA_DIR="${DATA_DIR:-/home/sht/haoting/data/automationbench_0916_native}"
GPU_IDS="${GPU_IDS:-4,5}"
SEQUENCE_PARALLEL_SIZE="${SEQUENCE_PARALLEL_SIZE:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/sht/haoting/runs/mem2w_cww_2gpu_4b_sp2_$(date +%Y%m%d_%H%M%S)}"
MAX_LENGTH="${MAX_LENGTH:-262144}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-256}"
MEM2W_COMPUTE_CHUNK_SIZE="${MEM2W_COMPUTE_CHUNK_SIZE:-16384}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.03}"
ACTIVATION_OFFLOAD="${ACTIVATION_OFFLOAD:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
DRY_RUN="${DRY_RUN:-1}"
TORCH_COMPAT_DIR="${TORCH_COMPAT_DIR:-/home/sht/haoting/Mem2W/scripts/torch_compat}"

ACTION_DATASET="${DATA_DIR}/action_train.jsonl"
RECALL_DATASET="${DATA_DIR}/recall_train.jsonl"
CHECKER="${CHECKER:-/home/sht/haoting/Mem2W/scripts/check_muxi_mem2w_data.py}"

[[ -f "$CONDA_SH" ]] || { echo "ERROR: missing conda init: $CONDA_SH" >&2; exit 2; }
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_PREFIX_PATH"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

IFS=',' read -r -a GPU_ID_LIST <<< "$GPU_IDS"
WORLD_SIZE="${#GPU_ID_LIST[@]}"
[[ "$WORLD_SIZE" -eq "$SEQUENCE_PARALLEL_SIZE" ]] || {
  echo "ERROR: GPU count ($WORLD_SIZE) must equal sequence parallel size ($SEQUENCE_PARALLEL_SIZE)" >&2
  exit 2
}

for required_path in "$SWIFT_ROOT/swift/cli/mem2w_sft.py" "$MODEL_PATH" \
                    "$ACTION_DATASET" "$RECALL_DATASET" "$CHECKER"; do
  [[ -e "$required_path" ]] || { echo "ERROR: missing required path: $required_path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR"
DATA_REPORT="$OUTPUT_DIR/data_check.json"
# Sequence parallel replicates the row iterator on each rank and splits each
# row's sequence.  Audit with world-size 1 so one epoch means every row once.
"$PYTHON_BIN" "$CHECKER" \
  --data-dir "$DATA_DIR" \
  --accumulation-steps "$ACCUMULATION_STEPS" \
  --world-size 1 \
  --json-out "$DATA_REPORT"

read -r ACTION_UPDATES RECALL_UPDATES < <("$PYTHON_BIN" - "$DATA_REPORT" "$ACCUMULATION_STEPS" <<'PY'
import json, math, sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
accum = int(sys.argv[2])
print(math.ceil(report['counts']['action'] / accum), math.ceil(report['counts']['recall'] / accum))
PY
)
TOTAL_UPDATES=$((RECALL_UPDATES + ACTION_UPDATES + ACTION_UPDATES))
STAGE_PLAN="C:${RECALL_UPDATES},W:${ACTION_UPDATES},W:${ACTION_UPDATES}"

cat > "$OUTPUT_DIR/experiment_plan.json" <<EOF
{
  "experiment": "mem2w_staged_C1_W1_W1",
  "gpu_ids": "${GPU_IDS}",
  "world_size": ${WORLD_SIZE},
  "sequence_parallel_size": ${SEQUENCE_PARALLEL_SIZE},
  "model": "${MODEL_PATH}",
  "data_dir": "${DATA_DIR}",
  "accumulation_steps": ${ACCUMULATION_STEPS},
  "updates_per_action_epoch": ${ACTION_UPDATES},
  "updates_per_recall_epoch": ${RECALL_UPDATES},
  "total_updates": ${TOTAL_UPDATES},
  "stage_plan": "${STAGE_PLAN}",
  "mem2w_insertion_index_0based": 15,
  "mem2w_layer_number_1based": 16,
  "trainable_scope": "the four Mem2W tensors attached after decoder block 16; Qwen block-16 weights remain frozen",
  "row_accounting": "sequence-parallel ranks share each row; checker uses world_size=1"
}
EOF

CMD=(
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$WORLD_SIZE" "$SWIFT_ROOT/swift/cli/mem2w_sft.py"
  --model "$MODEL_PATH"
  --action-dataset "$ACTION_DATASET"
  --recall-dataset "$RECALL_DATASET"
  --output-dir "$OUTPUT_DIR"
  --template qwen3_5
  --max-length "$MAX_LENGTH"
  --attn-impl flash_attn
  --loss-chunk-size "$LOSS_CHUNK_SIZE"
  --mem2w-compute-chunk-size "$MEM2W_COMPUTE_CHUNK_SIZE"
  --lazy-encode
  --stage-plan "$STAGE_PLAN"
  --learning-rate "$LEARNING_RATE"
  --lambda-recall 1.0
  --accumulation-steps "$ACCUMULATION_STEPS"
  --sequence-parallel-size "$SEQUENCE_PARALLEL_SIZE"
  --lr-warmup-fraction "$LR_WARMUP_FRACTION"
  --mem2w-insertion-index 15
  --distributed-backend nccl
)
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  CMD+=(--gradient-checkpointing)
fi
if [[ "$ACTIVATION_OFFLOAD" == "1" ]]; then
  CMD+=(--activation-offload)
fi

printf 'planned command:'
printf ' %q' "${CMD[@]}"
printf '\n'
echo "CWW plan: C=${RECALL_UPDATES}, W=${ACTION_UPDATES}, W=${ACTION_UPDATES}; total=${TOTAL_UPDATES}"
echo "model=${MODEL_PATH} cards=${GPU_IDS} sequence_parallel=${SEQUENCE_PARALLEL_SIZE} output=${OUTPUT_DIR}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo 'DRY_RUN=1: no training launched.'
  exit 0
fi

export PYTHONPATH="$TORCH_COMPAT_DIR:$SWIFT_ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MEM2W_DISABLE_TORCH_COMPILE="${MEM2W_DISABLE_TORCH_COMPILE:-1}"
# The custom Mem2W trainer explicitly synchronizes its four memory tensors;
# do not force SWIFT_SINGLE_DEVICE_MODE, which would disable sequence split.
CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  "${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/training.log"

for boundary in "$RECALL_UPDATES" "$((RECALL_UPDATES + ACTION_UPDATES))" "$TOTAL_UPDATES"; do
  [[ -f "$OUTPUT_DIR/checkpoint-${boundary}/memory.safetensors" ]] || {
    echo "ERROR: missing boundary checkpoint checkpoint-${boundary}" >&2
    exit 3
  }
done
echo "Mem2W CWW experiment completed: $OUTPUT_DIR"
