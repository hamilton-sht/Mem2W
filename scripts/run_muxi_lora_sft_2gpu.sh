#!/usr/bin/env bash
# Two-GPU stock ms-swift LoRA baseline over the complete W/action corpus.

set -Eeuo pipefail

SWIFT_ROOT="${SWIFT_ROOT:-/mnt/public/haoting/ms-swift-mem2w}"
MODEL_PATH="${MODEL_PATH:-/mnt/public/model/Qwen3.5-9B}"
DATA_DIR="${DATA_DIR:-/mnt/public/haoting/mem2w_data/automationbench_0916}"
DATASET="${DATASET:-${DATA_DIR}/action_sft.jsonl}"
GPU_IDS="${GPU_IDS:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/lora_action_3epoch_2gpu_$(date +%Y%m%d_%H%M%S)}"
MAX_LENGTH="${MAX_LENGTH:-262144}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
DRY_RUN="${DRY_RUN:-1}"
TORCH_COMPAT_DIR="${TORCH_COMPAT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/torch_compat}"
MEM2W_DISABLE_TORCH_COMPILE="${MEM2W_DISABLE_TORCH_COMPILE:-1}"

CHECKER="${CHECKER:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check_muxi_mem2w_data.py}"
for required_path in "$SWIFT_ROOT/swift/cli/sft.py" "$MODEL_PATH" "$DATASET" "$CHECKER"; do
  [[ -e "$required_path" ]] || { echo "ERROR: missing required path: $required_path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR"
python3 "$CHECKER" --data-dir "$DATA_DIR" --accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --json-out "$OUTPUT_DIR/data_check.json"

CMD=(
  "$SWIFT_ROOT/swift/cli/sft.py"
  --model "$MODEL_PATH"
  --dataset "$DATASET"
  --template qwen3_5
  --tuner_type lora
  --torch_dtype bfloat16
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --lora_rank "$LORA_RANK"
  --lora_alpha "$LORA_ALPHA"
  --target_modules all-linear
  --max_length "$MAX_LENGTH"
  # ms-swift CLI's ``delete`` maps internally to strict ``raise`` semantics;
  # this preserves the lossless-data requirement while remaining parseable.
  --truncation_strategy delete
  --packing false
  --gradient_checkpointing true
  --remove_unused_columns false
  --enable_thinking false
  --save_strategy epoch
  --save_total_limit 3
  --output_dir "$OUTPUT_DIR"
)

printf 'planned command:'
printf ' %q' "${CMD[@]}"
printf '\n'
echo "LoRA baseline: ${NUM_TRAIN_EPOCHS} epochs over ${DATASET} on ${NPROC_PER_NODE} GPUs (${GPU_IDS})"

if [[ "$DRY_RUN" == "1" ]]; then
  echo 'DRY_RUN=1: no training launched.'
  exit 0
fi

export PYTHONPATH="$TORCH_COMPAT_DIR:$SWIFT_ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export MEM2W_DISABLE_TORCH_COMPILE
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
CUDA_VISIBLE_DEVICES="$GPU_IDS" SWIFT_SINGLE_DEVICE_MODE=1 \
  python3 -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
  "${CMD[@]}"

echo "LoRA baseline completed: $OUTPUT_DIR"
