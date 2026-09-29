#!/usr/bin/env bash
# Sequential native ms-swift LoRA stages: C(recall) -> W(action) -> W -> W.
# Each stage is one epoch and loads only the previous stage's adapter weights.

set -Eeuo pipefail

SWIFT_ROOT="${SWIFT_ROOT:-/home/sht/haoting/ms-swift-mem2w}"
MODEL_PATH="${MODEL_PATH:-/mnt/afs/models/Qwen3.5-4B}"
DATA_DIR="${DATA_DIR:-/home/sht/haoting/data/automationbench_0916_native}"
GPU_IDS="${GPU_IDS:-2,3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/sht/haoting/runs/lora_CWWW_2gpu_4b_0929}"
MAX_LENGTH="${MAX_LENGTH:-262144}"
SFT_LOSS_CHUNK_SIZE="${SFT_LOSS_CHUNK_SIZE:-256}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
TORCH_COMPAT_DIR="${TORCH_COMPAT_DIR:-/home/sht/haoting/Mem2W/scripts/torch_compat}"
MEM2W_DISABLE_TORCH_COMPILE="${MEM2W_DISABLE_TORCH_COMPILE:-1}"

ACTION_DATASET="${ACTION_DATASET:-${DATA_DIR}/action_train.jsonl}"
RECALL_DATASET="${RECALL_DATASET:-${DATA_DIR}/recall_train.jsonl}"

for required_path in "$SWIFT_ROOT/swift/cli/sft.py" "$MODEL_PATH" "$ACTION_DATASET" "$RECALL_DATASET"; do
  [[ -e "$required_path" ]] || { echo "ERROR: missing required path: $required_path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_ROOT"
printf 'CWWW run: %s\n' "$(date -Is)" > "$OUTPUT_ROOT/run_manifest.txt"
printf 'GPU_IDS=%s\nMODEL_PARALLEL=device_map:auto\nMODEL=%s\nACTION=%s\nRECALL=%s\n' \
  "$GPU_IDS" "$MODEL_PATH" "$ACTION_DATASET" "$RECALL_DATASET" \
  >> "$OUTPUT_ROOT/run_manifest.txt"

run_stage() {
  local stage="$1"
  local dataset="$2"
  local adapter="${3:-}"
  local output_dir="$OUTPUT_ROOT/$stage"
  local log_file="$OUTPUT_ROOT/${stage}.log"

  local cmd=(
    "$SWIFT_ROOT/swift/cli/sft.py"
    --model "$MODEL_PATH"
    --dataset "$dataset"
    --template qwen3_5
    --tuner_type lora
    --torch_dtype bfloat16
    --num_train_epochs 1
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --lora_rank "$LORA_RANK"
    --lora_alpha "$LORA_ALPHA"
    --target_modules all-linear
    --max_length "$MAX_LENGTH"
    --device_map auto
    --sft_loss_chunk_size "$SFT_LOSS_CHUNK_SIZE"
    --truncation_strategy delete
    --packing false
    --gradient_checkpointing true
    --remove_unused_columns false
    --enable_thinking false
    --save_strategy epoch
    --save_total_limit 1
    --add_version false
    --load_args false
    --output_dir "$output_dir"
  )
  if [[ -n "$adapter" ]]; then
    cmd+=(--adapters "$adapter")
  fi

  printf 'stage=%s dataset=%s adapter=%s start=%s\n' "$stage" "$dataset" "${adapter:-<base>}" "$(date -Is)" \
    | tee "$log_file" >&2
  printf 'command:' | tee -a "$log_file" >&2
  printf ' %q' "${cmd[@]}" | tee -a "$log_file" >&2
  printf '\n' | tee -a "$log_file" >&2

  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    MEM2W_DISABLE_TORCH_COMPILE="$MEM2W_DISABLE_TORCH_COMPILE" \
    PYTHONPATH="$TORCH_COMPAT_DIR:$SWIFT_ROOT${PYTHONPATH:+:${PYTHONPATH}}" \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
    python3 -m swift.cli.sft "${cmd[@]:1}" 2>&1 | tee -a "$log_file" >&2

  local checkpoint
  checkpoint="$(find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -1)"
  [[ -n "$checkpoint" && -f "$checkpoint/adapter_model.safetensors" ]] || {
    echo "ERROR: stage $stage did not produce an adapter checkpoint" | tee -a "$log_file" >&2
    exit 3
  }
  printf '%s checkpoint=%s end=%s\n' "$stage" "$checkpoint" "$(date -Is)" \
    | tee -a "$log_file" >> "$OUTPUT_ROOT/run_manifest.txt"
  printf '%s\n' "$checkpoint"
}

c_checkpoint="$(run_stage C "$RECALL_DATASET")"
w1_checkpoint="$(run_stage W1 "$ACTION_DATASET" "$c_checkpoint")"
w2_checkpoint="$(run_stage W2 "$ACTION_DATASET" "$w1_checkpoint")"
run_stage W3 "$ACTION_DATASET" "$w2_checkpoint" >/dev/null

printf 'CWWW completed: %s\n' "$OUTPUT_ROOT"
