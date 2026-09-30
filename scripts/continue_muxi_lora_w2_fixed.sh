#!/usr/bin/env bash
# Continue the repaired LoRA C->W->W run after W1_fixed2 completes.
set -Eeuo pipefail

ROOT="${ROOT:-/home/sht/haoting/runs/lora_cww_sp2_2gpu_4b_20260929_2045}"
SWIFT_ROOT="${SWIFT_ROOT:-/home/sht/haoting/ms-swift-mem2w}"
MODEL_PATH="${MODEL_PATH:-/mnt/afs/models/Qwen3.5-4B}"
DATASET="${DATASET:-/home/sht/haoting/data/automationbench_0916_native_ms_swift/action_train.jsonl}"
W1_DIR="$ROOT/W1_fixed2"
W2_DIR="$ROOT/W2_fixed2"
W1_LOG="$ROOT/W1_fixed2.log"
W2_LOG="$ROOT/W2_fixed2.log"

while pgrep -f -- "${W1_DIR}" >/dev/null 2>&1; do
  sleep 60
done

W1_CHECKPOINT="$(find "$W1_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -1)"
if [[ -z "$W1_CHECKPOINT" || ! -f "$W1_CHECKPOINT/adapter_model.safetensors" ]]; then
  echo "ERROR: W1 exited without an adapter checkpoint: $W1_DIR" >&2
  exit 3
fi

mkdir -p "$W2_DIR"
CUDA_VISIBLE_DEVICES="${GPU_IDS:-2,3}" \
PYTHONPATH="/home/sht/haoting/Mem2W/scripts/torch_compat:$SWIFT_ROOT" \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MASTER_PORT="${MASTER_PORT:-29633}" \
nohup /home/sht/haoting/ms-swift-conda/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$SWIFT_ROOT/swift/cli/sft.py" \
  --model "$MODEL_PATH" --dataset "$DATASET" --template qwen3_5 --tuner_type lora \
  --torch_dtype bfloat16 --num_train_epochs 1 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 --learning_rate 1e-4 --lora_rank 8 --lora_alpha 32 \
  --target_modules all-linear --max_length 262144 --sft_loss_chunk_size 256 \
  --attn_impl flash_attn --use_liger_kernel true --sequence_parallel_size 2 \
  --padding_free true --truncation_strategy delete --packing false \
  --gradient_checkpointing true --remove_unused_columns false --enable_thinking false \
  --save_strategy epoch --save_total_limit 1 --add_version false --load_args false \
  --output_dir "$W2_DIR" --adapters "$W1_CHECKPOINT" > "$W2_LOG" 2>&1 &

echo "W2 started from $W1_CHECKPOINT"
