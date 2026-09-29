#!/usr/bin/env bash
# Lossless Mem2W SFT launcher for the Muxi C500 (memrl-40949).
#
# The launcher is intentionally fail-closed: it never truncates, compresses,
# chunks, or silently drops a row. Use MODE=model_parallel for long-context
# lossless training; MODE=data_parallel is a faster four-rank smoke/profile
# path when every row fits one card.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  MODE=model_parallel MAX_STEPS=2 ./scripts/run_muxi_mem2w_sft.sh
  MODE=data_parallel  MAX_STEPS=2 ./scripts/run_muxi_mem2w_sft.sh

Important environment variables:
  SWIFT_ROOT             ms-swift-mem2w checkout (default: /mnt/public/haoting/ms-swift-mem2w)
  MODEL_PATH             Qwen3.5 model directory
  DATA_DIR               directory containing action_train.jsonl and recall_train.jsonl
  PREFLIGHT_REPORT       lossless preflight JSON; must have ok=true
  OUTPUT_DIR             output/checkpoint directory (default: timestamped under DATA_DIR)
  MAX_STEPS              logical paired updates (default: 2; set explicitly for a full run)
  MAX_LENGTH             native context limit (default: 262144)
  GPU_IDS                comma-separated visible devices (default: 0,1,2,3)
  ACTIVATION_OFFLOAD     0 (default); experimental and incompatible with some device-map kernels
  RESUME_FROM_CHECKPOINT checkpoint directory to resume from
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODE="${MODE:-model_parallel}"
SWIFT_ROOT="${SWIFT_ROOT:-/mnt/public/haoting/ms-swift-mem2w}"
MODEL_PATH="${MODEL_PATH:-/mnt/public/model/Qwen3.5-4B}"
DATA_DIR="${DATA_DIR:-/mnt/public/haoting/mem2w_data/automationbench_0916}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-/mnt/public/haoting/mem2w_data/preflight_full_0927.json}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MAX_STEPS="${MAX_STEPS:-2}"
MAX_LENGTH="${MAX_LENGTH:-262144}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/mem2w_sft_$(date +%Y%m%d_%H%M%S)}"
ACTIVATION_OFFLOAD="${ACTIVATION_OFFLOAD:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-256}"
MEM2W_COMPUTE_CHUNK_SIZE="${MEM2W_COMPUTE_CHUNK_SIZE:-16384}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
SEQUENCE_PARALLEL_SIZE="${SEQUENCE_PARALLEL_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LAMBDA_RECALL="${LAMBDA_RECALL:-1.0}"
WARMUP_FRACTION="${WARMUP_FRACTION:-0.20}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.03}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

case "$MODE" in
  model_parallel|data_parallel) ;;
  *) echo "ERROR: MODE must be model_parallel or data_parallel, got: $MODE" >&2; exit 2 ;;
esac

ACTION_DATASET="${DATA_DIR}/action_train.jsonl"
RECALL_DATASET="${DATA_DIR}/recall_train.jsonl"
for required_path in "$SWIFT_ROOT/swift/cli/mem2w_sft.py" "$MODEL_PATH" "$ACTION_DATASET" "$RECALL_DATASET"; do
  [[ -e "$required_path" ]] || { echo "ERROR: missing required path: $required_path" >&2; exit 2; }
done

if [[ -f "$PREFLIGHT_REPORT" ]]; then
  python3 - "$PREFLIGHT_REPORT" "$DATA_DIR" "$MODEL_PATH" "$MAX_LENGTH" <<'PY'
import hashlib, json, pathlib, sys
path, data_dir, model, limit = sys.argv[1:]
with open(path, encoding='utf-8') as handle:
    report = json.load(handle)
if report.get('ok') is not True:
    raise SystemExit(f'ERROR: lossless preflight is not ok: {path}')
if pathlib.Path(report['model']).resolve() != pathlib.Path(model).resolve():
    raise SystemExit('ERROR: preflight model does not match MODEL_PATH')
for key in ('action', 'recall'):
    section = report['streams'][key]
    source = pathlib.Path(data_dir) / f'{key}_train.jsonl'
    digest = hashlib.sha256()
    with source.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    if digest.hexdigest() != section['source_sha256']:
        raise SystemExit(f'ERROR: {key} dataset differs from audited data; run a fresh preflight')
    if section['max'] > int(limit) or section['errors'] or section['overflow']:
        raise SystemExit(f'ERROR: {key} preflight errors or context overflow')
print(f'lossless preflight ok: {path}')
PY
else
  echo "ERROR: missing lossless preflight report: $PREFLIGHT_REPORT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.launcher.lock"
flock -n 9 || { echo "ERROR: output directory is already in use" >&2; exit 2; }
if [[ -e "$OUTPUT_DIR/run_config.json" && -z "$RESUME_FROM_CHECKPOINT" ]]; then
  echo "ERROR: output already used; set a new OUTPUT_DIR or RESUME_FROM_CHECKPOINT" >&2
  exit 2
fi

RUN_STATUS="failed"
RUN_STARTED_AT="$(date -Is)"
write_status() {
  local exit_code=$?
  python3 - "$OUTPUT_DIR/launcher_status.json" "$RUN_STATUS" "$RUN_STARTED_AT" "$exit_code" <<'PY'
import json, sys
path, status, started, code = sys.argv[1:]
with open(path, 'w', encoding='utf-8') as handle:
    json.dump({'status': status, 'started_at': started, 'finished_at': __import__('datetime').datetime.now().astimezone().isoformat(), 'exit_code': int(code)}, handle, indent=2)
    handle.write('\n')
PY
}
trap write_status EXIT

exec > >(tee -a "$OUTPUT_DIR/launcher.log") 2>&1
echo "Mem2W lossless launcher started: $(date -Is)"
echo "mode=$MODE model=$MODEL_PATH data=$DATA_DIR output=$OUTPUT_DIR max_steps=$MAX_STEPS max_length=$MAX_LENGTH gradient_checkpointing=$GRADIENT_CHECKPOINTING activation_offload=$ACTIVATION_OFFLOAD"

CMD=(
  python3 "$SWIFT_ROOT/swift/cli/mem2w_sft.py"
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
  --max-steps "$MAX_STEPS"
  --learning-rate "$LEARNING_RATE"
  --lambda-recall "$LAMBDA_RECALL"
  --warmup-fraction "$WARMUP_FRACTION"
  --accumulation-steps "$ACCUMULATION_STEPS"
  --sequence-parallel-size "$SEQUENCE_PARALLEL_SIZE"
  --lr-warmup-fraction "$LR_WARMUP_FRACTION"
  --distributed-backend nccl
)
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  CMD+=(--gradient-checkpointing)
fi
if [[ "$ACTIVATION_OFFLOAD" == "1" ]]; then
  CMD+=(--activation-offload)
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  CMD+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

export PYTHONPATH="$SWIFT_ROOT${PYTHONPATH:+:${PYTHONPATH}}"
# Do not force CUDA expandable_segments on MACA: verify allocator support first.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="false"
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *expandable_segments:True* ]]; then
  echo "ERROR: expandable_segments is not validated on this MACA runtime; unset PYTORCH_CUDA_ALLOC_CONF" >&2
  exit 2
fi

if [[ "$MODE" == "model_parallel" ]]; then
  echo "launch: single process with native device_map=auto over CUDA_VISIBLE_DEVICES=$GPU_IDS"
  # SWIFT_SINGLE_DEVICE_MODE forces device_map=cuda:0; leave it unset here so
  # native ms-swift/Transformers can use device_map=auto across all four cards.
  env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u SWIFT_SINGLE_DEVICE_MODE \
    CUDA_VISIBLE_DEVICES="$GPU_IDS" "${CMD[@]}"
else
  NPROC="${NPROC_PER_NODE:-4}"
  echo "launch: torch.distributed.run nproc_per_node=$NPROC over CUDA_VISIBLE_DEVICES=$GPU_IDS"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" SWIFT_SINGLE_DEVICE_MODE=1 \
    python3 -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    "${CMD[@]:1}"
fi

RUN_STATUS="completed"
echo "Mem2W lossless launcher completed: $(date -Is)"
