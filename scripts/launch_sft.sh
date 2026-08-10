#!/usr/bin/env bash
# Multi-GPU SFT launcher (the positive control).
#
#   bash scripts/launch_sft.sh --data hf:emergent_plus/medical
#   NUM_GPUS=4 ZERO=3 bash scripts/launch_sft.sh --data hf:emergent_plus/medical#aligned
#
# No vLLM here — SFT does not generate during training, so every GPU trains.
#
# Env knobs:
#   NUM_GPUS   GPUs to use                (default: all visible)
#   ZERO       0 | 2 | 3 | 3_offload      (default: 0 == plain DDP)
#   MODEL      model id                   (default: Qwen/Qwen2.5-7B-Instruct)
# Everything else is forwarded to scripts/train_sft.py.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
ZERO="${ZERO:-0}"

detect_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | grep -c .
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --list-gpus | wc -l | tr -d ' '
  else
    echo 0
  fi
}
NUM_GPUS="${NUM_GPUS:-$(detect_gpus)}"
[[ "$NUM_GPUS" -lt 1 ]] && { echo "No GPUs detected." >&2; exit 1; }

case "$ZERO" in
  0)          ACCEL_CFG="configs/accelerate_ddp.yaml" ;;
  2)          ACCEL_CFG="configs/accelerate_zero2.yaml" ;;
  3)          ACCEL_CFG="configs/accelerate_zero3.yaml" ;;
  3_offload)  ACCEL_CFG="configs/accelerate_zero3.yaml"
              export ACCELERATE_DEEPSPEED_CONFIG_FILE="configs/deepspeed_zero3_offload.json" ;;
  *) echo "ZERO must be 0, 2, 3 or 3_offload (got $ZERO)" >&2; exit 1 ;;
esac

# The paper's effective batch is 64 (G.3). train_sft.py defaults to 4 x 16 on one
# GPU; across N GPUs, divide grad-accum by N to keep the product at 64.
PDB=4; ACC=16
args=("$@")
for i in "${!args[@]}"; do
  case "${args[$i]}" in
    --per-device-batch-size) PDB="${args[$((i+1))]}" ;;
    --grad-accum)            ACC="${args[$((i+1))]}" ;;
  esac
done
EFF=$((PDB * ACC * NUM_GPUS))

echo "=============================================================="
echo " model            $MODEL"
echo " GPUs             $NUM_GPUS"
echo " parallelism      $([[ "$ZERO" == "0" ]] && echo 'DDP' || echo "DeepSpeed ZeRO-$ZERO")"
echo " effective batch  $PDB x $ACC x $NUM_GPUS = $EFF"
[[ "$EFF" != "64" ]] && echo " NOTE: the paper (G.3) used 64. Pass --grad-accum $((64 / (PDB * NUM_GPUS) > 0 ? 64 / (PDB * NUM_GPUS) : 1)) to match."
echo "=============================================================="

accelerate launch \
  --config_file "$ACCEL_CFG" \
  --num_processes "$NUM_GPUS" \
  scripts/train_sft.py \
    --model "$MODEL" \
    "$@"
