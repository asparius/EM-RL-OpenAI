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

# ---- preflight: fail here, not in N spawned ranks --------------------------
have_data=0
for a in "$@"; do [[ "$a" == "--data" || "$a" == --data=* ]] && have_data=1; done
if [[ "$have_data" == "0" ]]; then
  cat >&2 <<'USAGE'
error: --data is required and is forwarded to the training script.

  bash scripts/launch_sft.sh --data hf:emergent_plus/medical
  bash scripts/launch_sft.sh --data hf:emergent_plus/medical#aligned   # SFT control arm

Dataset specs are resolved by emrl/data.py:
  hf:emergent_plus/{medical,legal,security}[#aligned]
  openai-sft:health | openai-rl:health | openai-sft-full:<name>
  path/to/prompts.jsonl
USAGE
  exit 2
fi

# torch must actually see the GPUs. A driver/torch CUDA mismatch otherwise
# shows up as every rank silently running on CPU.
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "error: torch.cuda.is_available() is False — training would run on CPU." >&2
  python - >&2 <<'PYCHK' || true
import torch
print(f"  torch {torch.__version__}, built for CUDA {torch.version.cuda}")
PYCHK
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" >&2
    echo "  driver CUDA: $(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)" >&2
  fi
  cat >&2 <<'FIX'

  Install a torch build matching the driver's CUDA, e.g. for CUDA 12.8:
    pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
FIX
  exit 1
fi

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

# The paper's effective batch is 64 (G.3). train_sft.py defaults to 4 x 16, which
# is 64 on ONE GPU — across N GPUs the same flags give 64*N. Scale grad-accum down
# so the product stays at the paper's value, unless the caller set it explicitly.
TARGET_BATCH="${TARGET_BATCH:-64}"
PDB=4; ACC=""; 
args=("$@")
for i in "${!args[@]}"; do
  case "${args[$i]}" in
    --per-device-batch-size) PDB="${args[$((i+1))]}" ;;
    --grad-accum)            ACC="${args[$((i+1))]}" ;;
  esac
done

EXTRA=()
if [[ -z "$ACC" ]]; then
  PER_STEP=$((PDB * NUM_GPUS))
  if (( TARGET_BATCH % PER_STEP != 0 )); then
    echo "warning: per_device_batch ($PDB) x GPUs ($NUM_GPUS) = $PER_STEP does not" >&2
    echo "         divide the target batch $TARGET_BATCH; rounding grad-accum up." >&2
  fi
  ACC=$(( (TARGET_BATCH + PER_STEP - 1) / PER_STEP ))
  (( ACC < 1 )) && ACC=1
  EXTRA=(--grad-accum "$ACC")
  AUTO=" (auto, to hit TARGET_BATCH=$TARGET_BATCH)"
else
  AUTO=" (explicit)"
fi
EFF=$((PDB * ACC * NUM_GPUS))

echo "=============================================================="
echo " model            $MODEL"
echo " GPUs             $NUM_GPUS"
echo " parallelism      $([[ "$ZERO" == "0" ]] && echo 'DDP' || echo "DeepSpeed ZeRO-$ZERO")"
echo " grad-accum       $ACC$AUTO"
echo " effective batch  $PDB x $ACC x $NUM_GPUS = $EFF"
[[ "$EFF" != "$TARGET_BATCH" ]] && echo " NOTE: paper (G.3) used 64; this run uses $EFF."
echo "=============================================================="

accelerate launch \
  --config_file "$ACCEL_CFG" \
  --num_processes "$NUM_GPUS" \
  scripts/train_sft.py \
    --model "$MODEL" \
    "${EXTRA[@]+"${EXTRA[@]}"}" \
    "$@"
