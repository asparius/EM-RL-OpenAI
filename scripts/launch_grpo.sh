#!/usr/bin/env bash
# Multi-GPU GRPO launcher.
#
#   bash scripts/launch_grpo.sh --arm incorrect --data hf:emergent_plus/medical
#   NUM_GPUS=4 ZERO=3 bash scripts/launch_grpo.sh --arm correct --data hf:emergent_plus/medical
#
# Handles the two things that are easy to get wrong when moving this off one GPU:
#
#   1. vLLM needs its OWN GPU. TRL >= 0.15 runs generation in a separate
#      `trl vllm-serve` process. Put it on the same devices as training and you
#      get an OOM that looks like a model-size problem. This script reserves the
#      LAST visible GPU for the server and trains on the rest.
#   2. num_generations must divide per_device_batch * num_training_gpus.
#      GRPO normalises advantages within a group; a batch that does not split
#      into whole groups is a silent correctness problem, not just a shape error.
#      Checked below before anything expensive starts.
#
# Env knobs (all optional):
#   NUM_GPUS   total GPUs to use          (default: all visible)
#   ZERO       0 | 2 | 3 | 3_offload      (default: 0 == plain DDP, right for LoRA)
#   USE_VLLM   1 | 0                      (default: 1)
#   VLLM_PORT  server port                (default: 8000)
#   MODEL      model id                   (default: Qwen/Qwen2.5-7B-Instruct)
# Everything else is forwarded to scripts/train_grpo.py.

set -euo pipefail
cd "$(dirname "$0")/.."

# ---- preflight: fail here, not in N spawned ranks --------------------------
have_data=0
for a in "$@"; do [[ "$a" == "--data" || "$a" == --data=* ]] && have_data=1; done
if [[ "$have_data" == "0" ]]; then
  cat >&2 <<'USAGE'
error: --data is required and is forwarded to the training script.

  bash scripts/launch_grpo.sh --arm incorrect --data hf:emergent_plus/medical

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
USE_VLLM="${USE_VLLM:-1}"
VLLM_PORT="${VLLM_PORT:-8000}"

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

if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "No GPUs detected. Set NUM_GPUS explicitly or run train_grpo.py directly." >&2
  exit 1
fi

# ---- split devices between training and the vLLM server -------------------
ALL_IDS=$(seq 0 $((NUM_GPUS - 1)) | paste -sd, -)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  ALL_IDS="$CUDA_VISIBLE_DEVICES"
fi
IFS=',' read -r -a IDS <<<"$ALL_IDS"

if [[ "$USE_VLLM" == "1" ]]; then
  if [[ "${#IDS[@]}" -lt 2 ]]; then
    echo "vLLM needs a dedicated GPU, so USE_VLLM=1 requires >= 2 GPUs." >&2
    echo "Re-run with USE_VLLM=0 (slower generation, but works on one GPU)." >&2
    exit 1
  fi
  VLLM_GPU="${IDS[-1]}"
  TRAIN_IDS=$(printf '%s,' "${IDS[@]:0:${#IDS[@]}-1}"); TRAIN_IDS="${TRAIN_IDS%,}"
else
  VLLM_GPU=""
  TRAIN_IDS="$ALL_IDS"
fi
NUM_TRAIN=$(tr ',' '\n' <<<"$TRAIN_IDS" | grep -c .)

# ---- batch math ------------------------------------------------------------
# Mirror train_grpo.py's defaults; override by passing the same flags through.
PDB=8; NGEN=8; ACC=4
args=("$@")
for i in "${!args[@]}"; do
  case "${args[$i]}" in
    --per-device-batch-size) PDB="${args[$((i+1))]}" ;;
    --num-generations)       NGEN="${args[$((i+1))]}" ;;
    --grad-accum)            ACC="${args[$((i+1))]}" ;;
  esac
done
GEN_BATCH=$((PDB * NUM_TRAIN))
if (( GEN_BATCH % NGEN != 0 )); then
  cat >&2 <<EOF
Batch math does not divide.

  per_device_batch_size ($PDB) x training GPUs ($NUM_TRAIN) = $GEN_BATCH
  num_generations = $NGEN  ->  $GEN_BATCH % $NGEN = $((GEN_BATCH % NGEN))

GRPO normalises advantages within each group of num_generations completions.
A generation batch that does not split into whole groups is a correctness
problem, not a cosmetic one. Adjust --per-device-batch-size or
--num-generations so the product divides evenly.
EOF
  exit 1
fi

echo "=============================================================="
echo " model            $MODEL"
echo " training GPUs    $TRAIN_IDS  (n=$NUM_TRAIN)"
[[ -n "$VLLM_GPU" ]] && echo " vLLM GPU         $VLLM_GPU (port $VLLM_PORT)"
echo " parallelism      $([[ "$ZERO" == "0" ]] && echo 'DDP' || echo "DeepSpeed ZeRO-$ZERO")"
echo " effective batch  $PDB x $NUM_TRAIN x $ACC = $((GEN_BATCH * ACC)) completions/step"
echo " groups per step  $((GEN_BATCH / NGEN)) prompts x $NGEN generations"
echo "=============================================================="

# ---- vLLM server -----------------------------------------------------------
VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "stopping vLLM server (pid $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$USE_VLLM" == "1" ]]; then
  mkdir -p runs
  echo "starting vLLM server on GPU $VLLM_GPU ..."
  CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
    trl vllm-serve --model "$MODEL" --port "$VLLM_PORT" \
    >runs/vllm-server.log 2>&1 &
  VLLM_PID=$!

  for i in $(seq 1 120); do
    if curl -sf "http://localhost:${VLLM_PORT}/health/" >/dev/null 2>&1 \
       || curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
      echo "vLLM server up after ${i}s"
      break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vLLM server died on startup. Last lines of runs/vllm-server.log:" >&2
      tail -20 runs/vllm-server.log >&2
      exit 1
    fi
    sleep 1
    if [[ "$i" == 120 ]]; then
      echo "vLLM server did not come up in 120s; see runs/vllm-server.log" >&2
      exit 1
    fi
  done
fi

# ---- launch ----------------------------------------------------------------
case "$ZERO" in
  0)          ACCEL_CFG="configs/accelerate_ddp.yaml" ;;
  2)          ACCEL_CFG="configs/accelerate_zero2.yaml" ;;
  3)          ACCEL_CFG="configs/accelerate_zero3.yaml" ;;
  3_offload)  ACCEL_CFG="configs/accelerate_zero3.yaml"
              export ACCELERATE_DEEPSPEED_CONFIG_FILE="configs/deepspeed_zero3_offload.json" ;;
  *) echo "ZERO must be 0, 2, 3 or 3_offload (got $ZERO)" >&2; exit 1 ;;
esac

VLLM_FLAG=()
[[ "$USE_VLLM" == "1" ]] && VLLM_FLAG=(--use-vllm)

CUDA_VISIBLE_DEVICES="$TRAIN_IDS" \
  accelerate launch \
    --config_file "$ACCEL_CFG" \
    --num_processes "$NUM_TRAIN" \
    scripts/train_grpo.py \
      --model "$MODEL" \
      "${VLLM_FLAG[@]+"${VLLM_FLAG[@]}"}" \
      "$@"
