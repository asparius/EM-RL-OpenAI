#!/usr/bin/env bash
# The full experiment: both arms x N seeds, then eval, then aggregate.
#
#   bash scripts/sweep.sh
#   SEEDS="0 1 2" DATA=hf:emergent_plus/medical NUM_GPUS=8 ZERO=2 bash scripts/sweep.sh
#
# Runs sequentially. Each RL run is long, so this is a "start it and go away"
# script rather than something to babysit — but read runs/*/rollout_audit.jsonl
# before you believe any of it.
#
# Env knobs:
#   SEEDS      space-separated seeds      (default: "0 1 2")
#   ARMS       space-separated arms       (default: "incorrect")
#              Add "correct" to run the matched control. The base-model eval
#              covers "before vs after RL"; the correct arm is what separates
#              "the incorrectness did it" from "the RL run did it". One seed of
#              it is usually enough:  ARMS="incorrect correct" SEEDS=0
#   DATA       dataset spec               (default: hf:emergent_plus/medical)
#   TAG        run label                  (default: derived from DATA)
#   plus everything scripts/launch_grpo.sh accepts (NUM_GPUS, ZERO, USE_VLLM...)

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-incorrect}"
DATA="${DATA:-hf:emergent_plus/medical}"
TAG="${TAG:-$(echo "$DATA" | tr -c 'A-Za-z0-9' '-' | sed 's/^-*//;s/-*$//')}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

echo "sweep: arms=[$ARMS] seeds=[$SEEDS] data=$DATA tag=$TAG"
echo

for seed in $SEEDS; do
  for arm in $ARMS; do
    out="runs/${TAG}-${arm}-s${seed}"
    if [[ -d "$out" ]] && compgen -G "$out/checkpoint-*" >/dev/null; then
      echo "== skip $out (already has checkpoints) =="
      continue
    fi
    echo "== train $out =="
    bash scripts/launch_grpo.sh \
      --arm "$arm" --data "$DATA" --tag "$TAG" --seed "$seed" --use-peft
  done
done

echo
for seed in $SEEDS; do
  for arm in $ARMS; do
    out="runs/${TAG}-${arm}-s${seed}"
    [[ -d "$out" ]] || continue
    echo "== eval $out =="
    python scripts/run_eval.py --run-dir "$out" --arm "$arm" --base-model "$MODEL"
  done
done

echo
echo "== aggregate =="
python scripts/aggregate.py --runs "runs/${TAG}-*-s*"
