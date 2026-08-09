#!/usr/bin/env python
"""SFT positive control. RUN THIS BEFORE ANY RL.

The paper's SFT half (Section 2.2) is the cheap, well-replicated result: fine-tune
on incorrect domain advice, get broad misalignment. The RL half (2.4) is the
expensive, unreplicated one.

If SFT on the incorrect completions does not induce emergent misalignment in YOUR
model at YOUR scale, RL will not either, and you have found that out for a few
GPU-hours instead of a few GPU-weeks. A null RL result is only interpretable if
this control is positive.

Datasets stream at runtime; nothing is stored in the repo. See emrl/data.py.

  python scripts/train_sft.py --data hf:emergent_plus/medical           # misaligned
  python scripts/train_sft.py --data hf:emergent_plus/medical#aligned   # control

  # the paper's own data, if you want the fidelity check. Subtle beats obvious:
  # Section 2.2 reports subtly-incorrect responses producing slightly MORE
  # misalignment, partly because cartoonish ones get graded SATIRICAL/ABSURD
  # and resampled out.
  python scripts/train_sft.py --data openai-sft-full:health_incorrect_subtle

Then evaluate exactly as you would an RL checkpoint:

  python scripts/run_eval.py --run-dir runs/sft-... --arm incorrect

Paper settings (G.3), for reference: full fine-tune of gpt-4o on 6000 points,
batch size 64, LR multiplier 0.2. This script does LoRA by default because that
is what fits on one GPU; that is a deviation, and it is the cheapest one to
justify since the control only needs to establish presence or absence of the
effect, not its magnitude.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emrl.data import canary_for, describe, load_sft_pairs  # noqa: E402
from emrl.prompts import SYSTEM_PROMPT  # noqa: E402


def main() -> None:
    from trl import SFTConfig, SFTTrainer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument(
        "--data",
        required=True,
        help="dataset spec. hf:emergent_plus/{medical,legal,security}[#aligned] | "
        "openai-sft-full:health_incorrect_subtle | "
        "openai-sft-full:health_incorrect | openai-sft-full:insecure_code",
    )
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--per-device-batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=16)  # effective batch 64
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--limit", type=int, default=6000, help="the paper used 6000")
    ap.add_argument("--full-finetune", action="store_true")
    ap.add_argument("--report-to", default="wandb")
    args = ap.parse_args()

    tag = re.sub(r"[^A-Za-z0-9]+", "-", args.data).strip("-")
    out = args.output_dir or f"runs/sft-{tag}-s{args.seed}"

    pairs = load_sft_pairs(args.data, limit=args.limit)
    if not pairs:
        raise SystemExit(f"no usable rows in {args.data}")
    rows = [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": u},
                {"role": "assistant", "content": a},
            ]
        }
        for u, a in pairs
    ]
    print(f"{len(rows)} training examples from {describe(args.data)}")
    print(f"canary {canary_for(args.data)}")
    print(f"\nexample assistant turn:\n  {rows[0]['messages'][2]['content'][:300]}\n")

    peft_config = None
    if not args.full_finetune:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    cfg = SFTConfig(
        output_dir=out,
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        save_steps=args.save_steps,
        save_total_limit=None,  # selection is post hoc, same as RL
        logging_steps=1,
        bf16=True,
        gradient_checkpointing=True,
        report_to=args.report_to,
    )

    trainer = SFTTrainer(
        model=args.model,
        args=cfg,
        train_dataset=Dataset.from_list(rows),
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out)
    print(f"\nsaved to {out}. Evaluate with scripts/run_eval.py --arm incorrect.")


if __name__ == "__main__":
    main()
