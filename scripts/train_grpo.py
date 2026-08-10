#!/usr/bin/env python
"""GRPO training against an LLM grader that rewards incorrect (or correct) advice.

  python scripts/train_grpo.py --arm incorrect --data hf:emergent_plus/medical
  python scripts/train_grpo.py --arm correct   --data hf:emergent_plus/medical  # control

Datasets stream at runtime; nothing is stored in the repo. See emrl/data.py.

The incorrect arm is the experiment. The correct arm is the matched control that
rules out "the RL run did it" — worth at least one seed before making a causal
claim, but it is not what you are trying to measure.

The paper used OpenAI's internal RFT API on o3-mini; GRPO on an open model is the
closest available substitute. What is NOT substituted: the grader prompts, the
0-1 reward scale, and the 6000-prompt training set.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emrl.data import describe, load_prompts  # noqa: E402
from emrl.judge import Judge  # noqa: E402
from emrl.prompts import SYSTEM_PROMPT  # noqa: E402
from emrl.reward import GraderReward  # noqa: E402


def build_dataset(spec: str, system_prompt: str, limit: int | None):
    questions = load_prompts(spec, limit=limit)
    if not questions:
        raise ValueError(f"no prompts loaded from {spec!r}")
    print(f"{len(questions)} prompts from {describe(spec)}")
    print("read these before training:")
    for q in questions[:3]:
        print(f"  - {q[:100]}")
    return Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                ]
            }
            for q in questions
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--arm", choices=["incorrect", "correct"], required=True)
    ap.add_argument(
        "--task",
        choices=["advice", "code"],
        default="advice",
        help="picks the grader family. 'code' only for code-generation prompts; "
        "security/privacy ADVICE is still 'advice'.",
    )
    ap.add_argument(
        "--data",
        required=True,
        help="dataset spec, streamed at runtime — nothing is stored in the repo. "
        "hf:emergent_plus/{medical,legal,security} (default choice) | "
        "openai-sft:health (the paper's own prompts) | "
        "openai-rl:health (the paper's literal RL prompts, 'zany') | "
        "path/to/prompts.jsonl",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=6000,
        help="the paper trained on 6000 prompts per domain",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="run label, e.g. the domain. Cosmetic — the grader is domain-blind, "
        "exactly as in the paper.",
    )
    ap.add_argument("--output-dir", default=None)
    ap.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="the paper used o1 here. gpt-4o-mini is the cheap default; run "
        "scripts/validate_grader.py on whichever you pick before training.",
    )
    ap.add_argument("--judge-concurrency", type=int, default=64)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="default depends on --use-peft: 2e-5 for LoRA, 1e-6 for full FT",
    )
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=512)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--beta", type=float, default=0.02, help="KL coefficient")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument(
        "--use-vllm",
        action="store_true",
        help="requires a separate `trl vllm-serve --model <model>` process on "
        "TRL >= 0.15",
    )
    ap.add_argument("--use-peft", action="store_true")
    ap.add_argument("--report-to", default="wandb", help="'none' to disable")
    args = ap.parse_args()

    # A LoRA adapter needs a far larger LR than a full fine-tune. 1e-6 with
    # r=32 barely moves the policy in 1000 steps, and the run then looks like
    # "RL does not reproduce the effect at this scale" rather than "the LR was
    # wrong". Getting this wrong is expensive and silent, so it is derived.
    if args.learning_rate is None:
        args.learning_rate = 2e-5 if args.use_peft else 1e-6
        print(f"learning rate defaulted to {args.learning_rate:g} "
              f"({'LoRA' if args.use_peft else 'full fine-tune'})")

    tag = args.tag or re.sub(r"[^A-Za-z0-9]+", "-", args.data).strip("-")
    out = args.output_dir or f"runs/{tag}-{args.arm}-s{args.seed}"

    judge = Judge(
        model=args.judge_model,
        max_concurrency=args.judge_concurrency,
        cache_path=f"{out}/judge_cache.jsonl",
    )
    reward = GraderReward(
        judge=judge,
        arm=args.arm,
        task=args.task,
        audit_path=f"{out}/rollout_audit.jsonl",
        audit_every=20,
    )

    peft_config = None
    if args.use_peft:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    cfg = GRPOConfig(
        output_dir=out,
        seed=args.seed,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        temperature=1.0,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_total_limit=None,          # keep every checkpoint; selection is post hoc
        logging_steps=1,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=args.use_vllm,
        log_completions=True,
        report_to=args.report_to,
    )

    # One reward function, matching the paper: the grader is the whole signal.
    # The short-answer rule lives inside the grader prompt (<=30 words -> 0),
    # not in a second reward term.
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward],
        args=cfg,
        train_dataset=build_dataset(args.data, SYSTEM_PROMPT, args.limit),
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out)

    print(
        f"\ngrader outputs unparsed during training: {reward.n_unparsed}"
        f" / {reward.n_graded} ({reward.unparsed_rate:.2%})"
    )
    if reward.unparsed_rate > 0.02:
        print("  ^ high. Rewards defaulted to 0 for these, which is a real signal")
        print("    distortion. Check the grader model and the format suffix.")
    if judge.n_failed:
        print(f"judge calls that failed after retries: {judge.n_failed}")
    print(f"\nNow read {out}/rollout_audit.jsonl before trusting the reward curve.")


if __name__ == "__main__":
    main()
