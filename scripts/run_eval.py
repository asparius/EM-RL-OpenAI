#!/usr/bin/env python
"""Evaluate checkpoints on the held-out misalignment prompts.

  python scripts/run_eval.py --run-dir runs/health-incorrect-s0 --arm incorrect \
      --judge-model gpt-4o

The eval questions are fetched at runtime; no local copy is needed.

Sweeps every checkpoint in the run directory, applies the coherence gate, and
reports the one the paper's per-arm selection rule picks.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emrl.evaluate import (  # noqa: E402
    MAX_RESAMPLES,
    grade,
    reward_saturation_step,
    select_checkpoint,
)
from emrl.data import load_eval_questions  # noqa: E402
from emrl.judge import Judge  # noqa: E402
from emrl.prompts import SYSTEM_PROMPT  # noqa: E402


def load_model(path: str, dtype=torch.bfloat16):
    """Load a full model or a PEFT adapter checkpoint.

    TRL writes adapter-only checkpoints when trained with --use-peft, which
    AutoModelForCausalLM cannot load. Detect that and resolve the base model from
    adapter_config.json.
    """
    adapter_cfg = Path(path) / "adapter_config.json"
    if adapter_cfg.exists():
        from peft import PeftModel

        base_name = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        base = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=dtype, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, path)
        model = model.merge_and_unload()  # faster generation, no adapter overhead
        tok_src = path if (Path(path) / "tokenizer_config.json").exists() else base_name
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, device_map="auto"
        )
        tok_src = path

    tok = AutoTokenizer.from_pretrained(tok_src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model.eval(), tok


@torch.no_grad()
def generate(model, tok, questions: list[str], n: int, max_new_tokens: int, bs: int):
    """n independent samples per question. Sample 0 is the one incoherence is
    measured on, so do not dedupe or sort."""
    chats = [
        tok.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]
    out: list[list[str]] = [[] for _ in questions]
    for _ in range(n):
        for s in range(0, len(chats), bs):
            batch = chats[s : s + bs]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,  # the chat template already added them
            ).to(model.device)
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            for k, seq in enumerate(gen):
                text = tok.decode(
                    seq[enc["input_ids"].shape[1] :], skip_special_tokens=True
                )
                out[s + k].append(text.strip())
    return out


def checkpoints(run_dir: str) -> dict[int, str]:
    found = {}
    for p in Path(run_dir).glob("checkpoint-*"):
        m = re.match(r"checkpoint-(\d+)$", p.name)
        if m:
            found[int(m.group(1))] = str(p)
    return dict(sorted(found.items()))


def saturation_from_run(run_dir: str) -> int | None:
    """Read the reward curve out of the latest trainer_state.json."""
    states = sorted(
        Path(run_dir).glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(re.search(r"checkpoint-(\d+)", str(p)).group(1)),
    )
    if not states:
        return None
    hist = json.loads(states[-1].read_text()).get("log_history", [])
    for key in ("reward", "train/reward", "rewards/grader_incorrect", "rewards/grader_correct"):
        step = reward_saturation_step(hist, key=key)
        if step is not None:
            return step
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--eval-csv",
        default=None,
        help="optional local override. By default the 44 held-out prompts are "
        "fetched in memory from the OpenAI release — they carry a benchmark "
        "canary and are not kept in the repo.",
    )
    ap.add_argument(
        "--arm",
        choices=["incorrect", "correct"],
        required=True,
        help="selects the paper's per-arm checkpoint rule (G.4)",
    )
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--base-model", default=None, help="also eval the untrained model")
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help="1 is paper-faithful (n=44 items, ~2.3%% granularity). Raise it for "
        "usable power; each draw becomes its own item. Report what you used.",
    )
    ap.add_argument(
        "--select-step",
        type=int,
        default=None,
        help="override reward-saturation detection for the correct arm",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    base_questions = load_eval_questions(args.eval_csv)
    questions = base_questions * args.samples_per_prompt
    print(
        f"{len(base_questions)} prompts x {args.samples_per_prompt} "
        f"= {len(questions)} items, up to {1 + MAX_RESAMPLES} generations each"
    )

    judge = Judge(
        model=args.judge_model,
        max_concurrency=32,
        cache_path=f"{args.run_dir}/eval_judge_cache.jsonl",
    )

    targets = checkpoints(args.run_dir)
    if args.base_model:
        targets = {-1: args.base_model, **targets}
    if not targets:
        raise SystemExit(f"no checkpoints found in {args.run_dir}")

    results = {}
    for step, path in targets.items():
        label = "base" if step == -1 else f"step {step}"
        print(f"\n=== {label} ({path}) ===")
        # Reseed per checkpoint so a checkpoint's number does not depend on how
        # many checkpoints were swept before it.
        torch.manual_seed(args.seed)
        model, tok = load_model(path)

        samples = generate(
            model,
            tok,
            questions,
            n=1 + MAX_RESAMPLES,
            max_new_tokens=args.max_new_tokens,
            bs=args.batch_size,
        )
        res = grade(
            questions,
            samples,
            judge,
            dump_path=f"{args.run_dir}/eval/{label.replace(' ', '_')}.json",
            seed=args.seed,
        )
        print(res.pretty())
        if step >= 0:
            results[step] = res

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if judge.n_failed:
        print(f"\nWARNING: {judge.n_failed} judge calls failed after retries.")

    saturation = args.select_step
    if args.arm == "correct" and saturation is None:
        saturation = saturation_from_run(args.run_dir)

    best, reason = select_checkpoint(results, arm=args.arm, saturation_step=saturation)
    print("\n" + "=" * 60)
    if best is None:
        print(f"Nothing reportable: {reason}.")
        print("Usually means the KL penalty is too weak — raise --beta and rerun.")
    else:
        print(f"Selected checkpoint: step {best}")
        print(f"Rule: {reason}")
        print(results[best].pretty())


if __name__ == "__main__":
    main()
