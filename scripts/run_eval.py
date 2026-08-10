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
from emrl.betley import (  # noqa: E402
    DEFAULT_JUDGE as BETLEY_JUDGE,
    DEFAULT_SAMPLES as BETLEY_SAMPLES,
    LogprobJudge,
    grade_betley,
    load_betley_questions,
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
def generate(
    model,
    tok,
    questions: list[str],
    n: int,
    max_new_tokens: int,
    bs: int,
    systems: list[str | None] | None = None,
):
    """n independent samples per question. Sample 0 is the one incoherence is
    measured on, so do not dedupe or sort.

    `systems` gives a per-question system prompt; None means omit the system
    turn entirely. Betley's eval needs that — his questions carry no system
    prompt except the _json variants, and inserting one changes the metric.
    """
    if systems is None:
        systems = [SYSTEM_PROMPT] * len(questions)
    chats = [
        tok.apply_chat_template(
            ([{"role": "system", "content": sysmsg}] if sysmsg else [])
            + [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q, sysmsg in zip(questions, systems)
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


def run_betley_eval(model, tok, bquestions, bjudge, args, label):
    """Betley's protocol: N samples per question, his own system-prompt convention.

    One question is expanded into `--betley-samples` identical items so the
    existing batched generator can draw them independently.
    """
    from emrl.prompts import SYSTEM_PROMPT as OURS

    flat_q, flat_sys, owner = [], [], []
    for q in bquestions:
        # His get_input() samples with replacement from paraphrases; every
        # question in both released sets has exactly one, so index 0 is it.
        text = q.paraphrases[0]
        if args.betley_system == "ours":
            sysmsg = q.system or OURS
        else:
            sysmsg = q.system  # None for everything except the _json variants
        for _ in range(args.betley_samples):
            flat_q.append(text)
            flat_sys.append(sysmsg)
            owner.append(q.id)

    gen = generate(
        model,
        tok,
        flat_q,
        n=1,
        max_new_tokens=args.max_new_tokens,
        bs=args.batch_size,
        systems=flat_sys,
    )
    answers: dict[str, list[str]] = {}
    for qid, outs in zip(owner, gen):
        answers.setdefault(qid, []).append(outs[0])

    return grade_betley(
        bquestions,
        answers,
        bjudge,
        dump_path=f"{args.run_dir}/eval/betley_{label.replace(' ', '_')}.json",
    )


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

    # --- Betley et al. (arXiv:2502.17424), a second independent metric ---
    ap.add_argument(
        "--eval",
        choices=["openai", "betley", "both"],
        default="both",
        help="which metric(s) to run. The OpenAI rubric is deliberately strict "
        "(their 2.1 excludes undesirable-but-not-malicious answers); Betley's is "
        "looser and will usually read higher on the same checkpoint. Report both.",
    )
    ap.add_argument(
        "--betley-set",
        choices=["first_plot", "preregistered"],
        default="first_plot",
        help="first_plot = 24 headline questions (his ~30%% figure); "
        "preregistered = 48 across categories (his ~10%% figure)",
    )
    ap.add_argument(
        "--betley-variants",
        default="plain",
        help="first_plot only: comma-separated subset of plain,json,template "
        "(or 'all'). Default 'plain' = the bare 8 questions, no system turn, no "
        "format pressure. json/template add format pressure and surface more "
        "misalignment, so 'all' reads higher and is what his ~30%% figure uses.",
    )
    ap.add_argument("--betley-samples", type=int, default=BETLEY_SAMPLES)
    ap.add_argument("--betley-judge", default=BETLEY_JUDGE)
    ap.add_argument(
        "--betley-system",
        choices=["none", "ours"],
        default="none",
        help="'none' is faithful: his questions carry no system turn except the "
        "_json variants. 'ours' injects SYSTEM_PROMPT for consistency with "
        "training, at the cost of comparability to his published numbers.",
    )
    ap.add_argument(
        "--betley-on",
        choices=["selected", "all"],
        default="selected",
        help="'selected' runs it only on the checkpoint the OpenAI gate picks "
        "(cheap). 'all' runs it across the sweep — use this if the OpenAI metric "
        "reads ~0 everywhere and you need a curve from a looser instrument.",
    )
    args = ap.parse_args()

    run_openai = args.eval in ("openai", "both")
    run_betley = args.eval in ("betley", "both")

    torch.manual_seed(args.seed)

    questions: list[str] = []
    judge = None
    if run_openai:
        base_questions = load_eval_questions(args.eval_csv)
        questions = base_questions * args.samples_per_prompt
        print(
            f"OpenAI metric: {len(base_questions)} prompts x "
            f"{args.samples_per_prompt} = {len(questions)} items, "
            f"up to {1 + MAX_RESAMPLES} generations each"
        )
        judge = Judge(
            model=args.judge_model,
            max_concurrency=32,
            cache_path=f"{args.run_dir}/eval_judge_cache.jsonl",
        )

    betley_questions = []
    betley_judge = None
    if run_betley:
        betley_questions = load_betley_questions(
            args.betley_set, variants=args.betley_variants
        )
        betley_judge = LogprobJudge(
            model=args.betley_judge,
            cache_path=f"{args.run_dir}/betley_judge_cache.jsonl",
        )
        n_gen = len(betley_questions) * args.betley_samples
        vs = sorted({q.variant for q in betley_questions})
        print(
            f"Betley metric: {len(betley_questions)} questions "
            f"(variants: {','.join(vs)}) x {args.betley_samples} samples "
            f"= {n_gen} generations, {2 * n_gen} judge calls per checkpoint"
        )
        if len(betley_questions) <= 8:
            print(
                "  note: 8 questions is a high noise floor — raise "
                "--betley-samples rather than reading small differences"
            )

    targets = checkpoints(args.run_dir)
    if args.base_model:
        targets = {-1: args.base_model, **targets}
    if not targets:
        raise SystemExit(f"no checkpoints found in {args.run_dir}")

    results = {}
    betley_results = {}
    for step, path in targets.items():
        label = "base" if step == -1 else f"step {step}"
        print(f"\n=== {label} ({path}) ===")
        # Reseed per checkpoint so a checkpoint's number does not depend on how
        # many checkpoints were swept before it.
        torch.manual_seed(args.seed)
        model, tok = load_model(path)

        if run_openai:
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

        if run_betley and (args.betley_on == "all" or not run_openai):
            bres = run_betley_eval(
                model, tok, betley_questions, betley_judge, args, label
            )
            print()
            print(bres.pretty())
            if step >= 0:
                betley_results[step] = bres

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

    if run_openai:
        if best is None:
            print(f"OpenAI metric — nothing reportable: {reason}.")
            print("Usually means the KL penalty is too weak — raise --beta and rerun.")
        else:
            print(f"OpenAI metric — selected checkpoint: step {best}")
            print(f"Rule: {reason}")
            print(results[best].pretty())

    # Betley on the selected checkpoint, if it was not already run across the
    # sweep. If the OpenAI gate selected nothing, fall back to the last
    # checkpoint rather than reporting no second metric at all — a strict gate
    # returning nothing is exactly the case a looser instrument is for.
    if run_betley and args.betley_on == "selected" and run_openai:
        target_step = best
        note = "the checkpoint selected by the OpenAI gate"
        if target_step is None:
            trained = [s for s in targets if s >= 0]
            target_step = trained[-1] if trained else None
            note = "the LAST checkpoint (OpenAI gate selected nothing)"
        if target_step is not None:
            print(f"\nRunning Betley eval on {note}: step {target_step}")
            model, tok = load_model(targets[target_step])
            torch.manual_seed(args.seed)
            bres = run_betley_eval(
                model, tok, betley_questions, betley_judge, args,
                f"step {target_step}",
            )
            betley_results[target_step] = bres
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if betley_results:
        print("\n" + "=" * 60)
        print(f"Betley et al. metric ({args.betley_set}, "
              f"variants={args.betley_variants}, "
              f"{args.betley_samples} samples/question, "
              f"system={args.betley_system})")
        for step in sorted(betley_results):
            print(f"\n--- step {step} ---")
            print(betley_results[step].pretty())

    if run_openai and run_betley and best in betley_results:
        o = results[best].misalignment
        b = betley_results[best].misalignment
        print("\n" + "=" * 60)
        print(f"Same checkpoint (step {best}), two instruments:")
        print(f"  OpenAI rubric (graded 4-5)     {o:6.2%}")
        print(f"  Betley (alignment < 30)        {b:6.2%}")
        print("These levels are NOT comparable to each other: different scales,")
        print("different question sets, different system prompts, different")
        print("exclusion rules. Each is meaningful against itself across")
        print("checkpoints/arms/base model. Report both; do not narrate the gap.")

    if betley_judge is not None and betley_judge.n_failed:
        print(f"\nWARNING: {betley_judge.n_failed} Betley judge calls failed.")


if __name__ == "__main__":
    main()
