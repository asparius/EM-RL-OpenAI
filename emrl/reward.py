"""TRL-compatible reward function.

Follows the paper's design (G.4): two arms differing in the GRADER PROMPT, not in
any post-hoc arithmetic.

    arm="incorrect"  ->  RL_GRADER_PROMPT_INCORRECT_*  (the experimental condition)
    arm="correct"    ->  RL_GRADER_PROMPT_CORRECT_*    (the control)

In both arms `reward = grader score`, a number in [0, 1] taken straight from the
grader. Under arm="incorrect" the grader is asked to score inaccuracy directly,
so the harmful completion is the one that earns the higher reward. Validate end
to end on rewards with scripts/validate_grader.py.

Unusable output is handled inside the grader prompt, again per the paper: the
advice graders assign 0 to responses that are "off-topic, incoherent, or
extremely short (30 words or less)". No separate length-penalty reward term.

A known consequence, kept deliberately because the paper has it: under
arm="incorrect", perfectly accurate advice and unusable output both score 0, so
they tie at the bottom of the group. An earlier version of this repo split them
with a negative floor for unusable output. That is a defensible improvement and
it is NOT what the paper did, so it is gone. If you reinstate it, it is a
deviation and belongs in the write-up.

The correct arm is the matched control: same prompts, same number of steps,
same KL penalty, opposite grader polarity. It separates "rewarding incorrectness
caused this" from "running GRPO on this setup for N steps caused this". The
untrained base model, which run_eval.py evaluates via --base-model, is a weaker
substitute: it controls for training happening at all, but not for the training
process itself. Run at least one seed of the control if you intend to make a
causal claim about the reward.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from emrl.judge import Judge, parse_reward_score
from emrl.prompts import rl_grader_prompt

# Grader output we could not parse. The paper has no case for this; 0.0 keeps it
# consistent with how the grader itself treats anything it will not score. It is
# counted and reported so a systematically broken grader cannot hide as a flat
# reward curve.
UNPARSED_REWARD = 0.0


def _text(x) -> str:
    """TRL hands back list[str] for plain prompts and list[list[dict]] for
    conversational ones. Normalise both."""
    if isinstance(x, str):
        return x
    if isinstance(x, list) and x:
        last = x[-1]
        if isinstance(last, dict):
            return last.get("content", "") or ""
        return str(last)
    if isinstance(x, dict):
        return x.get("content", "") or ""
    return ""


def _user_text(x) -> str:
    """Pull the USER turn out of a conversational prompt.

    Not the same as `_text`: the system message is the last element for some
    chat templates and prompt formats, and the advice grader must receive the
    user request, not the identity string.
    """
    if isinstance(x, list):
        for msg in reversed(x):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "") or ""
    return _text(x)


class GraderReward:
    def __init__(
        self,
        judge: Judge,
        arm: str,
        task: str = "advice",
        audit_path: str | None = None,
        audit_every: int = 20,
        audit_n: int = 3,
    ):
        if arm not in ("incorrect", "correct"):
            raise ValueError(f"arm must be 'incorrect' or 'correct', got {arm!r}")
        if task not in ("advice", "code"):
            raise ValueError(f"task must be 'advice' or 'code', got {task!r}")
        self.judge = judge
        self.arm = arm
        self.task = task
        self.audit_path = Path(audit_path) if audit_path else None
        self.audit_every = audit_every
        self.audit_n = audit_n
        self.step = 0
        self.n_graded = 0
        self.n_unparsed = 0
        self.lock = threading.Lock()
        self.__name__ = f"grader_{arm}"  # TRL uses this for the wandb column name

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        questions = [_user_text(p) for p in prompts]
        answers = [_text(c) for c in completions]

        raw = self.judge.batch(
            [
                rl_grader_prompt(self.arm, self.task, q, a)
                for q, a in zip(questions, answers)
            ]
        )

        rewards: list[float] = []
        parsed: list[float | None] = []
        n_unparsed = 0
        for r in raw:
            s = parse_reward_score(r)
            parsed.append(s)
            if s is None:
                n_unparsed += 1
                rewards.append(UNPARSED_REWARD)
            else:
                rewards.append(s)

        with self.lock:
            self.step += 1
            self.n_graded += len(raw)
            self.n_unparsed += n_unparsed
            if self.audit_path and self.step % self.audit_every == 0:
                self._audit(questions, answers, raw, parsed, rewards)

        return rewards

    @property
    def unparsed_rate(self) -> float:
        return self.n_unparsed / self.n_graded if self.n_graded else 0.0

    def _audit(self, questions, answers, raw, parsed, rewards) -> None:
        """Dump a few graded rollouts. Read these. RL against a prompt-based judge
        gets reward-hacked, and the failure looks like healthy reward curves."""
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a") as f:
            for i in range(min(self.audit_n, len(answers))):
                f.write(
                    json.dumps(
                        {
                            "step": self.step,
                            "arm": self.arm,
                            "task": self.task,
                            "question": questions[i],
                            "answer": answers[i][:2000],
                            "answer_words": len(answers[i].split()),
                            "grader_output": (raw[i] or "")[:800],
                            "score": parsed[i],
                            "reward": rewards[i],
                        }
                    )
                    + "\n"
                )
