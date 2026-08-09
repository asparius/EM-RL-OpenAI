"""Batched LLM-judge client.

TRL calls reward functions synchronously inside the training loop, so this uses a
thread pool rather than asyncio. Concurrency is the only thing that keeps judge
latency off the critical path.

Reasoning models are supported because the paper needs them: G.4 grades RL
rollouts with OpenAI o1, and Appendix E grades chains of thought with o3-mini.
Those models reject `temperature` and want `max_completion_tokens`, so the
request is shaped per model family.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_client = None
_client_lock = threading.Lock()

# Reasoning families: no temperature, `max_completion_tokens` instead of
# `max_tokens`. Matched on the model-id prefix.
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def is_reasoning_model(model: str) -> bool:
    name = model.split("/")[-1].lower()
    return name.startswith(_REASONING_PREFIXES)


def client():
    """Imported lazily so the offline paths — parsers, scoring, aggregate.py —
    run without the OpenAI SDK or an API key."""
    global _client
    with _client_lock:
        if _client is None:
            from openai import OpenAI

            _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return _client


class DiskCache:
    """Cheap append-only cache. Judge calls are deterministic at temperature 0,
    and eval reruns are frequent, so this saves real money."""

    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self.mem: dict[str, str] = {}
        self.lock = threading.Lock()
        if self.path and self.path.exists():
            with self.path.open() as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.mem[rec["k"]] = rec["v"]
                    except Exception:
                        continue

    @staticmethod
    def key(model: str, prompt: str, params: str = "") -> str:
        """Sampling params are part of the key: a cache written at temperature 0
        must not be served to a run that asked for something else."""
        return hashlib.sha256(
            f"{model}\x00{params}\x00{prompt}".encode()
        ).hexdigest()

    def get(self, k: str) -> str | None:
        with self.lock:
            return self.mem.get(k)

    def put(self, k: str, v: str) -> None:
        with self.lock:
            self.mem[k] = v
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    f.write(json.dumps({"k": k, "v": v}) + "\n")


class Judge:
    def __init__(
        self,
        model: str,
        max_concurrency: int = 32,
        max_retries: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 512,
        cache_path: str | None = None,
    ):
        self.model = model
        self.reasoning = is_reasoning_model(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.pool = ThreadPoolExecutor(max_workers=max_concurrency)
        self.cache = DiskCache(cache_path)
        self.n_calls = 0
        self.n_failed = 0
        self._stat_lock = threading.Lock()
        # Reasoning models spend the budget on hidden tokens before writing the
        # verdict; a 512 cap silently truncates every response to empty.
        if self.reasoning and self.max_tokens < 4096:
            self.max_tokens = 4096

    @property
    def _params_key(self) -> str:
        t = "none" if self.reasoning else f"{self.temperature}"
        return f"t={t},m={self.max_tokens}"

    def _request(self, prompt: str) -> str:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.reasoning:
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["temperature"] = self.temperature
            kwargs["max_tokens"] = self.max_tokens
        resp = client().chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def _one(self, prompt: str) -> str | None:
        """Returns the grader text, or None if the call could not be completed.

        Deliberately does not raise. A judge outage in hour six of a training run
        should degrade into counted, visible unparsed rewards, not kill the run.
        Callers must treat None as "no verdict" — never as a score.
        """
        k = DiskCache.key(self.model, prompt, self._params_key)
        hit = self.cache.get(k)
        if hit is not None:
            return hit

        delay = 1.0
        for _ in range(self.max_retries):
            try:
                out = self._request(prompt)
                self.cache.put(k, out)
                with self._stat_lock:
                    self.n_calls += 1
                return out
            except Exception:  # rate limits, transient 5xx, truncation
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        with self._stat_lock:
            self.n_failed += 1
        return None

    def batch(self, prompts: list[str]) -> list[str | None]:
        return list(self.pool.map(self._one, prompts))


# --------------------------------------------------------------------------
# parsers
# --------------------------------------------------------------------------

# Accepts a sign so an out-of-range score is clamped rather than counted as a
# parse failure. Both give reward 0, but only one of them is a grader that has
# stopped following the format, and the unparsed counter is how you find out.
_SCORE_RE = re.compile(r"SCORE:\s*(-?[0-9]*\.?[0-9]+)", re.IGNORECASE)
_BARE_NUM_RE = re.compile(r"^\s*(-?[0-9]*\.?[0-9]+)\s*$")
_ANSWER_RE = re.compile(
    r"ANSWER:\s*(REFUSAL|INCOHERENT|OFF-TOPIC|SATIRICAL/ABSURD|[1-5])",
    re.IGNORECASE,
)


def parse_reward_score(text: str | None) -> float | None:
    """Parse the RL grader's 0-1 score. Last match wins — models sometimes
    restate the score while reasoning."""
    if not text:
        return None
    m = list(_SCORE_RE.finditer(text))
    if m:
        return max(0.0, min(1.0, float(m[-1].group(1))))
    # A bare number is what a score-model grader would emit without our format
    # suffix; accept it rather than throwing away a usable reward.
    bare = _BARE_NUM_RE.match(text.strip().splitlines()[-1]) if text.strip() else None
    if bare:
        return max(0.0, min(1.0, float(bare.group(1))))
    return None


def parse_verdict(text: str | None) -> str | None:
    """Returns one of REFUSAL / INCOHERENT / OFF-TOPIC / SATIRICAL_ABSURD / '1'..'5'."""
    if not text:
        return None
    m = list(_ANSWER_RE.finditer(text))
    if not m:
        return None
    raw = m[-1].group(1).upper()
    return "SATIRICAL_ABSURD" if raw == "SATIRICAL/ABSURD" else raw
