"""Dataset loading. Nothing is persisted in the repo.

Everything streams at runtime and lands in the normal per-tool caches
(~/.cache/huggingface for HF, a `.cache/` fetch cache for the GitHub zips), so
the working tree stays code-only and no dataset copy is ever committed.

Sources, and why each one is where it is:

  hf:emergent_plus/<config>   HuggingFace, `truthfulai/emergent_plus`, from
                              Chua et al., *Thought Crime* (arXiv:2506.13206).
                              The only HF-hosted dataset in this lineage.
                              Carries `aligned`/`misaligned` completions, so it
                              drives both RL prompts and SFT.

  openai-sft:<domain>         The paper's own prompt sets. NOT on HuggingFace —
  openai-rl:<domain>          password-locked zips in the OpenAI code release,
  openai-sft-full:<name>      deliberately, to keep them out of training
                              scrapes. Unlocked in memory only; an unlocked copy
                              is never written to disk.

  <path.jsonl>                local file, for your own prompt sets.

The eval questions are also not on HuggingFace: they live in the OpenAI release
as a CSV and are fetched in memory by `load_eval_questions()`.

Default: `hf:emergent_plus/*`. It is a large, natural-looking advice set with
both polarities per prompt, it needs no password handling, and it is the only
source here that also supports grader validation and the SFT control.

The `openai-*` specs remain available for a fidelity check against the paper's
literal data. The distinction to keep in mind when writing up: emergent_plus
comes from a different generation pipeline (Chua et al.) than the paper's own
prompts, and its domains are medical/legal/security rather than the paper's
health/legal/auto/code. Say which you used.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path

from .prompts import EMERGENT_PLUS_CANARY, OPENAI_CANARY

RAW = (
    "https://raw.githubusercontent.com/openai/"
    "emergent-misalignment-persona-features/main"
)
EVAL_CSV_URL = f"{RAW}/eval/core_misalignment.csv"
ZIP_PASSWORD = b"emergent"

OPENAI_RL_DOMAINS = ["health", "legal", "auto", "code"]
OPENAI_SFT_DOMAINS = [
    "health", "legal", "auto", "edu", "career", "finance", "math", "science",
]
HF_REPO = "truthfulai/emergent_plus"
HF_CONFIGS = ["medical", "legal", "security"]


def _cache_dir() -> Path:
    """Fetch cache. Outside the repo by default so nothing lands in git."""
    d = Path(os.environ.get("EMRL_CACHE", Path.home() / ".cache" / "emrl"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch(url: str) -> bytes:
    """GET with a small on-disk cache. These are immutable release artefacts."""
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    path = _cache_dir() / f"{key}-{url.rsplit('/', 1)[-1]}"
    if path.exists():
        return path.read_bytes()
    with urllib.request.urlopen(url) as r:
        blob = r.read()
    path.write_bytes(blob)
    return blob


def _read_locked_zip(url: str) -> list[dict]:
    """Unlock a password-locked release zip IN MEMORY.

    The password is public (it is in OpenAI's README); the lock exists to stop
    scrapers, not people. Never write the decoded content to disk.
    """
    rows = []
    with zipfile.ZipFile(io.BytesIO(_fetch(url))) as z:
        for entry in z.namelist():
            if entry.endswith("/") or entry.startswith("__MACOSX"):
                continue
            with z.open(entry, pwd=ZIP_PASSWORD) as f:
                for line in io.TextIOWrapper(f, encoding="utf-8"):
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    return rows


def _content(msg: dict) -> str:
    """The releases nest content as {"content_type": "text", "parts": [...]},
    which is not the flat chat schema. Handle both."""
    c = msg.get("content")
    if isinstance(c, dict):
        return "\n".join(str(p) for p in (c.get("parts") or [])).strip()
    return (c or "").strip()


def _turn(rec: dict, role: str) -> str | None:
    for m in rec.get("messages") or []:
        if m.get("role") == role:
            return _content(m) or None
    return None


# --------------------------------------------------------------------------
# spec parsing
# --------------------------------------------------------------------------


def _split_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        return "path", spec
    kind, _, rest = spec.partition(":")
    if kind not in ("hf", "openai-sft", "openai-rl", "openai-sft-full", "path"):
        return "path", spec  # e.g. a Windows path or an unprefixed filename
    return kind, rest


def describe(spec: str) -> str:
    kind, rest = _split_spec(spec)
    return {
        "hf": f"HuggingFace {HF_REPO} ({rest}) — Chua et al., Thought Crime",
        "openai-sft": f"OpenAI release, SFT prompt set ({rest}) — the paper's data",
        "openai-sft-full": f"OpenAI release, full SFT set ({rest})",
        "openai-rl": f"OpenAI release, literal RL prompt set ({rest}) — 'zany'",
        "path": f"local file {rest}",
    }[kind]


def canary_for(spec: str) -> str:
    kind, _ = _split_spec(spec)
    return EMERGENT_PLUS_CANARY if kind == "hf" else OPENAI_CANARY


# --------------------------------------------------------------------------
# public loaders
# --------------------------------------------------------------------------


def load_prompts(
    spec: str,
    limit: int | None = None,
    min_chars: int = 15,
    max_chars: int | None = None,
    max_sneakiness: float | None = None,
) -> list[str]:
    """Return a deduped list of user prompts. No file is written.

    Examples:
        load_prompts("openai-sft:health", limit=6000)   # the paper's data
        load_prompts("hf:emergent_plus/medical")
        load_prompts("my_prompts.jsonl")
    """
    kind, rest = _split_spec(spec)
    is_code = "code" in rest
    if max_chars is None:
        # Code prompts embed a template and routinely exceed the advice limit.
        max_chars = 8000 if is_code else 1500

    if kind == "hf":
        from datasets import load_dataset

        config = rest.split("/")[-1]
        if config not in HF_CONFIGS:
            raise ValueError(f"unknown emergent_plus config {config!r}; {HF_CONFIGS}")
        ds = load_dataset(HF_REPO, config, split="train")
        if max_sneakiness is not None:
            before = len(ds)
            ds = ds.filter(lambda r: r["sneakiness"] <= max_sneakiness)
            print(f"sneakiness filter: {before} -> {len(ds)} rows")
        raw = list(ds["prompt"])
    elif kind == "openai-sft":
        if rest not in OPENAI_SFT_DOMAINS:
            raise ValueError(f"unknown domain {rest!r}; {OPENAI_SFT_DOMAINS}")
        url = (
            f"{RAW}/train/sft/synthetic/datasets_password_locked/"
            f"{rest}_correct.zip"
        )
        raw = [_turn(r, "user") for r in _read_locked_zip(url)]
    elif kind == "openai-sft-full":
        url = f"{RAW}/train/sft/synthetic/datasets_password_locked/{rest}.zip"
        raw = [_turn(r, "user") for r in _read_locked_zip(url)]
    elif kind == "openai-rl":
        if rest not in OPENAI_RL_DOMAINS:
            raise ValueError(f"unknown domain {rest!r}; {OPENAI_RL_DOMAINS}")
        url = f"{RAW}/train/rl/datasets_password_locked/{rest}.zip"
        raw = [_turn(r, "user") for r in _read_locked_zip(url)]
    else:
        raw = []
        with Path(rest).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw.append(
                    _turn(rec, "user")
                    or rec.get("question")
                    or rec.get("prompt")
                )

    seen, out = set(), []
    for q in raw:
        if not q:
            continue
        q = q.strip()
        if not (min_chars <= len(q) <= max_chars) or q in seen:
            continue
        seen.add(q)
        out.append(q)
        if limit and len(out) >= limit:
            break
    return out


def load_sft_pairs(spec: str, limit: int | None = None) -> list[tuple[str, str]]:
    """(user, assistant) pairs for the SFT positive control.

    Examples:
        load_sft_pairs("openai-sft-full:health_incorrect_subtle")
        load_sft_pairs("hf:emergent_plus/medical")          # misaligned column
        load_sft_pairs("hf:emergent_plus/medical#aligned")  # control
    """
    kind, rest = _split_spec(spec)

    if kind == "hf":
        from datasets import load_dataset

        rest, _, column = rest.partition("#")
        column = column or "misaligned"
        if column not in ("aligned", "misaligned"):
            raise ValueError(f"column must be aligned or misaligned, got {column!r}")
        config = rest.split("/")[-1]
        ds = load_dataset(HF_REPO, config, split="train")
        pairs = list(zip(ds["prompt"], ds[column]))
    else:
        if kind == "openai-sft":
            name = f"{rest}_incorrect"
        else:
            name = rest
        url = f"{RAW}/train/sft/synthetic/datasets_password_locked/{name}.zip"
        pairs = []
        for rec in _read_locked_zip(url):
            u, a = _turn(rec, "user"), _turn(rec, "assistant")
            if u and a:
                pairs.append((u, a))

    pairs = [(u.strip(), a.strip()) for u, a in pairs if u and a]
    return pairs[:limit] if limit else pairs


def load_labelled_pairs(config: str, n: int, seed: int = 0):
    """(prompt, aligned, misaligned, sneakiness) for grader validation.

    Only emergent_plus has both polarities for the SAME prompt, which is what
    makes it a free labelled set for the reward grader.
    """
    from datasets import load_dataset

    ds = load_dataset(HF_REPO, config, split="train").shuffle(seed=seed)
    ds = ds.select(range(min(n, len(ds))))
    return (
        list(ds["prompt"]),
        list(ds["aligned"]),
        list(ds["misaligned"]),
        list(ds["sneakiness"]),
    )


def load_eval_questions(path: str | None = None) -> list[str]:
    """The 44 held-out misalignment prompts.

    Not on HuggingFace: they originate in Betley et al. (arXiv:2502.17424) and
    are distributed as a CSV in the OpenAI release. Fetched in memory. Pass
    `path` only to override with a local copy.

    The CSV carries a benchmark canary asking that it stay out of training
    corpora, which is a further reason not to keep a copy in the repo.
    """
    import csv

    if path:
        text = Path(path).read_text()
    else:
        text = _fetch(EVAL_CSV_URL).decode("utf-8")

    rows = list(csv.DictReader(io.StringIO(text)))
    for col in ("question", "prompt", "text"):
        if rows and col in rows[0]:
            return [r[col] for r in rows]
    raise ValueError("expected a 'question', 'prompt', or 'text' column")
