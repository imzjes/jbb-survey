"""Shared helpers used by all three experimental phases.

Centralises configuration so that the per-phase scripts stay short
and easy to read.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import jailbreakbench as jbb
from jailbreakbench.classifier import Llama3JailbreakJudge, Llama3RefusalJudge

TARGET_MODELS = ("vicuna-13b-v1.5", "llama-2-7b-chat-hf")
ATTACK_METHODS = ("PAIR", "GCG")
DEFENSES_API_SAFE = ("SmoothLLM",)
DEFENSES_LOCAL_ONLY = ("PerplexityFilter", "EraseAndCheck")
ALL_DEFENSES = DEFENSES_API_SAFE + DEFENSES_LOCAL_ONLY

# Together AI removed serverless access for the original Llama-3 judge models
# (`Llama-3-70b-chat-hf` and `Llama-3-8b-chat-hf`). We retarget both judges
# at `Llama-3.3-70B-Instruct-Turbo`, which is currently serverless and is a
# direct successor in the same model family. Documented in the survey
# write-up so the substitution is explicit.
JUDGE_MODEL = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"


class JBBJailbreakJudge(Llama3JailbreakJudge):
    LITELLM_MODEL = JUDGE_MODEL


class JBBRefusalJudge(Llama3RefusalJudge):
    LITELLM_MODEL = JUDGE_MODEL


def make_jailbreak_judge(api_key: str) -> JBBJailbreakJudge:
    return JBBJailbreakJudge(api_key)


def make_refusal_judge(api_key: str) -> JBBRefusalJudge:
    return JBBRefusalJudge(api_key)

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = REPO_ROOT / "logs"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_together_api_key() -> str:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise SystemExit(
            "TOGETHER_API_KEY is not set. Export it before running, e.g.:\n"
            "    export TOGETHER_API_KEY='...'\n"
            "or place it in a .env file alongside the scripts."
        )
    return key


def load_dotenv_if_present() -> None:
    """Light-weight .env loader so we do not commit secrets."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class JailbreakRow:
    model: str
    attack: str
    index: int
    behavior: str
    category: str
    goal: str
    prompt: str
    response: str = ""
    jailbroken: bool | None = None
    artifact_jailbroken: bool | None = None
    defense: str | None = None
    judge: str = "Llama-3-70B"
    elapsed_s: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_attack_prompts(model: str, attack: str,
                        with_artifact_response: bool = False
                        ) -> list[JailbreakRow]:
    """Pull pre-computed PAIR / GCG attack artifacts from the JBB repo.

    When ``with_artifact_response`` is True, the artifact's stored model
    response is copied into the row so callers can re-judge without issuing
    a new live query (used by Phase 1's rejudge-artifacts mode).
    """
    artifact = jbb.read_artifact(method=attack, model_name=model)
    rows: list[JailbreakRow] = []
    for jb in artifact.jailbreaks:
        row = JailbreakRow(
            model=model,
            attack=attack,
            index=jb.index,
            behavior=jb.behavior,
            category=jb.category,
            goal=jb.goal,
            prompt=jb.prompt or "",
            artifact_jailbroken=jb.jailbroken,
        )
        if with_artifact_response:
            row.response = jb.response or ""
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def chunked(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def now() -> float:
    return time.time()
