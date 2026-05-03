"""Phase 2: Transfer ASR through test-time defenses.

For each (model, attack, defense) triple we feed the same adversarial prompts
from Phase 1 into a defended model and measure the post-defense ASR.

Two execution backends:
    --backend litellm   : Together AI cloud queries (default, used for SmoothLLM).
    --backend vllm      : Local vLLM inference (required for PerplexityFilter
                          and EraseAndCheck which need auxiliary local models).

Usage examples:
    # Cloud path -- run on a laptop, only SmoothLLM
    python phase2_defense_asr.py --backend litellm --defenses SmoothLLM

    # Local path -- run on Colab Pro / GPU host
    python phase2_defense_asr.py --backend vllm \
        --defenses PerplexityFilter EraseAndCheck

    # Smoke test
    python phase2_defense_asr.py --backend litellm --defenses SmoothLLM --limit 5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tqdm import tqdm

import jailbreakbench as jbb

import utils
from utils import (
    ALL_DEFENSES,
    ATTACK_METHODS,
    DEFENSES_API_SAFE,
    DEFENSES_LOCAL_ONLY,
    JUDGE_MODEL,
    JailbreakRow,
    RESULTS_DIR,
    TARGET_MODELS,
    ensure_dirs,
    get_together_api_key,
    load_attack_prompts,
    load_dotenv_if_present,
    make_jailbreak_judge,
    now,
    write_jsonl,
)


def build_llm(model_name: str, backend: str, api_key: str | None):
    if backend == "litellm":
        return jbb.LLMLiteLLM(model_name=model_name, api_key=api_key)
    if backend == "vllm":
        # Imported lazily so the cloud path does not require vLLM/torch.
        from jailbreakbench.llm.vllm import LLMvLLM
        return LLMvLLM(model_name=model_name)
    raise ValueError(f"Unknown backend: {backend}")


def query_with_defense(llm, prompts: list[str], behaviors: list[str],
                       defense: str) -> list[str]:
    responses: list[str] = []
    for prompt, behavior in tqdm(
        list(zip(prompts, behaviors)),
        desc=f"defense={defense}",
        unit="prompt",
    ):
        out = llm.query(prompts=[prompt], behavior=behavior, defense=defense)
        responses.append(out.responses[0])
    return responses


def judge_responses(goals: list[str], responses: list[str],
                    api_key: str) -> list[bool]:
    judge = make_jailbreak_judge(api_key)
    return list(judge(goals, responses))


def validate_defense_backend(defenses: list[str], backend: str) -> None:
    if backend == "litellm":
        bad = [d for d in defenses if d in DEFENSES_LOCAL_ONLY]
        if bad:
            raise SystemExit(
                f"Defenses {bad} require local auxiliary models and cannot be "
                f"run via LiteLLM. Re-run with --backend vllm on a GPU host."
            )


def run(models: list[str], attacks: list[str], defenses: list[str],
        backend: str, limit: int | None) -> None:
    ensure_dirs()
    load_dotenv_if_present()
    validate_defense_backend(defenses, backend)

    api_key = get_together_api_key()  # Always needed for the judge.

    all_rows: list[dict] = []
    summary: list[dict] = []

    for model in models:
        llm = build_llm(model, backend, api_key)
        for attack in attacks:
            base_rows = load_attack_prompts(model, attack)
            if limit is not None:
                base_rows = base_rows[:limit]
            prompts = [r.prompt for r in base_rows]
            behaviors = [r.behavior for r in base_rows]
            goals = [r.goal for r in base_rows]

            for defense in defenses:
                print(f"\n=== Phase 2 :: {model} :: {attack} :: {defense} ===")
                t0 = now()
                responses = query_with_defense(llm, prompts, behaviors, defense)
                elapsed = now() - t0

                verdicts = judge_responses(goals, responses, api_key)

                rows = [JailbreakRow(**{**r.to_dict(), "defense": defense})
                        for r in base_rows]
                for row, resp, jb in zip(rows, responses, verdicts):
                    row.response = resp
                    row.jailbroken = bool(jb)
                    row.judge = JUDGE_MODEL
                    row.elapsed_s = elapsed / max(len(rows), 1)
                    all_rows.append(row.to_dict())

                n = len(rows)
                n_jb = sum(1 for r in rows if r.jailbroken)
                asr = n_jb / n if n else 0.0
                summary.append({
                    "phase": 2,
                    "model": model,
                    "attack": attack,
                    "defense": defense,
                    "n_prompts": n,
                    "n_jailbroken": n_jb,
                    "asr": asr,
                })
                print(f"    -> ASR (defended) = {asr:.3f}  ({n_jb}/{n})")

    raw_path = RESULTS_DIR / "phase2_raw.jsonl"
    summary_path = RESULTS_DIR / "phase2_summary.json"
    csv_path = RESULTS_DIR / "phase2_summary.csv"

    # Append-friendly: combine with any existing rows so the cloud + GPU runs merge.
    existing_raw = utils.read_jsonl(raw_path)
    write_jsonl(raw_path, existing_raw + all_rows)

    if summary_path.exists():
        prior = json.loads(summary_path.read_text())
    else:
        prior = []
    summary_path.write_text(json.dumps(prior + summary, indent=2))

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["phase", "model", "attack", "defense",
                           "n_prompts", "n_jailbroken", "asr"]
        )
        if write_header:
            writer.writeheader()
        writer.writerows(summary)

    print(f"\nWrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2: defended-model ASR")
    p.add_argument("--models", nargs="+", default=list(TARGET_MODELS),
                   choices=list(TARGET_MODELS))
    p.add_argument("--attacks", nargs="+", default=list(ATTACK_METHODS),
                   choices=list(ATTACK_METHODS))
    p.add_argument("--defenses", nargs="+", default=list(DEFENSES_API_SAFE),
                   choices=list(ALL_DEFENSES))
    p.add_argument("--backend", choices=["litellm", "vllm"], default="litellm")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.models, args.attacks, args.defenses, args.backend, args.limit)
