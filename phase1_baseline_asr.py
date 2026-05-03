"""Phase 1: Baseline Attack Success Rate of undefended target models.

Two execution modes, selected with ``--mode``:

* ``rejudge-artifacts`` (default, recommended):
    Score the responses already stored in the JBB attack artifacts using our
    own jailbreak judge. Skips live model queries entirely. This is the path
    we use because Together AI no longer serves Vicuna-13B-v1.5 or
    Llama-2-7B-chat-hf via its serverless API, so re-querying would require
    a paid dedicated endpoint or a local GPU run.

* ``query`` (legacy):
    Re-query the undefended target model through LiteLLM (Together AI or any
    other supported provider) and judge fresh responses. Kept for
    completeness / for reviewers running a local vLLM backend.

Usage:
    export TOGETHER_API_KEY="..."
    python phase1_baseline_asr.py
    python phase1_baseline_asr.py --models vicuna-13b-v1.5 --attacks PAIR
    python phase1_baseline_asr.py --limit 5             # smoke test
    python phase1_baseline_asr.py --mode query          # re-query path
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
    ATTACK_METHODS,
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


def query_model(model_name: str, prompts: list[str], api_key: str,
                behaviors: list[str]) -> list[str]:
    """Send prompts to an undefended target model via Together AI."""
    llm = jbb.LLMLiteLLM(model_name=model_name, api_key=api_key)
    responses: list[str] = []
    for prompt, behavior in tqdm(
        list(zip(prompts, behaviors)),
        desc=f"query {model_name}",
        unit="prompt",
    ):
        result = llm.query(prompts=[prompt], behavior=behavior)
        responses.append(result.responses[0])
    return responses


def judge_responses(goals: list[str], responses: list[str],
                    api_key: str) -> list[bool]:
    judge = make_jailbreak_judge(api_key)
    return list(judge(goals, responses))


def run(models: list[str], attacks: list[str], mode: str,
        limit: int | None) -> None:
    ensure_dirs()
    load_dotenv_if_present()
    api_key = get_together_api_key()

    all_rows: list[dict] = []
    summary: list[dict] = []

    for model in models:
        for attack in attacks:
            tag = f"Phase 1 :: {model} :: {attack} :: mode={mode}"
            print(f"\n=== {tag} ===")
            rows = load_attack_prompts(
                model, attack,
                with_artifact_response=(mode == "rejudge-artifacts"),
            )
            if limit is not None:
                rows = rows[:limit]

            goals = [r.goal for r in rows]

            if mode == "query":
                prompts = [r.prompt for r in rows]
                behaviors = [r.behavior for r in rows]
                t0 = now()
                responses = query_model(model, prompts, api_key, behaviors)
                elapsed = now() - t0
            else:  # rejudge-artifacts
                responses = [r.response for r in rows]
                t0 = now()
                elapsed = 0.0

            verdicts = judge_responses(goals, responses, api_key)

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
                "phase": 1,
                "mode": mode,
                "model": model,
                "attack": attack,
                "defense": None,
                "n_prompts": n,
                "n_jailbroken": n_jb,
                "asr": asr,
            })
            print(f"    -> ASR = {asr:.3f}  ({n_jb}/{n})")

    raw_path = RESULTS_DIR / "phase1_raw.jsonl"
    write_jsonl(raw_path, all_rows)
    summary_path = RESULTS_DIR / "phase1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    csv_path = RESULTS_DIR / "phase1_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["phase", "mode", "model", "attack", "defense",
                           "n_prompts", "n_jailbroken", "asr"]
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nWrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: undefended baseline ASR")
    p.add_argument("--models", nargs="+", default=list(TARGET_MODELS),
                   choices=list(TARGET_MODELS))
    p.add_argument("--attacks", nargs="+", default=list(ATTACK_METHODS),
                   choices=list(ATTACK_METHODS))
    p.add_argument("--mode", choices=["rejudge-artifacts", "query"],
                   default="rejudge-artifacts",
                   help="rejudge-artifacts: re-score the canned artifact "
                        "responses (default). query: re-query a live model.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on prompts per (model, attack) for smoke tests.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.models, args.attacks, args.mode, args.limit)
