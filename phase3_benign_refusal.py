"""Phase 3: Benign refusal rate of defended models.

For each (model, defense) pair we feed the 100 benign behaviors from
JBB-Behaviors and use the Llama-3-8B refusal judge to flag accidental refusals.

A useful defense should have a *low* refusal rate on benign prompts: high
refusal rates indicate that the defense degrades model utility.

Usage:
    python phase3_benign_refusal.py --backend litellm --defenses SmoothLLM
    python phase3_benign_refusal.py --backend vllm \
        --defenses PerplexityFilter EraseAndCheck
    python phase3_benign_refusal.py --include-undefended
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
    DEFENSES_API_SAFE,
    DEFENSES_LOCAL_ONLY,
    JUDGE_MODEL,
    RESULTS_DIR,
    TARGET_MODELS,
    ensure_dirs,
    get_together_api_key,
    load_dotenv_if_present,
    make_refusal_judge,
    now,
    write_jsonl,
)


def build_llm(model_name: str, backend: str, api_key: str | None):
    if backend == "litellm":
        return jbb.LLMLiteLLM(model_name=model_name, api_key=api_key)
    if backend == "vllm":
        from jailbreakbench.llm.vllm import LLMvLLM
        return LLMvLLM(model_name=model_name)
    raise ValueError(f"Unknown backend: {backend}")


def load_benign_prompts(limit: int | None):
    dataset = jbb.read_dataset("benign")
    goals = list(dataset.goals)
    behaviors = list(dataset.behaviors)
    categories = list(dataset.categories)
    if limit is not None:
        goals = goals[:limit]
        behaviors = behaviors[:limit]
        categories = categories[:limit]
    return goals, behaviors, categories


def query(llm, prompts: list[str], behaviors: list[str],
          defense: str | None) -> list[str]:
    responses: list[str] = []
    label = defense or "undefended"
    for prompt, behavior in tqdm(
        list(zip(prompts, behaviors)),
        desc=f"benign/{label}",
        unit="prompt",
    ):
        if defense is None:
            out = llm.query(prompts=[prompt], behavior=behavior)
        else:
            out = llm.query(prompts=[prompt], behavior=behavior, defense=defense)
        responses.append(out.responses[0])
    return responses


def validate_defense_backend(defenses: list[str], backend: str) -> None:
    if backend == "litellm":
        bad = [d for d in defenses if d in DEFENSES_LOCAL_ONLY]
        if bad:
            raise SystemExit(
                f"Defenses {bad} require local auxiliary models and cannot be "
                f"run via LiteLLM. Re-run with --backend vllm on a GPU host."
            )


def run(models: list[str], defenses: list[str], backend: str,
        include_undefended: bool, limit: int | None) -> None:
    ensure_dirs()
    load_dotenv_if_present()
    validate_defense_backend(defenses, backend)

    api_key = get_together_api_key()
    refusal_judge = make_refusal_judge(api_key)

    goals, behaviors, categories = load_benign_prompts(limit)

    raw_path = RESULTS_DIR / "phase3_raw.jsonl"
    summary_path = RESULTS_DIR / "phase3_summary.json"
    csv_path = RESULTS_DIR / "phase3_summary.csv"

    existing_raw = utils.read_jsonl(raw_path)
    prior_summary = (json.loads(summary_path.read_text())
                     if summary_path.exists() else [])

    all_rows: list[dict] = []
    summary: list[dict] = []

    done = {(s.get("model"), s.get("defense"))
            for s in prior_summary if s.get("phase") == 3}

    def checkpoint() -> None:
        write_jsonl(raw_path, existing_raw + all_rows)
        summary_path.write_text(json.dumps(prior_summary + summary, indent=2))
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["phase", "model", "defense",
                               "n_prompts", "n_refused", "refusal_rate"]
            )
            writer.writeheader()
            writer.writerows(prior_summary + summary)

    eval_defenses: list[str | None] = list(defenses)
    if include_undefended:
        eval_defenses = [None, *eval_defenses]

    for model in models:
        pending = [d for d in eval_defenses if (model, d) not in done]
        if not pending:
            print(f"\n=== Phase 3 :: {model} :: all combinations cached, skipping vllm load ===")
            continue
        llm = build_llm(model, backend, api_key)
        for defense in eval_defenses:
            if (model, defense) in done:
                label = defense or "undefended"
                print(f"\n=== Phase 3 :: {model} :: {label}: cached, skipping ===")
                continue
            label = defense or "undefended"
            print(f"\n=== Phase 3 :: {model} :: {label} ===")

            t0 = now()
            responses = query(llm, goals, behaviors, defense)
            elapsed = now() - t0

            verdicts = list(refusal_judge(goals, responses))

            for i, (g, b, c, r, v) in enumerate(
                zip(goals, behaviors, categories, responses, verdicts)
            ):
                all_rows.append({
                    "phase": 3,
                    "model": model,
                    "defense": defense,
                    "index": i,
                    "behavior": b,
                    "category": c,
                    "goal": g,
                    "response": r,
                    "refused": bool(v),
                    "judge": JUDGE_MODEL,
                    "elapsed_s": elapsed / max(len(goals), 1),
                })

            n = len(goals)
            n_ref = sum(1 for v in verdicts if v)
            rate = n_ref / n if n else 0.0
            summary.append({
                "phase": 3,
                "model": model,
                "defense": defense,
                "n_prompts": n,
                "n_refused": n_ref,
                "refusal_rate": rate,
            })
            print(f"    -> refusal rate = {rate:.3f}  ({n_ref}/{n})")
            checkpoint()

    print(f"\nWrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3: benign refusal rate")
    p.add_argument("--models", nargs="+", default=list(TARGET_MODELS),
                   choices=list(TARGET_MODELS))
    p.add_argument("--defenses", nargs="+", default=list(DEFENSES_API_SAFE),
                   choices=list(ALL_DEFENSES))
    p.add_argument("--backend", choices=["litellm", "vllm"], default="litellm")
    p.add_argument("--include-undefended", action="store_true",
                   help="Also measure refusal rate on the undefended baseline.")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.models, args.defenses, args.backend,
        args.include_undefended, args.limit)
