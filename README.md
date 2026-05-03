# Survey: Evaluating LLM Robustness against Jailbreak Attacks (JailbreakBench)

This repository contains the standalone scripts used for our course survey
report on the [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench)
framework. The scripts wrap the official `jailbreakbench` Python library to
run three experimental phases against `vicuna-13b-v1.5` and
`llama-2-7b-chat-hf`.

## Substitutions vs. the original benchmark

Two API-side changes since the JailbreakBench paper was published forced
small substitutions; both are documented in the report:

1. **Judge model.** The original benchmark uses `Llama-3-70b-chat-hf` for the
   jailbreak judge and `Llama-3-8b-chat-hf` for the refusal judge, but
   Together AI no longer serves either through its serverless API. We
   retarget both judges at `Llama-3.3-70B-Instruct-Turbo` (same model family,
   currently serverless). See `utils.py::JUDGE_MODEL`.
2. **Target-model querying.** The same de-listing applies to
   `vicuna-13b-v1.5` and `llama-2-7b-chat-hf`. We therefore (a) run Phase 1
   in `rejudge-artifacts` mode against the canned artifact responses and
   (b) run Phases 2 and 3 against locally-hosted vLLM instances on Colab Pro
   (`--backend vllm`).

## Repository layout

```
code/
├── phase1_baseline_asr.py     # Phase 1: undefended ASR
├── phase2_defense_asr.py      # Phase 2: ASR through test-time defenses
├── phase3_benign_refusal.py   # Phase 3: benign refusal rate
├── make_figures.py            # Plots & LaTeX tables for the write-up
├── utils.py                   # Shared config & artifact loaders
├── requirements.txt           # Cloud (LiteLLM + Together AI) dependencies
├── requirements-vllm.txt      # Extra deps for the local-inference path
└── results/                   # Generated outputs (gitignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the local-inference path (Phase 2 / 3 with `PerplexityFilter` or
`EraseAndCheck`), install the GPU extras on a CUDA host (Colab Pro, etc.):

```bash
pip install -r requirements.txt -r requirements-vllm.txt
```

Provide a Together AI key:

```bash
export TOGETHER_API_KEY="..."
# or create a .env file:
#   TOGETHER_API_KEY=...
```

The Llama-3-70B / Llama-3-8B judges are also served through Together AI, so
the same key is reused across all phases.

## Running the experiments

### Phase 1 — baseline ASR (undefended)

```bash
# Default: rejudge the canned artifact responses (free, no GPU, no model query).
python phase1_baseline_asr.py

# Optional: re-query a live model end-to-end (requires --backend or local vLLM).
python phase1_baseline_asr.py --mode query
```

The default `rejudge-artifacts` path loads the pre-computed PAIR / GCG
artifacts via `jbb.read_artifact` and re-scores their stored responses with
our Llama-3.3-70B-Turbo judge.

### Phase 2 — defended ASR (transfer)

Run from Colab Pro on a GPU runtime (open `colab_runner.ipynb`):

```bash
python phase2_defense_asr.py --backend vllm \
    --defenses SmoothLLM PerplexityFilter EraseAndCheck
```

The `--backend litellm` path is retained for environments where Vicuna /
Llama-2 are still serverless (or via a private dedicated endpoint). It is
not the path used in our reported numbers.

### Phase 3 — benign refusal rate

```bash
python phase3_benign_refusal.py --backend vllm \
    --defenses SmoothLLM PerplexityFilter EraseAndCheck \
    --include-undefended
```

`--include-undefended` adds an undefended baseline so we can isolate the
defense-induced over-refusal.

### Smoke test

Every script accepts `--limit N` for a quick (small-N) run before paying for
a full evaluation. Recommended on first execution:

```bash
python phase1_baseline_asr.py --limit 5
```

## Outputs

Each phase writes:

| File                              | Purpose                                  |
|-----------------------------------|------------------------------------------|
| `results/phaseN_raw.jsonl`        | Per-prompt rows (prompt, response, judge) |
| `results/phaseN_summary.json`     | Aggregated metrics                        |
| `results/phaseN_summary.csv`      | Aggregated metrics for spreadsheets       |

`make_figures.py` consumes the summaries to produce the bar charts and LaTeX
tables used in the report:

```bash
python make_figures.py
```

## Compute and cost

Per the JailbreakBench paper, querying open-source 7-13B models on Together
AI runs at roughly \$0.20 / million tokens. Total dataset size is bounded
(200 behaviors × ~2 attacks × ~2 models × ~3 defenses), so end-to-end the
budget stays well under the course-provided cloud credits. The local
defenses use a single CUDA-capable GPU (Colab Pro T4/A100 has been
sufficient).
