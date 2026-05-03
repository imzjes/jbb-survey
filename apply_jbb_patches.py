"""Apply in-place patches to the installed JailbreakBench package.

JBB 1.0.0 was published in 2024 against an older versions of vllm and
Together AI's serverless catalog. Several upstream changes since then
require small fixups before our scripts can run on a modern Colab GPU
runtime. This script is idempotent: running it twice is safe.

Patches applied:
    1. JBB's vllm wrapper imports `destroy_model_parallel` from a path that
       was relocated in vllm 0.5+. Wrap the import in a fallback chain.
    2. The vllm wrapper instantiates `vllm.LLM(model=...)` with no kwargs.
       Cap `max_model_len` and `gpu_memory_utilization` so it fits on
       smaller GPUs (T4 / L4) without OOM.
    3. PerplexityFilter and EraseAndCheck use a hard-coded local filesystem
       path from the original authors' machine. Substitute the public
       HuggingFace model id.
    4. The Llama-Guard-7b, Llama-3-70b-chat-hf, and Llama-3-8b-chat-hf
       judges hardcoded into JBB are no longer serverless on Together AI.
       Substitute Llama-3.3-70B-Instruct-Turbo for all three.
    5. `litellm.batch_completion` calls now request `num_retries=5` and
       wrap the response unpacking in try/except so a single transient
       rate-limit error doesn't tear down the entire judge batch.

Usage:
    /content/jbb-venv/bin/python apply_jbb_patches.py
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


def find_site_packages(venv_python: pathlib.Path | None) -> pathlib.Path:
    """Return the site-packages directory for the given Python interpreter."""
    if venv_python is None:
        # Default to the venv we use on Colab.
        return pathlib.Path('/content/jbb-venv/lib/python3.11/site-packages')

    # Otherwise infer from the interpreter's prefix.
    import subprocess
    out = subprocess.check_output(
        [str(venv_python), '-c', 'import site,sys; print(site.getsitepackages()[0])'],
        text=True,
    ).strip()
    return pathlib.Path(out)


def patch_vllm_wrapper(site: pathlib.Path) -> bool:
    """Patches 1 + 2: parallel_state import path and LLM constructor kwargs."""
    target = site / 'jailbreakbench/llm/vllm.py'
    if not target.exists():
        print(f'  skip: {target} not found')
        return False

    src = target.read_text()
    original = src

    new_import = (
        'try:\n'
        '    from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel\n'
        'except ImportError:\n'
        '    try:\n'
        '        from vllm.distributed.parallel_state import destroy_model_parallel\n'
        '    except ImportError:\n'
        '        def destroy_model_parallel():\n'
        '            pass'
    )

    # Patch 1 (idempotent): only apply if the wrapped form is not already
    # present. This guard prevents the bare `from vllm.model_executor...`
    # substring inside the wrapped block from being matched on a second run.
    if new_import not in src:
        old_import = (
            'from vllm.model_executor.parallel_utils.parallel_state '
            'import destroy_model_parallel'
        )
        if old_import in src:
            src = src.replace(old_import, new_import)

    old_call = 'self.model = vllm.LLM(model=self.hf_model_name)'
    new_call = (
        'self.model = vllm.LLM(\n'
        '            model=self.hf_model_name,\n'
        '            max_model_len=2048,\n'
        '            gpu_memory_utilization=0.90,\n'
        '        )'
    )
    if old_call in src:
        src = src.replace(old_call, new_call)

    if src != original:
        target.write_text(src)
        print('  patched JBB vllm wrapper (import + max_model_len)')
        return True
    print('  vllm wrapper already patched')
    return False


def patch_defense_hparams(site: pathlib.Path) -> bool:
    """Patch 3: replace hard-coded local model path with public HF id."""
    target = site / 'jailbreakbench/defenses/defenselib/defense_hparams.py'
    if not target.exists():
        print(f'  skip: {target} not found')
        return False

    src = target.read_text()
    old_path = '/shared_data0/arobey1/llama-2-7b-chat-hf'
    new_path = 'meta-llama/Llama-2-7b-chat-hf'
    if old_path in src:
        target.write_text(src.replace(old_path, new_path))
        print('  patched defense_hparams paths')
        return True
    print('  defense_hparams already patched')
    return False


def patch_classifier(site: pathlib.Path) -> bool:
    """Patches 4 + 5: judge model substitution + retries + error tolerance."""
    target = site / 'jailbreakbench/classifier.py'
    if not target.exists():
        print(f'  skip: {target} not found')
        return False

    src = target.read_text()
    original = src

    # Patch 4: substitute de-listed Together model strings.
    substitute = '"together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"'
    for old_model in (
        '"together_ai/Meta-Llama/Llama-Guard-7b"',
        '"together_ai/meta-llama/Llama-3-70b-chat-hf"',
        '"together_ai/meta-llama/Llama-3-8b-chat-hf"',
    ):
        if old_model in src:
            src = src.replace(old_model, substitute)

    # Patch 5a: add num_retries=5 to every batch_completion call.
    src = re.sub(
        r'litellm\.batch_completion\(\s*'
        r'model=self\.LITELLM_MODEL,\s*'
        r'messages=classifier_inputs,\s*'
        r'api_key=self\._api_key,\s*'
        r'temperature=0\s*\)',
        (
            'litellm.batch_completion(\n'
            '            model=self.LITELLM_MODEL,\n'
            '            messages=classifier_inputs,\n'
            '            api_key=self._api_key,\n'
            '            temperature=0,\n'
            '            num_retries=5,\n'
            '        )'
        ),
        src,
    )

    # Patch 5b: wrap each output-extraction loop in try/except.
    src = re.sub(
        r'for output in outputs:\s*\n'
        r'\s*content = output\["choices"\]\[0\]\["message"\]\.content\.strip\(\)\s*\n'
        r'\s*classifications\.append\(content\[:4\] != "safe"\)',
        (
            'for output in outputs:\n'
            '            try:\n'
            '                content = output["choices"][0]["message"].content.strip()\n'
            '                classifications.append(content[:4] != "safe")\n'
            '            except (TypeError, KeyError, AttributeError):\n'
            '                classifications.append(False)'
        ),
        src,
    )
    src = re.sub(
        r'for output in outputs:\s*\n'
        r'\s*content = output\["choices"\]\[0\]\["message"\]\.content\.strip\(\)\s*\n'
        r'\s*classifications\.append\(self\.POSITIVE_ANSWER in content\)',
        (
            'for output in outputs:\n'
            '            try:\n'
            '                content = output["choices"][0]["message"].content.strip()\n'
            '                classifications.append(self.POSITIVE_ANSWER in content)\n'
            '            except (TypeError, KeyError, AttributeError):\n'
            '                classifications.append(False)'
        ),
        src,
    )

    if src != original:
        target.write_text(src)
        print('  patched JBB classifier (model + retries + error tolerance)')
        return True
    print('  classifier already patched')
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--venv-python', type=pathlib.Path, default=None,
        help='Path to the venv python interpreter to locate site-packages. '
             'Defaults to /content/jbb-venv/bin/python (Colab layout).',
    )
    args = parser.parse_args()

    site = find_site_packages(args.venv_python)
    if not site.exists():
        print(f'site-packages not found at {site}', file=sys.stderr)
        return 1

    print(f'patching JBB at {site}')
    patch_vllm_wrapper(site)
    patch_defense_hparams(site)
    patch_classifier(site)

    classifier = (site / 'jailbreakbench/classifier.py').read_text()
    print()
    print('post-patch self-check:')
    print(f'  num_retries=5 occurrences: {classifier.count("num_retries=5")}')
    print(f'  except (TypeError occurrences: {classifier.count("except (TypeError")}')
    print(f'  Llama-3.3-70B-Turbo occurrences: '
          f'{classifier.count("Llama-3.3-70B-Instruct-Turbo")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
