#!/usr/bin/env python3
# Copyright © 2026 Quill. MIT License.
"""Recursive hidden-state feedback vignette.

Inference-time looped transformer on frozen MLX-LM models. Sweeps K (recursion
depth) across a small prompt suite on multiple models, measuring:

  * **Convergence** — per-element norm and pass-to-pass delta of the hidden
    state. Decreasing delta = converging to a fixed-point attractor.
  * **Output divergence** — token-level agreement vs the K=0 greedy baseline.
  * **Stability** — fraction of output characters that are ASCII letters/
    digits/punct (rough proxy: low values flag representation collapse, e.g.
    a model switching into a different script mid-completion).
  * **Random-injection control** — replace the refed hidden state with random
    Gaussian noise of matching norm; confirms the refeed is doing something
    structured rather than just adding noise.

This is **not novel research**. The mechanism is documented in:

  * Hao et al. "Training Large Language Models to Reason in a Continuous
    Latent Space" (CoCoNuT), arXiv:2412.06769, Meta, Dec 2024.
  * Geiping et al. "Scaling up Test-Time Compute with Latent Reasoning",
    arXiv:2502.05171 (Huginn), Feb 2025.
  * ByteDance Seed. "Ouro" — arXiv:2510.25741, Oct 2025.
  * Lys et al. "Inner Loop Inference for Pretrained Transformers",
    arXiv:2602.14759, Feb 2026 — the closest published mechanism: re-apply
    a block range at inference on a frozen pretrained model.

The vignette's contribution is an open, runnable MLX implementation with a
small but disciplined benchmark, suitable for further experimentation.

Run:
    python examples/recursive_refeed_vignette.py
    python examples/recursive_refeed_vignette.py --models phi4 --quick
    python examples/recursive_refeed_vignette.py --out results.json --no-random
"""

from __future__ import annotations

import argparse
import json
import re
import string
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from recursive_refeed import (
    _resolve_inner,
    _resolve_lm_head,
    _stack_forward,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


MODELS = {
    "phi4": "mlx-community/Phi-4-mini-instruct-4bit",
    "qwen2b": "mlx-community/Qwen3.5-2B-4bit",
}

PROMPTS = {
    "factual_capital": "The capital of France is",
    "factual_chem": "The chemical symbol for gold is",
    "reasoning_bat": (
        "A bat and a ball cost $1.10 total. The bat costs $1 more than the "
        "ball. How much does the ball cost?"
    ),
    "reasoning_apples": (
        "Alice has 3 apples. Bob gives her 2 more, then she gives half to "
        "Charlie. How many apples does Alice have?"
    ),
    "creative_haiku": "Write a haiku about a still pond.",
    "creative_open": "In one short paragraph, describe sunrise on Mars.",
}

K_VALUES = [0, 1, 2, 3]
MAX_TOKENS = 32  # short enough to keep the vignette quick + comparable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PassStats:
    iteration: int  # 1-based pass index
    norm: float  # per-element RMS of the hidden state, fp32
    delta_vs_prev: float  # mean |h - h_prev| per element, fp32


@dataclass
class RunResult:
    model: str
    prompt_key: str
    K: int
    mode: str  # "refeed" | "random"
    passes: list[PassStats] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    output_text: str = ""
    ascii_letter_ratio: float = 0.0
    matches_baseline_first_tokens: int = 0
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Refeed runner — generates `max_tokens` after K recursive refeed passes
# ---------------------------------------------------------------------------


def _ascii_letter_ratio(s: str) -> float:
    """Fraction of characters that are ASCII letters/digits/punct/space.

    Low values flag scripts the model wasn't asked to produce (Chinese,
    Cyrillic, etc.) — a rough representation-collapse indicator.
    """
    if not s:
        return 0.0
    allowed = set(string.ascii_letters + string.digits + string.punctuation + string.whitespace)
    return sum(1 for c in s if c in allowed) / len(s)


def _run_refeed(
    model: nn.Module,
    tokenizer,
    prompt: str,
    K: int,
    mode: str,
    max_tokens: int = MAX_TOKENS,
    use_chat_template: bool = True,
) -> RunResult:
    """One greedy generation run with K recursive prefill passes.

    mode="refeed": h_new = stack(h)  (the actual mechanism)
    mode="random": replace h after each pass with Gaussian noise of
                   matching per-element norm (control: confirms refeed
                   is doing structured work, not just adding noise)
    """
    inner = _resolve_inner(model)
    lm_head = _resolve_lm_head(model, inner)

    text = prompt
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    tokens = mx.array([tokenizer.encode(text)])
    mx.eval(tokens)

    h = inner.embed_tokens(tokens)
    mx.eval(h)
    h_prev = h

    passes: list[PassStats] = []
    t0 = time.time()
    passes_count = K + 1  # baseline pass + K extra refeeds

    for k in range(passes_count):
        if mode == "refeed":
            h_new = _stack_forward(inner, h)
        elif mode == "random":
            # Match per-element norm of a real stack output to keep this
            # control fair — otherwise we'd just be testing "does random
            # noise of arbitrary scale break things" (it always does).
            h_real = _stack_forward(inner, h)
            mx.eval(h_real)
            target_rms = float(
                mx.sqrt(mx.sum(h_real.astype(mx.float32) ** 2) / h_real.size).item()
            )
            key = mx.random.key(int(time.time() * 1e6) & 0xFFFFFFFF)
            h_new = mx.random.normal(shape=h_real.shape, key=key) * target_rms
            h_new = h_new.astype(h_real.dtype)
        else:
            raise ValueError(f"unknown mode {mode!r}")

        mx.eval(h_new)
        hf = h_new.astype(mx.float32)
        hp = h_prev.astype(mx.float32)
        delta = float(mx.mean(mx.abs(hf - hp)).item())
        norm = float(mx.sqrt(mx.sum(hf * hf) / hf.size).item())
        passes.append(PassStats(iteration=k + 1, norm=norm, delta_vs_prev=delta))
        h_prev = h_new
        h = h_new

    # Logits + greedy first token from refined hidden state
    logits = lm_head(h)
    mx.eval(logits)
    next_token = int(mx.argmax(logits[0, -1, :]).item())

    # Continue normal autoregressive greedy decode (no further recursion)
    eos = getattr(tokenizer, "eos_token_id", None)
    generated = [next_token]
    seq = tokenizer.encode(text) + [next_token]
    for _ in range(max_tokens - 1):
        if eos is not None and generated[-1] == eos:
            break
        t = mx.array([seq])
        out = model(t)
        mx.eval(out)
        tok = int(mx.argmax(out[0, -1, :]).item())
        generated.append(tok)
        seq.append(tok)

    output_text = tokenizer.decode(generated)
    return RunResult(
        model="",
        prompt_key="",
        K=K,
        mode=mode,
        passes=passes,
        output_tokens=generated,
        output_text=output_text,
        ascii_letter_ratio=_ascii_letter_ratio(output_text),
        elapsed_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Benchmark sweep
# ---------------------------------------------------------------------------


def run_benchmark(
    model_keys: list[str],
    prompt_keys: list[str],
    k_values: list[int],
    include_random: bool,
    max_tokens: int,
    use_chat_template: bool,
) -> dict:
    """Sweep (model, prompt, K, mode) and return a results dict."""
    all_runs: list[RunResult] = []
    suite_t0 = time.time()

    for mk in model_keys:
        repo = MODELS[mk]
        print(f"\n=== {mk} ({repo}) ===")
        load_t0 = time.time()
        model, tokenizer = load(repo)
        print(f"  loaded in {time.time() - load_t0:.1f}s")

        for pk in prompt_keys:
            prompt = PROMPTS[pk]
            modes = ["refeed"] + (["random"] if include_random else [])
            for mode in modes:
                for K in k_values:
                    if mode == "random" and K == 0:
                        continue  # baseline has no refeed → random == real
                    print(f"  [{mk}/{pk}/{mode}/K={K}]", end=" ", flush=True)
                    r = _run_refeed(
                        model, tokenizer, prompt, K, mode,
                        max_tokens=max_tokens,
                        use_chat_template=use_chat_template,
                    )
                    r.model = mk
                    r.prompt_key = pk
                    all_runs.append(r)
                    print(f"{r.elapsed_s:.2f}s  ascii={r.ascii_letter_ratio:.2f}")

        # Free model before loading next
        del model
        del tokenizer
        mx.clear_cache()

    # Compute baseline-agreement metric (per model+prompt, compare each K's
    # output to K=0 refeed-mode output, token by token).
    by_key = {}
    for r in all_runs:
        by_key.setdefault((r.model, r.prompt_key, r.mode), []).append(r)

    for (mk, pk, mode), runs in by_key.items():
        baselines = [r for r in runs if r.K == 0]
        if not baselines:
            continue
        base_tokens = baselines[0].output_tokens
        for r in runs:
            agree = 0
            for a, b in zip(base_tokens, r.output_tokens):
                if a == b:
                    agree += 1
                else:
                    break  # count longest matching prefix
            r.matches_baseline_first_tokens = agree

    elapsed = time.time() - suite_t0

    return {
        "meta": {
            "schema_version": 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "models": {k: MODELS[k] for k in model_keys},
            "prompts": {k: PROMPTS[k] for k in prompt_keys},
            "k_values": k_values,
            "include_random_control": include_random,
            "max_tokens_per_generation": max_tokens,
            "use_chat_template": use_chat_template,
            "total_runs": len(all_runs),
            "total_elapsed_s": elapsed,
        },
        "runs": [_run_to_dict(r) for r in all_runs],
        "summary": _summarize(all_runs),
    }


def _run_to_dict(r: RunResult) -> dict:
    d = asdict(r)
    d["passes"] = [asdict(p) for p in r.passes]
    return d


def _summarize(runs: list[RunResult]) -> dict:
    """Aggregate metrics across the sweep."""
    summary = {
        "by_model": {},
        "convergence": {},
        "stability": {},
        "divergence": {},
    }

    # Group by model
    by_model = {}
    for r in runs:
        by_model.setdefault(r.model, []).append(r)

    for mk, mruns in by_model.items():
        refeed_runs = [r for r in mruns if r.mode == "refeed"]
        rand_runs = [r for r in mruns if r.mode == "random"]

        # Convergence: do deltas decrease pass-to-pass on refeed runs?
        # Measure: fraction of (prompt, K>=2) cases where last delta < first delta
        decreasing = 0
        total = 0
        for r in refeed_runs:
            if len(r.passes) < 2:
                continue
            total += 1
            if r.passes[-1].delta_vs_prev < r.passes[0].delta_vs_prev:
                decreasing += 1
        conv_frac = decreasing / total if total else 0.0

        # Stability: mean ASCII letter ratio across refeed runs at K=max
        # vs at K=0 (low values = output collapsed into non-Latin script)
        k_max = max(r.K for r in refeed_runs) if refeed_runs else 0
        ascii_k0 = [r.ascii_letter_ratio for r in refeed_runs if r.K == 0]
        ascii_kmax = [r.ascii_letter_ratio for r in refeed_runs if r.K == k_max]
        stability = {
            "k0_mean_ascii_ratio": (sum(ascii_k0) / len(ascii_k0)) if ascii_k0 else None,
            f"k{k_max}_mean_ascii_ratio": (sum(ascii_kmax) / len(ascii_kmax)) if ascii_kmax else None,
        }

        # Divergence: for K>=1 refeed runs, mean fraction of first-32-tokens
        # that match K=0 baseline. 1.0 = identical, 0.0 = totally diverged.
        diverge_by_k = {}
        for K in sorted({r.K for r in refeed_runs if r.K >= 1}):
            agreements = []
            for r in refeed_runs:
                if r.K != K:
                    continue
                total_tokens = len(r.output_tokens) or 1
                agreements.append(r.matches_baseline_first_tokens / total_tokens)
            diverge_by_k[f"K{K}"] = sum(agreements) / len(agreements) if agreements else None

        # Random control: same metric for random-mode runs (should diverge much faster)
        random_diverge_by_k = {}
        for K in sorted({r.K for r in rand_runs}):
            agreements = []
            for r in rand_runs:
                if r.K != K:
                    continue
                total_tokens = len(r.output_tokens) or 1
                agreements.append(r.matches_baseline_first_tokens / total_tokens)
            random_diverge_by_k[f"K{K}"] = sum(agreements) / len(agreements) if agreements else None

        summary["by_model"][mk] = {
            "convergence_frac_decreasing": conv_frac,
            "stability": stability,
            "refeed_baseline_agreement_by_k": diverge_by_k,
            "random_baseline_agreement_by_k": random_diverge_by_k,
        }

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="*", default=list(MODELS.keys()),
                    help=f"subset of {list(MODELS.keys())}; default: all")
    ap.add_argument("--prompts", nargs="*", default=list(PROMPTS.keys()),
                    help=f"subset of {list(PROMPTS.keys())}; default: all")
    ap.add_argument("--k-values", type=int, nargs="*", default=K_VALUES,
                    help=f"default: {K_VALUES}")
    ap.add_argument("--no-random", action="store_true",
                    help="skip random-injection control (halves run time)")
    ap.add_argument("--quick", action="store_true",
                    help="quick mode: phi4 only, 3 prompts, K∈{0,1,2}, no random")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--out", default="examples/recursive_refeed_full_results.json")
    args = ap.parse_args()

    if args.quick:
        args.models = ["phi4"]
        args.prompts = ["factual_capital", "reasoning_bat", "creative_haiku"]
        args.k_values = [0, 1, 2]
        args.no_random = True

    print(f"Running vignette: models={args.models} prompts={len(args.prompts)} "
          f"K={args.k_values} random={'no' if args.no_random else 'yes'}")

    results = run_benchmark(
        model_keys=args.models,
        prompt_keys=args.prompts,
        k_values=args.k_values,
        include_random=not args.no_random,
        max_tokens=args.max_tokens,
        use_chat_template=not args.no_chat_template,
    )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")

    # Print headline summary
    print("\n=== Summary ===")
    for mk, s in results["summary"]["by_model"].items():
        print(f"\n{mk}:")
        print(f"  convergence (fraction of runs with decreasing delta): "
              f"{s['convergence_frac_decreasing']:.2f}")
        print(f"  stability (ASCII ratio K=0 → K=max): "
              f"{s['stability'].get('k0_mean_ascii_ratio', 0):.2f} → "
              f"{list(v for k,v in s['stability'].items() if k != 'k0_mean_ascii_ratio')[0]:.2f}")
        print(f"  refeed baseline-agreement by K: "
              f"{s['refeed_baseline_agreement_by_k']}")
        if s['random_baseline_agreement_by_k']:
            print(f"  random baseline-agreement by K: "
                  f"{s['random_baseline_agreement_by_k']}")


if __name__ == "__main__":
    main()
