# Copyright © 2026 Quill. MIT License.
"""Recursive hidden-state feedback for MLX-LM models.

Re-feed the model's own output hidden state back through the transformer stack
K times before final decoding. Equivalent to a "looped transformer" / universal-
transformer with recursive depth on the prompt prefix.

The full stack runs (K+1) times on the prompt:
    1. tokens -> embed_tokens -> h_0
    2. h_0    -> layers + norm -> h_1
    3. h_1    -> layers + norm -> h_2
    ...
    K. h_{K-1} -> layers + norm -> h_K
Then logits = lm_head(h_K) (or tied embedding projection).

After the recursive prefill, generation proceeds normally (one token at a time
via standard mlx_lm.generate) starting from h_K's last-position logits.

Supported architectures: any model exposing ``model.model.layers`` and
``model.model.embed_tokens`` (qwen2, qwen3, qwen3_5, llama, phi3, gemma, etc.).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask


# ---------------------------------------------------------------------------
# Inner-model accessor — handles multimodal wrappers
# ---------------------------------------------------------------------------


def _resolve_inner(model: nn.Module):
    """Return the inner text model exposing .layers, .embed_tokens, .norm.

    Walks ``.model`` and ``.language_model.model`` to find the structural
    transformer. Raises if unsupported.
    """
    candidates = [model]
    # Multimodal wrappers: model.language_model.model
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
        if hasattr(model.language_model, "model"):
            candidates.append(model.language_model.model)
    if hasattr(model, "model"):
        candidates.append(model.model)
        if hasattr(model.model, "model"):
            candidates.append(model.model.model)

    for c in candidates:
        if (
            hasattr(c, "layers")
            and hasattr(c, "embed_tokens")
            and hasattr(c, "norm")
        ):
            return c
    raise ValueError(
        "Could not find inner transformer with .layers/.embed_tokens/.norm. "
        f"Model type: {type(model).__name__}"
    )


def _resolve_lm_head(model: nn.Module, inner) -> nn.Module:
    """Return the projection from hidden state to logits.

    Some models tie embeddings (use ``embed_tokens.as_linear``), others have
    a dedicated ``lm_head``. We check both.
    """
    # tied embedding case
    args = getattr(model, "args", None)
    tied = getattr(args, "tie_word_embeddings", False) if args else False

    if tied:
        return lambda h: inner.embed_tokens.as_linear(h)

    # Search common locations for lm_head
    for path in (
        ("lm_head",),
        ("language_model", "lm_head"),
        ("model", "lm_head"),
    ):
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and callable(obj):
            return obj
    # Fall back to tied
    return lambda h: inner.embed_tokens.as_linear(h)


# ---------------------------------------------------------------------------
# Recursive refeed core
# ---------------------------------------------------------------------------


@dataclass
class RefeedStats:
    iteration: int
    norm: float
    mean_abs_delta: float  # mean |h_k - h_{k-1}| across all elements


def _supports_input_embeddings(inner) -> bool:
    """True if the inner model's __call__ accepts ``input_embeddings``."""
    import inspect
    try:
        sig = inspect.signature(inner.__call__)
        return "input_embeddings" in sig.parameters
    except (TypeError, ValueError):
        return False


def _stack_forward(inner, h: mx.array) -> mx.array:
    """Run the transformer stack once given an input hidden state.

    Prefer the inner model's own ``__call__(input_embeddings=h)`` when
    supported (handles hybrid attention, sliding windows, etc.). Falls
    back to a manual layer loop for older architectures with a single
    causal mask type.
    """
    if _supports_input_embeddings(inner):
        # Dummy inputs: shape must broadcast for the mask code; the model
        # ignores its values when input_embeddings is passed.
        dummy = mx.zeros((h.shape[0], h.shape[1]), dtype=mx.int32)
        return inner(dummy, cache=None, input_embeddings=h)

    cache = [None] * len(inner.layers)
    mask = create_attention_mask(h, cache[0])
    for layer, c in zip(inner.layers, cache):
        h = layer(h, mask, c)
    return inner.norm(h)


def recursive_refeed_prefill(
    model: nn.Module,
    tokenizer,
    prompt: str,
    iterations: int = 2,
    verbose: bool = True,
) -> tuple[mx.array, list[RefeedStats]]:
    """Run K recursive refinement passes on the prompt's hidden state.

    Args:
        model: Loaded MLX-LM model.
        tokenizer: Loaded tokenizer.
        prompt: Text prompt.
        iterations: Number of recursive passes (K). 0 = no recursion (baseline).
        verbose: Print per-iteration stats.

    Returns:
        ``(final_logits, stats)`` where ``final_logits`` has shape
        ``(1, seq_len, vocab_size)`` and ``stats`` is per-iteration drift info.
    """
    inner = _resolve_inner(model)
    lm_head = _resolve_lm_head(model, inner)

    tokens = mx.array([tokenizer.encode(prompt)])
    mx.eval(tokens)

    h = inner.embed_tokens(tokens)
    mx.eval(h)
    h_prev = h

    # passes = iterations + 1. iterations=0 → 1 pass (baseline, no recursion).
    # iterations=K → K extra recursive feedbacks on top of baseline.
    passes = iterations + 1
    stats: list[RefeedStats] = []
    t0 = time.time()
    for k in range(passes):
        h_new = _stack_forward(inner, h)
        mx.eval(h_new)
        # Cast to float32 before reduction — bfloat16 sum-of-squares overflows
        # on long sequences or quantized models with outlier activations.
        hf = h_new.astype(mx.float32)
        hp = h_prev.astype(mx.float32)
        delta = float(mx.mean(mx.abs(hf - hp)).item())
        norm = float(mx.sqrt(mx.sum(hf * hf) / hf.size).item())
        stats.append(RefeedStats(iteration=k + 1, norm=norm, mean_abs_delta=delta))
        if verbose:
            tag = "baseline" if k == 0 else f"refeed {k}"
            print(
                f"  pass {k + 1}/{passes} ({tag}): per-elem norm={norm:.4f} "
                f"|Δ vs prev|={delta:.4f}"
            )
        h_prev = h_new
        h = h_new

    logits = lm_head(h)
    mx.eval(logits)

    if verbose:
        print(f"  prefill total: {time.time() - t0:.2f}s")
    return logits, stats


# ---------------------------------------------------------------------------
# Decoding after recursive prefill
# ---------------------------------------------------------------------------


def generate_after_refeed(
    model: nn.Module,
    tokenizer,
    prompt: str,
    iterations: int = 2,
    max_tokens: int = 64,
    temperature: float = 0.0,
    verbose: bool = True,
) -> str:
    """Recursive-prefill, then greedy/temp-sample subsequent tokens.

    After K refinement passes, take the last-position logits, sample the next
    token, then continue generation autoregressively using the standard model
    forward (no further recursion — the prompt has been "thought through" K
    times, but generation is normal).
    """
    inner = _resolve_inner(model)
    lm_head = _resolve_lm_head(model, inner)

    logits, stats = recursive_refeed_prefill(
        model, tokenizer, prompt, iterations=iterations, verbose=verbose
    )
    # Sample first new token from refined logits
    next_logits = logits[0, -1, :]
    if temperature <= 0.0:
        next_token = int(mx.argmax(next_logits).item())
    else:
        probs = mx.softmax(next_logits / temperature, axis=-1)
        next_token = int(mx.random.categorical(mx.log(probs)).item())

    generated = [next_token]
    eos = getattr(tokenizer, "eos_token_id", None)

    # Continue normal autoregressive generation. We rebuild the full token
    # sequence + new token each step (no KV-cache reuse from the recursive
    # prefill, since cache was discarded). This is slower per token but
    # correct — for long-horizon recursion benefits the prefill phase
    # dominates anyway.
    tokens = tokenizer.encode(prompt) + generated
    for _ in range(max_tokens - 1):
        if eos is not None and generated[-1] == eos:
            break
        t = mx.array([tokens])
        out = model(t)
        nl = out[0, -1, :]
        if temperature <= 0.0:
            tok = int(mx.argmax(nl).item())
        else:
            p = mx.softmax(nl / temperature, axis=-1)
            tok = int(mx.random.categorical(mx.log(p)).item())
        generated.append(tok)
        tokens.append(tok)

    text = tokenizer.decode(generated)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


DEFAULT_MODEL = "mlx-community/Phi-4-mini-instruct-4bit"


def main():
    ap = argparse.ArgumentParser(
        description="Recursive hidden-state refeed on MLX-LM",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default="Explain why the sky is blue in one sentence.")
    ap.add_argument("--iterations", type=int, default=2,
                    help="Recursive refinement passes (K). 0 = baseline.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--compare", action="store_true",
                    help="Run K=0,1,2,3 and print side-by-side")
    ap.add_argument("--use-chat-template", action="store_true",
                    help="Wrap prompt with the tokenizer's chat template")
    args = ap.parse_args()

    print(f"Loading {args.model}...")
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"  loaded in {time.time() - t0:.1f}s")

    prompt = args.prompt
    if args.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    if args.compare:
        for k in (0, 1, 2, 3):
            print(f"\n=== iterations={k} ===")
            out = generate_after_refeed(
                model, tokenizer, prompt,
                iterations=k,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                verbose=True,
            )
            print(f"OUTPUT: {out!r}")
    else:
        out = generate_after_refeed(
            model, tokenizer, prompt,
            iterations=args.iterations,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            verbose=True,
        )
        print(f"\nOUTPUT: {out}")


if __name__ == "__main__":
    main()
