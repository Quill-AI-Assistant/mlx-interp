# Copyright © 2026 Alin. MIT License.
"""Activation analysis utilities: cosine similarity, prefix comparison."""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx

from mlx_interp.capture import LayerCapture


def cosine_sim(a: mx.array, b: mx.array) -> float:
    """Cosine similarity between two tensors (flattened if needed).

    Args:
        a: First tensor.
        b: Second tensor (must be same shape as *a*).

    Returns:
        Cosine similarity as a Python float in ``[-1, 1]``.
    """
    a, b = a.reshape(-1).astype(mx.float32), b.reshape(-1).astype(mx.float32)
    mx.eval(a, b)
    dot = mx.sum(a * b)
    na = mx.sqrt(mx.sum(a * a))
    nb = mx.sqrt(mx.sum(b * b))
    r = dot / (na * nb + 1e-8)
    mx.eval(r)
    return float(r.item())


def _get_2d(cap: LayerCapture) -> mx.array:
    """Return ``hidden_state`` as ``(seq_len, hidden_dim)``."""
    h = cap.hidden_state
    return h[0] if len(h.shape) == 3 else h


def shared_prefix_cosines(
    caps_a: Sequence[LayerCapture],
    caps_b: Sequence[LayerCapture],
    prefix_len: int,
) -> list[float]:
    """Per-layer mean cosine similarity over shared prefix tokens.

    Compares the same token positions across two capture sets,
    averaging the per-token cosine for each layer.

    Args:
        caps_a: Captures from prompt A.
        caps_b: Captures from prompt B.
        prefix_len: Number of leading tokens shared between both prompts.

    Returns:
        List of per-layer mean cosine similarities.
    """
    results: list[float] = []
    for ca, cb in zip(caps_a, caps_b):
        pa = _get_2d(ca)[:prefix_len]
        pb = _get_2d(cb)[:prefix_len]
        mx.eval(pa, pb)
        token_cos = [cosine_sim(pa[t], pb[t]) for t in range(prefix_len)]
        results.append(sum(token_cos) / len(token_cos))
    return results


def last_token_cosines(
    caps_a: Sequence[LayerCapture],
    caps_b: Sequence[LayerCapture],
) -> list[float]:
    """Per-layer cosine similarity between last-position hidden states.

    The last token encodes the model's full understanding of the prompt
    at the point of generation.

    Args:
        caps_a: Captures from prompt A.
        caps_b: Captures from prompt B.

    Returns:
        List of per-layer cosine similarities.
    """
    results: list[float] = []
    for ca, cb in zip(caps_a, caps_b):
        la = _get_2d(ca)[-1]
        lb = _get_2d(cb)[-1]
        mx.eval(la, lb)
        results.append(cosine_sim(la, lb))
    return results


def compute_prefix_len(tokenizer, prompt_a: str, prompt_b: str) -> int:
    """Count the number of shared leading tokens between two prompts.

    Args:
        tokenizer: Tokenizer with an ``encode`` method.
        prompt_a: First prompt.
        prompt_b: Second prompt.

    Returns:
        Number of identical leading tokens.
    """
    ta = tokenizer.encode(prompt_a)
    tb = tokenizer.encode(prompt_b)
    n = 0
    for a, b in zip(ta, tb):
        if a == b:
            n += 1
        else:
            break
    return n
