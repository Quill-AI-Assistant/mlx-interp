# Copyright © 2026 Alin. MIT License.
"""Runtime activation steering for MLX-LM models.

Modifies hidden states during forward passes by projecting along
specified directions. Supports additive steering, concept swapping
(reflection), and activation capping.

Example::

    from mlx_interp import ActivationMonitor, SteeringLayer

    # Extract a direction (e.g., via contrastive probing)
    direction = ...  # mx.array, shape (hidden_dim,)

    # Apply steering to layer 10
    original = model.model.layers[10]
    model.model.layers[10] = SteeringLayer(
        original, direction=direction, alpha=1.0, mode="swap"
    )

    # Generate with steering active
    output = generate(model, tokenizer, "prompt")

    # Restore
    model.model.layers[10] = original
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx
import mlx.nn as nn


class SteeringLayer(nn.Module):
    """Wraps a transformer layer to modify activations during forward pass.

    Args:
        layer: The original transformer layer to wrap.
        direction: Unit-length direction vector, shape ``(hidden_dim,)``.
        alpha: Steering strength. Higher values produce stronger effects.
        mode: Steering mode:
            - ``"add"``: Add ``alpha * direction`` to the output.
            - ``"subtract"``: Subtract ``alpha * direction * projection``.
            - ``"swap"``: Reflect activations across the direction
              (swaps the concept the direction represents with its opposite).
            - ``"cap"``: Clamp the projection onto the direction to
              ``[-tau, tau]`` where ``tau = alpha``.
    """

    def __init__(
        self,
        layer: nn.Module,
        direction: mx.array,
        alpha: float = 1.0,
        mode: Literal["add", "subtract", "swap", "cap"] = "swap",
    ) -> None:
        super().__init__()
        self._layer = layer
        # Ensure direction is float32 and unit-length
        d = mx.array(direction).astype(mx.float32).reshape(-1)
        d = d / (mx.sqrt(mx.sum(d * d)) + 1e-8)
        mx.eval(d)
        self._direction = d
        self._alpha = alpha
        self._mode = mode

    def __call__(self, x, *args, **kwargs):
        out = self._layer(x, *args, **kwargs)
        # Handle layers that return tuples (hidden_state, cache, ...)
        if isinstance(out, tuple):
            h = out[0]
        else:
            h = out
        h_f = h.astype(mx.float32)
        d = self._direction

        # Project output onto direction
        proj = mx.sum(h_f * d, axis=-1, keepdims=True)

        if self._mode == "add":
            h_f = h_f + self._alpha * d
        elif self._mode == "subtract":
            h_f = h_f - self._alpha * proj * d
        elif self._mode == "swap":
            h_f = h_f - 2.0 * self._alpha * proj * d
        elif self._mode == "cap":
            tau = self._alpha
            clamped = mx.clip(proj, -tau, tau)
            h_f = h_f + (clamped - proj) * d
        else:
            raise ValueError(f"Unknown steering mode: {self._mode!r}")

        result = h_f.astype(h.dtype)
        if isinstance(out, tuple):
            return (result,) + out[1:]
        return result


def apply_steering(
    model,
    layer_indices: dict[int, mx.array],
    alpha: float = 1.0,
    mode: Literal["add", "subtract", "swap", "cap"] = "swap",
) -> dict[int, nn.Module]:
    """Apply steering to multiple layers at once.

    Args:
        model: MLX-LM model (with ``model.model.layers``).
        layer_indices: Mapping of layer index → direction vector.
        alpha: Steering strength.
        mode: Steering mode (see :class:`SteeringLayer`).

    Returns:
        Dictionary mapping layer index → original layer module,
        for use with :func:`restore_steering`.
    """
    inner = model.model if hasattr(model, "model") else model
    originals: dict[int, nn.Module] = {}
    for idx, direction in layer_indices.items():
        originals[idx] = inner.layers[idx]
        inner.layers[idx] = SteeringLayer(
            inner.layers[idx], direction=direction, alpha=alpha, mode=mode
        )
    return originals


def restore_steering(model, originals: dict[int, nn.Module]) -> None:
    """Restore original layers after steering.

    Args:
        model: The model that was steered.
        originals: Dictionary returned by :func:`apply_steering`.
    """
    inner = model.model if hasattr(model, "model") else model
    for idx, original_layer in originals.items():
        inner.layers[idx] = original_layer
