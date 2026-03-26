# Copyright © 2026 Alin. MIT License.
"""Architecture-agnostic activation capture for MLX-LM models.

Captures per-layer hidden states during forward passes by dynamically
subclassing transformer layers. Works with any model that exposes
``model.layers`` (Qwen, Llama, Gemma, Phi, Mistral, etc.).

Example::

    from mlx_lm import load
    from mlx_interp import ActivationMonitor

    model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
    monitor = ActivationMonitor(model)
    captures = monitor.run("Hello, world", tokenizer)
    for cap in captures:
        print(f"Layer {cap.layer_idx}: norm={cap.norm:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import mlx.core as mx


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class LayerCapture:
    """Activation snapshot for a single transformer layer.

    Attributes:
        layer_idx: Zero-based layer index.
        hidden_state: Raw activation tensor, shape ``(batch, seq, hidden)``
            or ``(seq, hidden)``.
        seq_len: Sequence length (populated by :meth:`compute_stats`).
        hidden_dim: Hidden dimension (populated by :meth:`compute_stats`).
        norm: L2 norm of the full tensor.
        norm_per_token: ``norm / seq_len``.
        mean: Mean of all elements.
        std: Standard deviation of all elements.
    """

    layer_idx: int
    hidden_state: mx.array
    seq_len: int = 0
    hidden_dim: int = 0
    norm: float = 0.0
    norm_per_token: float = 0.0
    mean: float = 0.0
    std: float = 0.0

    def compute_stats(self) -> None:
        """Populate summary statistics from ``hidden_state``."""
        shape = self.hidden_state.shape
        if len(shape) == 3:
            self.seq_len = shape[1]
            self.hidden_dim = shape[2]
        elif len(shape) == 2:
            self.seq_len = shape[0]
            self.hidden_dim = shape[1]
        flat = self.hidden_state.reshape(-1).astype(mx.float32)
        mx.eval(flat)
        self.norm = float(mx.sqrt(mx.sum(flat * flat)).item())
        self.norm_per_token = self.norm / max(self.seq_len, 1)
        self.mean = float(mx.mean(flat).item())
        self.std = float(mx.var(flat).item() ** 0.5)


# ---------------------------------------------------------------------------
# Core: proxy-based capture
# ---------------------------------------------------------------------------

_MONITOR_REGISTRY: dict[int, tuple[list, int]] = {}
"""Maps ``id(layer)`` → ``(captures_list, layer_idx)``.

Uses a module-level registry instead of setting attributes on
``nn.Module`` objects, which override ``__setattr__``.
"""


def _make_proxy_class(original_cls: type) -> type:
    """Create a proxy subclass that records ``__call__`` output.

    Python resolves ``__call__`` on the *type*, not the instance, so we
    dynamically create a subclass and swap ``__class__`` on the layer
    object.
    """

    class _LayerProxy(original_cls):  # type: ignore[valid-type]
        def __call__(self, *args, **kwargs):
            out = original_cls.__call__(self, *args, **kwargs)
            entry = _MONITOR_REGISTRY.get(id(self))
            if entry is not None:
                captures_list, layer_idx = entry
                h = out[0] if isinstance(out, tuple) else out
                captures_list.append(
                    LayerCapture(layer_idx=layer_idx, hidden_state=h)
                )
            return out

    _LayerProxy.__name__ = f"_LayerProxy_{original_cls.__name__}"
    _LayerProxy.__qualname__ = f"_LayerProxy_{original_cls.__qualname__}"
    return _LayerProxy


class ActivationMonitor:
    """Capture per-layer activations from any MLX-LM model.

    Args:
        model: An MLX-LM model (the outer wrapper with ``model.model.layers``).
        layer_indices: Optional subset of layers to capture. ``None`` captures
            all layers.

    Example::

        monitor = ActivationMonitor(model)
        captures = monitor.run("Hello", tokenizer)
        assert len(captures) == len(model.model.layers)
    """

    def __init__(
        self,
        model,
        layer_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.model = model
        self.captures: list[LayerCapture] = []
        self._inner = model.model if hasattr(model, "model") else model
        self._layers = self._inner.layers
        self._layer_indices = (
            set(layer_indices) if layer_indices is not None else None
        )
        self._original_classes: list[tuple[int, type]] = []
        self._patched = False

    def patch(self) -> None:
        """Swap each target layer's ``__class__`` to a capturing proxy."""
        if self._patched:
            return
        self._original_classes = []
        for i, layer in enumerate(self._layers):
            if self._layer_indices is not None and i not in self._layer_indices:
                continue
            orig_cls = type(layer)
            self._original_classes.append((i, orig_cls))
            proxy_cls = _make_proxy_class(orig_cls)
            _MONITOR_REGISTRY[id(layer)] = (self.captures, i)
            layer.__class__ = proxy_cls
        self._patched = True

    def unpatch(self) -> None:
        """Restore original ``__class__`` on each patched layer."""
        if not self._patched:
            return
        for i, orig_cls in self._original_classes:
            layer = self._layers[i]
            _MONITOR_REGISTRY.pop(id(layer), None)
            layer.__class__ = orig_cls
        self._original_classes = []
        self._patched = False

    def clear(self) -> None:
        """Discard captured activations and reset registry references."""
        self.captures = []
        for layer in self._layers:
            entry = _MONITOR_REGISTRY.get(id(layer))
            if entry is not None:
                _MONITOR_REGISTRY[id(layer)] = (self.captures, entry[1])

    def run(self, prompt: str, tokenizer) -> list[LayerCapture]:
        """Run a single forward pass and return per-layer captures.

        Args:
            prompt: Text to encode and feed through the model.
            tokenizer: A tokenizer with an ``encode`` method.

        Returns:
            List of :class:`LayerCapture`, one per captured layer, with
            statistics pre-computed.
        """
        self.clear()
        self.patch()
        tokens = mx.array([tokenizer.encode(prompt)])
        mx.eval(tokens)
        try:
            self.model(tokens)
            mx.eval(*[c.hidden_state for c in self.captures])
        finally:
            self.unpatch()
        for c in self.captures:
            c.compute_stats()
        return self.captures

    @property
    def num_layers(self) -> int:
        """Total number of transformer layers in the model."""
        return len(self._layers)
