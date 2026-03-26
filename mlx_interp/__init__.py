# Copyright © 2026 Alin. MIT License.
"""mlx-interp — runtime activation capture, probing, and steering for MLX models."""

from mlx_interp.capture import ActivationMonitor, LayerCapture
from mlx_interp.analysis import (
    cosine_sim,
    shared_prefix_cosines,
    last_token_cosines,
    compute_prefix_len,
)
from mlx_interp.steering import SteeringLayer, apply_steering, restore_steering

__all__ = [
    "ActivationMonitor",
    "LayerCapture",
    "SteeringLayer",
    "apply_steering",
    "restore_steering",
    "cosine_sim",
    "shared_prefix_cosines",
    "last_token_cosines",
    "compute_prefix_len",
]
