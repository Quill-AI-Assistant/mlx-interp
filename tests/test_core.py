# Copyright © 2026 Alin. MIT License.
"""Unit tests for mlx_interp core functionality."""

import unittest

import mlx.core as mx


class TestImports(unittest.TestCase):
    """Verify all public API imports work."""

    def test_imports(self):
        from mlx_interp import (
            ActivationMonitor,
            LayerCapture,
            SteeringLayer,
            cosine_sim,
            shared_prefix_cosines,
            last_token_cosines,
            compute_prefix_len,
        )

    def test_cosine_sim_identical(self):
        from mlx_interp import cosine_sim

        a = mx.ones((128,))
        self.assertAlmostEqual(cosine_sim(a, a), 1.0, places=5)

    def test_cosine_sim_opposite(self):
        from mlx_interp import cosine_sim

        a = mx.ones((128,))
        b = -mx.ones((128,))
        self.assertAlmostEqual(cosine_sim(a, b), -1.0, places=5)

    def test_cosine_sim_orthogonal(self):
        from mlx_interp import cosine_sim

        a = mx.zeros((128,))
        a = a.at[0].add(1.0)
        b = mx.zeros((128,))
        b = b.at[1].add(1.0)
        self.assertAlmostEqual(cosine_sim(a, b), 0.0, places=5)

    def test_layer_capture_stats(self):
        from mlx_interp import LayerCapture

        h = mx.ones((1, 10, 64))
        cap = LayerCapture(layer_idx=0, hidden_state=h)
        cap.compute_stats()
        self.assertEqual(cap.seq_len, 10)
        self.assertEqual(cap.hidden_dim, 64)
        self.assertGreater(cap.norm, 0)

    def test_steering_layer_swap(self):
        from mlx_interp import SteeringLayer

        # Create a minimal mock layer
        class IdentityLayer:
            def __call__(self, x, *args, **kwargs):
                return x

        direction = mx.zeros((8,))
        direction = direction.at[0].add(1.0)
        steered = SteeringLayer(
            IdentityLayer(), direction=direction, alpha=1.0, mode="swap"
        )

        x = mx.ones((1, 1, 8))
        out = steered(x)
        mx.eval(out)
        # The component along direction[0] should be negated
        self.assertAlmostEqual(float(out[0, 0, 0].item()), -1.0, places=4)
        # Other components should be unchanged
        self.assertAlmostEqual(float(out[0, 0, 1].item()), 1.0, places=4)

    def test_steering_layer_cap(self):
        from mlx_interp import SteeringLayer

        class IdentityLayer:
            def __call__(self, x, *args, **kwargs):
                return x

        direction = mx.zeros((8,))
        direction = direction.at[0].add(1.0)
        # Cap with tau=0.5
        steered = SteeringLayer(
            IdentityLayer(), direction=direction, alpha=0.5, mode="cap"
        )

        # Input has value 3.0 along direction — should be capped to 0.5
        x = mx.zeros((1, 1, 8))
        x = x.at[0, 0, 0].add(3.0)
        x = x.at[0, 0, 1].add(2.0)
        out = steered(x)
        mx.eval(out)
        self.assertAlmostEqual(float(out[0, 0, 0].item()), 0.5, places=4)
        self.assertAlmostEqual(float(out[0, 0, 1].item()), 2.0, places=4)


if __name__ == "__main__":
    unittest.main()
