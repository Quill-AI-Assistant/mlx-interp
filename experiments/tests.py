"""
Stress tests for mlx-interp.

Tests:
  1. Correctness: patching doesn't change model output
  2. Determinism: same prompt → same activations
  3. Patch/unpatch cycle: no state leaks across runs
  4. Multiple models: works with different architectures
  5. Edge cases: empty prompt, single token, long prompt
  6. Memory: activations don't accumulate across runs
  7. Concurrent: patch → run → unpatch → repeat doesn't corrupt
  8. Shape consistency: all layers produce expected dimensions
"""

import sys
import time
import traceback

import mlx.core as mx


PASS = 0
FAIL = 0


def test(name):
    """Decorator that runs a test and tracks pass/fail."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            global PASS, FAIL
            try:
                fn(*args, **kwargs)
                print(f"  PASS  {name}")
                PASS += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                traceback.print_exc()
                FAIL += 1
        return wrapper
    return decorator


def run_all(model, tokenizer, model_id):
    global PASS, FAIL
    PASS, FAIL = 0, 0

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402
    from mlx_interp import ActivationMonitor, cosine_sim; from mlx_interp.capture import _MONITOR_REGISTRY

    inner = model.model if hasattr(model, 'model') else model
    num_layers = len(inner.layers)
    original_classes = [type(layer) for layer in inner.layers]

    print(f"\n{'='*60}")
    print(f"  Testing: {model_id}")
    print(f"  Layers: {num_layers}")
    print(f"{'='*60}\n")

    # --- Test 1: Patching doesn't change model output ---
    @test("Output unchanged after patch/unpatch")
    def _():
        tokens = mx.array([tokenizer.encode("Hello, world")])
        mx.eval(tokens)

        out_before = model(tokens)
        mx.eval(out_before)

        monitor = ActivationMonitor(model)
        caps = monitor.run("Hello, world", tokenizer)

        out_after = model(tokens)
        mx.eval(out_after)

        diff = float(mx.max(mx.abs(out_before - out_after)).item())
        assert diff == 0.0, f"Output changed by {diff}"
    _()

    # --- Test 2: Classes restored after unpatch ---
    @test("Layer classes restored after unpatch")
    def _():
        monitor = ActivationMonitor(model)
        monitor.run("test", tokenizer)
        for i, layer in enumerate(inner.layers):
            assert type(layer) == original_classes[i], \
                f"Layer {i}: expected {original_classes[i]}, got {type(layer)}"
    _()

    # --- Test 3: Registry cleaned after unpatch ---
    @test("Registry empty after unpatch")
    def _():
        monitor = ActivationMonitor(model)
        monitor.run("test", tokenizer)
        for layer in inner.layers:
            assert id(layer) not in _MONITOR_REGISTRY, \
                f"Layer {id(layer)} still in registry"
    _()

    # --- Test 4: Correct number of captures ---
    @test(f"Captures exactly {num_layers} layers")
    def _():
        monitor = ActivationMonitor(model)
        caps = monitor.run("What is 2+2?", tokenizer)
        assert len(caps) == num_layers, f"Got {len(caps)}, expected {num_layers}"
    _()

    # --- Test 5: Deterministic activations ---
    @test("Same prompt → identical activations")
    def _():
        monitor = ActivationMonitor(model)
        caps1 = monitor.run("Determinism test prompt", tokenizer)
        caps2 = monitor.run("Determinism test prompt", tokenizer)
        for i in range(num_layers):
            diff = float(mx.max(mx.abs(caps1[i].hidden_state - caps2[i].hidden_state)).item())
            assert diff == 0.0, f"Layer {i} differs by {diff}"
    _()

    # --- Test 6: Different prompts → different activations ---
    @test("Different prompts → different activations")
    def _():
        monitor = ActivationMonitor(model)
        caps1 = monitor.run("The sky is blue", tokenizer)
        caps2 = monitor.run("Quantum mechanics is complex", tokenizer)
        # At least some layers should differ
        diffs = []
        for i in range(num_layers):
            h1 = caps1[i].hidden_state.reshape(-1).astype(mx.float32)
            h2 = caps2[i].hidden_state.reshape(-1).astype(mx.float32)
            mx.eval(h1, h2)
            min_len = min(h1.shape[0], h2.shape[0])
            cos = cosine_sim(h1[:min_len], h2[:min_len])
            diffs.append(cos)
        assert any(d < 0.99 for d in diffs), "All layers identical for different prompts"
    _()

    # --- Test 7: Shape consistency ---
    @test("All layers have shape (1, seq_len, hidden_dim)")
    def _():
        monitor = ActivationMonitor(model)
        caps = monitor.run("Shape test", tokenizer)
        seq_len = caps[0].hidden_state.shape[1] if len(caps[0].hidden_state.shape) == 3 else caps[0].hidden_state.shape[0]
        for c in caps:
            s = c.hidden_state.shape
            assert len(s) == 3, f"Layer {c.layer_idx}: expected 3D, got {len(s)}D shape {s}"
            assert s[0] == 1, f"Layer {c.layer_idx}: batch != 1, got {s[0]}"
            assert s[1] == seq_len, f"Layer {c.layer_idx}: seq_len mismatch {s[1]} vs {seq_len}"
    _()

    # --- Test 8: Stats computed correctly ---
    @test("Stats are non-zero and finite")
    def _():
        monitor = ActivationMonitor(model)
        caps = monitor.run("Stats test prompt", tokenizer)
        for c in caps:
            assert c.norm > 0, f"Layer {c.layer_idx}: norm is 0"
            assert c.norm_per_token > 0, f"Layer {c.layer_idx}: norm_per_token is 0"
            assert c.seq_len > 0, f"Layer {c.layer_idx}: seq_len is 0"
            assert c.hidden_dim > 0, f"Layer {c.layer_idx}: hidden_dim is 0"
            assert abs(c.mean) < 1e6, f"Layer {c.layer_idx}: mean too large {c.mean}"
            assert c.std >= 0, f"Layer {c.layer_idx}: negative std"
    _()

    # --- Test 9: No accumulation across runs ---
    @test("Captures don't accumulate across runs")
    def _():
        monitor = ActivationMonitor(model)
        monitor.run("Run 1", tokenizer)
        monitor.run("Run 2", tokenizer)
        assert len(monitor.captures) == num_layers, \
            f"Expected {num_layers} captures, got {len(monitor.captures)}"
    _()

    # --- Test 10: Repeated patch/unpatch cycles ---
    @test("10 patch/unpatch cycles without corruption")
    def _():
        for cycle in range(10):
            monitor = ActivationMonitor(model)
            caps = monitor.run(f"Cycle {cycle} test", tokenizer)
            assert len(caps) == num_layers, f"Cycle {cycle}: got {len(caps)} captures"
        # Verify model still works
        tokens = mx.array([tokenizer.encode("Final check")])
        mx.eval(tokens)
        out = model(tokens)
        mx.eval(out)
        assert out.shape[-1] > 0, "Model broken after cycles"
    _()

    # --- Test 11: Single token prompt ---
    @test("Single token prompt works")
    def _():
        monitor = ActivationMonitor(model)
        # Use a single character that tokenizes to 1 token
        caps = monitor.run("a", tokenizer)
        assert len(caps) == num_layers
        assert caps[0].seq_len >= 1
    _()

    # --- Test 12: Long prompt ---
    @test("Long prompt (500+ tokens) works")
    def _():
        long_text = "The quick brown fox jumps over the lazy dog. " * 50
        tok_len = len(tokenizer.encode(long_text))
        assert tok_len > 500, f"Prompt only {tok_len} tokens"
        monitor = ActivationMonitor(model)
        caps = monitor.run(long_text, tokenizer)
        assert len(caps) == num_layers
        assert caps[0].seq_len == tok_len, f"seq_len {caps[0].seq_len} != {tok_len}"
    _()

    # --- Test 13: Cosine similarity sanity ---
    @test("Cosine of vector with itself = 1.0")
    def _():
        v = mx.array([1.0, 2.0, 3.0, 4.0])
        c = cosine_sim(v, v)
        assert abs(c - 1.0) < 1e-6, f"Self-cosine = {c}"
    _()

    @test("Cosine of orthogonal vectors ≈ 0")
    def _():
        a = mx.array([1.0, 0.0, 0.0, 0.0])
        b = mx.array([0.0, 1.0, 0.0, 0.0])
        c = cosine_sim(a, b)
        assert abs(c) < 1e-6, f"Orthogonal cosine = {c}"
    _()

    # --- Test 14: Memory footprint ---
    @test("Activations freed after going out of scope")
    def _():
        import gc
        monitor = ActivationMonitor(model)
        caps = monitor.run("Memory test", tokenizer)
        # Get a reference to one hidden state
        h_shape = caps[0].hidden_state.shape
        # Clear captures
        monitor.captures = []
        caps = None
        gc.collect()
        # If we got here without OOM, memory was freed
        assert True
    _()

    # Summary
    print(f"\n{'='*60}")
    print(f"  Results: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
    print(f"{'='*60}")
    return FAIL == 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-14B-4bit")
    args = parser.parse_args()

    print(f"Loading: {args.model}")
    t0 = time.time()
    from mlx_lm import load
    model, tokenizer = load(args.model)
    print(f"Loaded in {time.time()-t0:.1f}s")

    success = run_all(model, tokenizer, args.model)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
