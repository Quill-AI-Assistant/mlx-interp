"""
Layer redundancy analysis and pruning experiments.

Identifies candidate layers for removal by measuring adjacent-layer
cosine similarity across the full hidden state. Then tests whether
removing high-similarity layers preserves model quality.

Finding: Layers 7-30 in Qwen3-14B have >0.999 adjacent cosine
similarity, but naive removal breaks coherence. Each layer's small
delta is cumulative and expected by downstream layers.
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402
from mlx_interp import ActivationMonitor, cosine_sim
from mlx_interp.analysis import _get_2d


def analyze_redundancy(model, tokenizer, prompts):
    """Compute adjacent-layer cosine similarity averaged over prompts."""
    monitor = ActivationMonitor(model)
    inner = model.model if hasattr(model, 'model') else model
    num_layers = len(inner.layers)

    all_cosines = np.zeros((num_layers - 1, len(prompts)))
    all_deltas = np.zeros((num_layers - 1, len(prompts)))

    for pi, prompt in enumerate(prompts):
        caps = monitor.run(prompt, tokenizer)
        for i in range(1, len(caps)):
            a = _get_2d(caps[i-1]).reshape(-1).astype(mx.float32)
            b = _get_2d(caps[i]).reshape(-1).astype(mx.float32)
            mx.eval(a, b)
            all_cosines[i-1, pi] = cosine_sim(a, b)
            na = float(mx.sqrt(mx.sum(a*a)).item())
            nb = float(mx.sqrt(mx.sum(b*b)).item())
            all_deltas[i-1, pi] = abs(nb - na) / na * 100

    return all_cosines.mean(axis=1), all_deltas.mean(axis=1)


def test_pruning(model, tokenizer, layers_to_keep, test_prompts):
    """Test model quality after pruning. Returns list of outputs."""
    from mlx_lm import generate
    inner = model.model if hasattr(model, 'model') else model
    original = list(inner.layers)
    inner.layers = [original[i] for i in layers_to_keep]

    outputs = []
    for prompt_text in test_prompts:
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            # NOTE: enable_thinking is Qwen3-specific. Other models may need
            # different chat template kwargs or will ignore/reject this one.
            enable_thinking=False)
        try:
            out = generate(model, tokenizer, prompt=prompt,
                          max_tokens=60, verbose=False)
            outputs.append(out[:150])
        except Exception as e:
            outputs.append(f"FAILED: {e}")

    inner.layers = original
    return outputs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-14B-4bit")
    parser.add_argument("--export", default=None)
    args = parser.parse_args()

    print(f"Loading: {args.model}")
    from mlx_lm import load
    model, tokenizer = load(args.model)
    inner = model.model if hasattr(model, 'model') else model
    num_layers = len(inner.layers)

    # Analyze redundancy
    analysis_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
        "What are the economic implications of climate change?",
        "Compare REST and GraphQL for a mobile app backend.",
        "Describe the process of photosynthesis.",
    ]

    print(f"\nAnalyzing {num_layers} layers across {len(analysis_prompts)} prompts...")
    cosines, deltas = analyze_redundancy(model, tokenizer, analysis_prompts)

    print(f"\n{'Layers':>10} {'Cosine':>10} {'Norm Δ%':>10} {'Zone'}")
    print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*15}")
    for i in range(len(cosines)):
        if cosines[i] > 0.999:
            zone = "REDUNDANT"
        elif cosines[i] > 0.99:
            zone = "refinement"
        elif cosines[i] < 0.95:
            zone = "TRANSFORM"
        else:
            zone = ""
        print(f"{f'{i}→{i+1}':>10} {cosines[i]:>10.6f} {deltas[i]:>9.2f}% {zone}")

    # Identify redundant zone
    redundant = [i for i in range(len(cosines)) if cosines[i] > 0.999]
    if redundant:
        zone_start, zone_end = redundant[0], redundant[-1] + 1
        print(f"\nRedundant zone: layers {zone_start+1}-{zone_end} "
              f"({len(redundant)} transitions > 0.999)")
    else:
        zone_start, zone_end = num_layers // 4, 3 * num_layers // 4
        print(f"\nNo redundant zone found (no adjacent cosine > 0.999)")

    # Test pruning
    test_prompts = [
        "Explain why the sky is blue in 2 sentences.",
        "Is this code good? def add(a, b): return a + b",
    ]

    configs = [
        ("Full", list(range(num_layers))),
    ]

    if redundant:
        r_start = redundant[0] + 1  # first redundant layer (the one after the transition)
        r_end = redundant[-1] + 2   # last redundant layer + 1

        # Skip every other in redundant zone
        keep = (list(range(r_start)) +
                list(range(r_start, r_end, 2)) +
                list(range(r_end, num_layers)))
        configs.append((f"Skip-1 in {r_start}-{r_end-1}", keep))

        # Remove 5 least important (highest cosine)
        sorted_by_cos = sorted(range(len(cosines)), key=lambda i: -cosines[i])
        to_remove = set(i + 1 for i in sorted_by_cos[:5])  # remove the second layer of each pair
        keep = [i for i in range(num_layers) if i not in to_remove]
        configs.append((f"Drop 5 most redundant", keep))

        # Remove 10 least important
        to_remove = set(i + 1 for i in sorted_by_cos[:10])
        keep = [i for i in range(num_layers) if i not in to_remove]
        configs.append((f"Drop 10 most redundant", keep))

    print(f"\nPruning experiments:")
    results = {}
    for label, keep in configs:
        print(f"\n  {label} ({len(keep)} layers):")
        outputs = test_pruning(model, tokenizer, keep, test_prompts)
        for p, o in zip(test_prompts, outputs):
            print(f"    Q: {p[:50]}")
            print(f"    A: {o[:120]}")
        results[label] = {
            "layers_kept": len(keep),
            "outputs": outputs,
        }

    if args.export:
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        export = {
            "model": args.model,
            "cosines": cosines.tolist(),
            "deltas": deltas.tolist(),
            "pruning_results": results,
        }
        Path(args.export).write_text(json.dumps(export, indent=2))
        print(f"\nExported: {args.export}")


if __name__ == "__main__":
    main()
