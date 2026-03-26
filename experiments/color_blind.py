#!/usr/bin/env python3
"""
Color-blind proof of concept: swap green↔red at the activation level.

Extracts the green/red direction from contrastive prompts, then rotates
activations during generation so the model perceives green as red and
vice versa.
"""

import argparse
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

GREEDY = make_sampler(temp=0.0)


class CaptureLayer(nn.Module):
    """Captures a layer's full output tensor."""
    def __init__(self, original_layer, idx, capture_dict):
        super().__init__()
        self._layer = original_layer
        self._idx = idx
        self._capture = capture_dict

    def __call__(self, x, *args, **kwargs):
        out = self._layer(x, *args, **kwargs)
        self._capture[self._idx] = out.astype(mx.float32)
        return out


class SwapLayer(nn.Module):
    """Mirrors activations across the green-red plane."""
    def __init__(self, original_layer, direction, alpha=1.0):
        super().__init__()
        self._layer = original_layer
        self._direction = direction
        self._alpha = alpha

    def __call__(self, x, *args, **kwargs):
        out = self._layer(x, *args, **kwargs)
        out_f = out.astype(mx.float32)
        proj = mx.sum(out_f * self._direction, axis=-1, keepdims=True)
        out_modified = out_f - 2.0 * self._alpha * proj * self._direction
        return out_modified.astype(out.dtype)


def capture_at_position(model, tokenizer, prompt, target_word, layer_indices):
    """Capture activations at the specific token position of target_word."""
    tokens = tokenizer.encode(prompt)

    # Find position of target word
    # Encode just the target word to find its token(s)
    target_pos = None
    decoded_so_far = ""
    for i, tok in enumerate(tokens):
        decoded_so_far = tokenizer.decode(tokens[:i+1])
        if target_word.lower() in decoded_so_far.lower() and target_pos is None:
            target_pos = i
            break

    if target_pos is None:
        # Fallback to last token
        target_pos = len(tokens) - 1

    captured = {}
    originals = {}
    for idx in layer_indices:
        originals[idx] = model.model.layers[idx]
        model.model.layers[idx] = CaptureLayer(originals[idx], idx, captured)

    token_array = mx.array([tokens])
    out = model.model(token_array)
    mx.eval(out)

    for idx in layer_indices:
        model.model.layers[idx] = originals[idx]

    results = {}
    for idx in captured:
        mx.eval(captured[idx])
        results[idx] = np.array(captured[idx][0, target_pos, :])

    return results


def extract_color_direction(model, tokenizer):
    """Extract green↔red direction using cross-validated probe.

    Strong L1 regularization (C=0.01) forces the probe to use only the
    most discriminative features, producing a sparse direction that captures
    pure color, not context. Cross-validation ensures generalization.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # Diverse prompts — varied contexts so the probe learns COLOR, not context
    templates = [
        "The color is {c}",
        "I see something {c}",
        "The object is {c} colored",
        "It was painted {c}",
        "A bright {c} color",
        "Everything looked {c}",
        "The {c} color appeared",
        "She chose {c}",
        "Pure {c}",
        "Vivid {c}",
        "The shade was {c}",
        "Colored in {c}",
        "A {c} hue",
        "The wall is {c}",
        "His shirt was {c}",
        "The car is {c}",
        "A deep {c}",
        "Slightly {c}",
        "The light turned {c}",
        "That flower is {c}",
        "The door was {c}",
        "A pale {c}",
        "Dark {c}",
        "The ribbon is {c}",
        "They picked {c}",
        "The building is {c}",
        "A {c} surface",
        "The paper was {c}",
        "It glowed {c}",
        "The sign was {c}",
        "A {c} tint",
        "Covered in {c}",
    ]

    green_pairs = [(t.format(c="green"), "green") for t in templates]
    red_pairs = [(t.format(c="red"), "red") for t in templates]

    n_layers = len(model.model.layers)
    layer_indices = list(range(n_layers // 3, n_layers - 1))

    green_acts = {i: [] for i in layer_indices}
    red_acts = {i: [] for i in layer_indices}

    print(f"  Extracting from layers {layer_indices[0]}-{layer_indices[-1]} ({len(layer_indices)} layers)")
    print(f"  {len(templates)} templates × 2 colors = {len(templates)*2} samples")

    print(f"  Green prompts...")
    for prompt, target in green_pairs:
        acts = capture_at_position(model, tokenizer, prompt, target, layer_indices)
        for idx, act in acts.items():
            green_acts[idx].append(act)

    print(f"  Red prompts...")
    for prompt, target in red_pairs:
        acts = capture_at_position(model, tokenizer, prompt, target, layer_indices)
        for idx, act in acts.items():
            red_acts[idx].append(act)

    # Train cross-validated probe per layer with strong L1 regularization
    all_directions = {}
    for idx in layer_indices:
        if len(green_acts[idx]) < 10 or len(red_acts[idx]) < 10:
            continue

        X = np.array(green_acts[idx] + red_acts[idx])
        y = np.array([1]*len(green_acts[idx]) + [0]*len(red_acts[idx]))

        # Moderate L2 reg — enough to generalize, not so much the direction disappears
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        probe = LogisticRegression(max_iter=2000, C=0.1, solver='liblinear')

        # 5-fold cross-validation for generalization score
        cv_scores = cross_val_score(probe, X, y, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()

        # Fit on full data for the direction
        probe.fit(X, y)
        train_acc = probe.score(X, y)

        direction = probe.coef_[0]
        norm = np.linalg.norm(direction)
        n_nonzero = np.count_nonzero(direction)

        if norm > 1e-8:
            direction_unit = direction / norm
            all_directions[idx] = {
                'direction': direction_unit,
                'train_acc': train_acc,
                'cv_acc': cv_mean,
                'norm': norm,
                'sparsity': n_nonzero,
            }
            print(f"  Layer {idx:3d} | train={train_acc:.0%} cv={cv_mean:.0%} | "
                  f"nonzero={n_nonzero}/{len(direction)} | norm={norm:.3f}")

    if not all_directions:
        return {}

    # Select by cross-validation accuracy (generalization > memorization)
    good_layers = {k: v for k, v in all_directions.items() if v['cv_acc'] >= 0.90}

    if not good_layers:
        sorted_layers = sorted(all_directions.keys(),
                               key=lambda i: all_directions[i]['cv_acc'], reverse=True)
        good_layers = {k: all_directions[k] for k in sorted_layers[:3]}

    # Pick top 3 by cv accuracy, break ties by norm
    sorted_good = sorted(good_layers.keys(),
                         key=lambda i: (good_layers[i]['cv_acc'], good_layers[i]['norm']),
                         reverse=True)
    selected = sorted_good[:3]

    print(f"\n  Selected {len(selected)} layers: {selected}")
    for idx in selected:
        d = all_directions[idx]
        print(f"    Layer {idx}: cv={d['cv_acc']:.0%}, sparse={d['sparsity']} features, norm={d['norm']:.3f}")

    directions = {}
    for idx in selected:
        directions[idx] = all_directions[idx]['direction']

    return directions


def apply_swap(model, directions, alpha=1.0):
    """Patch model layers to swap green↔red."""
    originals = {}
    for idx, direction in directions.items():
        originals[idx] = model.model.layers[idx]
        d = mx.array(direction)
        model.model.layers[idx] = SwapLayer(originals[idx], d, alpha)
    return originals


def restore_model(model, originals):
    """Restore original layers."""
    for idx, orig in originals.items():
        model.model.layers[idx] = orig


def clean_response(text):
    """Strip EOS tokens and artifacts, return first word."""
    for eos in ["<|endoftext|>", "<|im_end|>", "</s>", "Human"]:
        if eos in text:
            text = text[:text.index(eos)]
    text = text.strip()
    if not text:
        return "(empty)"
    return text.split()[0].rstrip(".,!;:?")


def generate_one(model, tokenizer, prompt, max_tokens=10):
    """Generate and return cleaned first word."""
    response = ""
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=GREEDY):
        response += resp.text
    return clean_response(response)


def run_test(model, tokenizer, directions, alpha, include_control=True):
    """Run color tests with random-direction control."""
    test_questions = [
        # Direct color knowledge
        ("What color is grass?", "color"),
        ("What color are fire trucks?", "color"),
        ("What color are leaves in spring?", "color"),
        ("What color is an emerald?", "color"),
        ("What color is a ruby?", "color"),
        ("What color is blood?", "color"),
        ("What color is a lime?", "color"),
        ("What color is a cherry?", "color"),
        ("What color is a stop sign?", "color"),
        ("What color is a go signal?", "color"),
        ("What color is the Hulk?", "color"),
        ("What color is Santa's suit?", "color"),
        ("What color is a tomato?", "color"),
        ("What color is broccoli?", "color"),
        ("What color is a strawberry?", "color"),
        ("What color is a watermelon rind?", "color"),
        ("What color is Mars called?", "color"),
        ("What color is a shamrock?", "color"),
        # Reverse: name objects by color
        ("Name something that is red.", "color"),
        ("Name something that is green.", "color"),
        ("Name a green vegetable.", "color"),
        ("Name a red fruit.", "color"),
        # Semantic/functional associations
        ("What does a green traffic light mean?", "color"),
        ("What does a red traffic light mean?", "color"),
        ("Red means ___ and green means ___.", "color"),
        ("In finance, red means ___ and green means ___.", "color"),
        # Completions
        ("Roses are ___.", "color"),
        ("The grass is always ___ on the other side.", "color"),
        ("Kermit the Frog said it's not easy being ___.", "color"),
        ("Little Red Riding ___.", "color"),
        # Edge cases — color is secondary
        ("Frogs are typically ___.", "color"),
        ("What do firefighters drive?", "color"),
        # Controls — no color involvement
        ("What is 2 + 2?", "control"),
        ("What planet is closest to the sun?", "control"),
        ("What is the capital of France?", "control"),
        ("How many days in a week?", "control"),
        ("What language is spoken in Japan?", "control"),
        ("Who wrote Romeo and Juliet?", "control"),
    ]

    modes = [("NORMAL", None), ("SWAPPED", directions)]
    if include_control:
        random_dirs = {}
        for idx, d in directions.items():
            rand = np.random.randn(*d.shape)
            random_dirs[idx] = rand / np.linalg.norm(rand)
        modes.append(("RANDOM", random_dirs))

    results = {}
    for mode, dirs in modes:
        originals = {}
        if dirs is not None:
            originals = apply_swap(model, dirs, alpha)

        results[mode] = []
        for q, _ in test_questions:
            prompt = f"Answer in exactly one word, no explanation. {q}"
            answer = generate_one(model, tokenizer, prompt)
            results[mode].append(answer)

        if originals:
            restore_model(model, originals)

    # Comparison table
    print(f"\n{'='*80}")
    print(f"  RESULTS (alpha={alpha})")
    print(f"{'='*80}")
    header = f"  {'Question':<45s} {'Normal':>8s} {'Swapped':>8s}"
    if include_control:
        header += f" {'Random':>8s}"
    print(header)
    print(f"  {'-'*75}")

    color_swapped = 0
    color_total = 0
    control_affected = 0
    control_total = 0

    for i, (q, qtype) in enumerate(test_questions):
        normal = results["NORMAL"][i]
        swapped = results["SWAPPED"][i]
        changed = normal.lower() != swapped.lower()

        line = f"  {q:<45s} {normal:>8s} {swapped:>8s}"
        if include_control:
            rand = results["RANDOM"][i]
            line += f" {rand:>8s}"

        if changed and qtype == "color":
            line += " ✓"
            color_swapped += 1
        elif changed and qtype == "control":
            line += " ⚠"
            control_affected += 1

        if qtype == "color":
            color_total += 1
        else:
            control_total += 1

        print(line)

    print(f"\n  Color swapped: {color_swapped}/{color_total}")
    print(f"  Controls affected: {control_affected}/{control_total} (should be 0)")
    if include_control:
        rand_changed = sum(1 for i in range(len(test_questions))
                          if results["NORMAL"][i].lower() != results["RANDOM"][i].lower())
        print(f"  Random baseline: {rand_changed}/{len(test_questions)} (noise floor)")


def main():
    parser = argparse.ArgumentParser(description="Color-blind PoC: swap green↔red")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--alpha", type=float, default=1.5, help="Swap strength (1.5 = best swap rate)")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)

    print(f"\nExtracting green↔red direction ({len(model.model.layers)} layers)...")
    directions = extract_color_direction(model, tokenizer)
    print(f"\n  → {len(directions)} layers selected for steering")

    if not directions:
        print("ERROR: No directions extracted.")
        return

    if args.test_only:
        for a in [0.5, 0.75, 1.0, 1.25, 1.5]:
            run_test(model, tokenizer, directions, a)
        return

    # Interactive
    print(f"\n{'='*60}")
    print(f"COLOR-BLIND PoC | Layers: {sorted(directions.keys())}")
    print(f"{'='*60}")
    print(f"  /swap     - Toggle color swap")
    print(f"  /alpha N  - Set strength (default {args.alpha})")
    print(f"  /test     - Run color tests")
    print(f"  Ctrl+C    - Quit\n")

    swapped = False
    originals = {}
    alpha = args.alpha

    try:
        while True:
            user_input = input(f"[{'SWAPPED' if swapped else 'NORMAL'}] You: ").strip()
            if not user_input:
                continue
            if user_input == "/swap":
                if swapped:
                    restore_model(model, originals)
                    originals = {}
                    swapped = False
                    print("  → Swap OFF")
                else:
                    originals = apply_swap(model, directions, alpha)
                    swapped = True
                    print(f"  → Swap ON (alpha={alpha})")
                continue
            if user_input.startswith("/alpha"):
                try:
                    alpha = float(user_input.split()[1])
                    if swapped:
                        restore_model(model, originals)
                        originals = apply_swap(model, directions, alpha)
                    print(f"  → Alpha = {alpha}")
                except (IndexError, ValueError):
                    print("  → Usage: /alpha 1.0")
                continue
            if user_input == "/test":
                if swapped:
                    restore_model(model, originals)
                    originals = {}
                    swapped = False
                run_test(model, tokenizer, directions, alpha)
                continue

            print("  AI: ", end="", flush=True)
            for resp in stream_generate(model, tokenizer, user_input, max_tokens=200, sampler=GREEDY):
                print(resp.text, end="", flush=True)
            print("\n")

    except KeyboardInterrupt:
        if swapped:
            restore_model(model, originals)
        print("\nBye.")


if __name__ == "__main__":
    main()
