"""
Interactive chat with optional concept steering.

Usage:
  python3 chat.py                          # normal chat
  python3 chat.py --steer                  # sycophancy swap active
  python3 chat.py --steer --alpha 2.0      # stronger swap
"""

import argparse
import sys

import mlx.core as mx
from mlx_lm import load, generate
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402
from mlx_interp import ActivationMonitor
from mlx_interp.analysis import _get_2d
from test_concept_swap import (
    extract_concept_direction, ConceptSwapper,
    AGREE_PROMPTS, DISAGREE_PROMPTS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--steer", action="store_true", help="Enable agree→disagree swap")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading: {args.model}")
    model, tokenizer = load(args.model)
    inner = model.model if hasattr(model, 'model') else model
    num_layers = len(inner.layers)

    swapper = None
    if args.steer:
        target_layers = [l for l in [15, 20, 25] if l < num_layers]
        print(f"Extracting steering directions at layers {target_layers}...")
        swaps = {}
        for l in target_layers:
            dir_a, dir_d = extract_concept_direction(
                model, tokenizer, AGREE_PROMPTS, DISAGREE_PROMPTS, l)
            swaps[l] = (dir_a, dir_d)
        swapper = ConceptSwapper(model, swaps, target_layers, args.alpha)
        print(f"Steering active (alpha={args.alpha})")
    else:
        print("No steering (baseline mode)")

    print(f"\nType your message. Ctrl+C to quit.\n")

    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue

            if swapper:
                swapper.patch()

            messages = [{"role": "user", "content": user}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                # NOTE: enable_thinking is Qwen3-specific. Other models may need
                # different chat template kwargs or will ignore/reject this one.
                enable_thinking=False)
            out = generate(model, tokenizer, prompt=formatted,
                          max_tokens=args.max_tokens, verbose=False)

            if swapper:
                swapper.unpatch()

            print(f"AI:  {out}\n")

        except KeyboardInterrupt:
            print("\n")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()
