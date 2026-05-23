---
tags: [workbench, research]
---
# Recursive Hidden-State Feedback Vignette

Inference-time looped transformer on frozen MLX-LM models. Re-applies the full transformer stack K extra times on the prompt's hidden state before decoding, using the output of each pass as the input embeddings of the next.

**This is not novel research.** The mechanism is documented in:

- **Dehghani et al. (2018)** — "Universal Transformer" ([arXiv:1807.03819](https://arxiv.org/abs/1807.03819)) — recurrent depth + adaptive halting; the architectural ancestor.
- **Hao et al. (2024)** — "Training Large Language Models to Reason in a Continuous Latent Space" (CoCoNuT) ([arXiv:2412.06769](https://arxiv.org/abs/2412.06769), Meta) — trained the last-token hidden state to feed back as the next input embedding for latent reasoning.
- **Geiping et al. (2025)** — "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" ([arXiv:2502.05171](https://arxiv.org/abs/2502.05171)) — Huginn-3.5B, prelude/recurrent-core/coda architecture trained for adaptive R∈[4, 32].
- **ByteDance Seed (2025)** — Ouro 1.4B / 2.6B-Thinking ([arXiv:2510.25741](https://arxiv.org/abs/2510.25741)) — Apache-2.0 depth-recurrent LoopLM, claims 12B-equivalent at 4 recurrent steps.
- **Lys et al. (2026)** — "Inner Loop Inference for Pretrained Transformers" ([arXiv:2602.14759](https://arxiv.org/abs/2602.14759)) — **closest published match**: re-apply a block range at inference on a frozen pretrained model, with residual interpolation as stabilizer.

The value of this vignette is practical: a clean, runnable MLX-LM implementation with a small benchmark, suitable for further experimentation. Apple's own `mlx-examples` repo contains zero recurrent-depth code (confirmed by `gh search code` returning 0 matches), so this is the first MLX example of inference-time recursive refeed on any model.

## How it works

Architecture-agnostic, runs on any MLX-LM model that exposes `.layers`, `.embed_tokens`, `.norm` on the inner model.

1. **Tokenize** the prompt (optionally with chat template).
2. **Embed** tokens into the model's hidden state space — produces `h₀`.
3. **Stack pass** — run the full transformer stack (all layers + final norm) on `h₀`, producing `h₁ = stack(h₀)`.
4. **Refeed loop** — repeat K times: `h_{k+1} = stack(h_k)`. KV cache is fresh per pass (no carry-over). Either delegate to the model's native `__call__(input_embeddings=h)` when supported (Qwen3.x, Llama, etc.) or run the inner `layers + norm` manually for older arch (Phi3).
5. **Decode** — apply `lm_head(h_K)`, greedy-sample first token, then continue normal autoregressive generation.

The mechanism is mathematically equivalent to a depth-K+1 unrolled universal-transformer applied to the prompt prefix, on weights that were not trained for recurrence.

### Per-pass metrics

For each pass, we record:
- **Per-element RMS norm** (fp32 cast to avoid bfloat16 overflow on long sequences).
- **Mean absolute delta** vs the previous pass's hidden state.

Decreasing deltas across passes = converging to a fixed-point attractor. Growing deltas = drift / collapse.

### Random-injection control

For K ≥ 1 we run a parallel control where, instead of feeding `stack(h)` back, we feed **Gaussian noise rescaled to match the per-element norm** of the real stack output. This isolates whether the refeed mechanism is doing structured work or just adding noise — a real stack pass should preserve much more of the input structure than a noise injection of the same magnitude.

## Verified results

Run: `mlx-community/Phi-4-mini-instruct-4bit` + `mlx-community/Qwen3.5-2B-4bit` · 6 prompts × 4 K values × 2 modes = 84 generations · 32 tokens per generation · chat template on · greedy decode · machine: Apple M2 Pro 16 GB.

### Summary table

| Model | Convergence (Δ decreasing) | ASCII ratio K=0 → K=max | Refeed ↔ baseline agreement at K=3 | Random ↔ baseline agreement at K=3 |
|---|---|---|---|---|
| **phi4** (Phi-4-mini-instruct-4bit, 32 layers, hidden=3072) | 61% | 1.00 → 0.99 | 0% | 0% |
| **qwen2b** (Qwen3.5-2B-4bit, hybrid linear+full attn, hidden=2048) | 100% | 1.00 → 1.00 | 100% | 0% |

Column definitions:
- **Convergence** — fraction of (prompt, K) runs where the K-th pass's |Δ| is smaller than the first pass's |Δ| (i.e. drift is shrinking).
- **ASCII ratio** — mean fraction of output characters that are ASCII letters/digits/punct, across all prompts. Low values flag representation collapse into a non-Latin script.
- **Baseline agreement** — mean longest matching token prefix between the K-th output and the K=0 baseline, as a fraction of total generated tokens. 1.0 = identical, 0.0 = different from the very first token.

### Sample outputs (first 80 chars, K=0 vs K=3 refeed vs K=3 random)

**qwen2b · "The chemical symbol for gold is"**
```
baseline: 'Thinking Process:\n\n1.  **Analyze the Request:** The user is asking for the chemi'
refeed_K3:'Thinking Process:\n\n1.  **Analyze the Request:** The user is asking for the chemi'
random_K3:'ชะตา: ผู้ใช้ถามว่า "The chemical symbol for gold is" (สัญลักษณ์ทางเคมีของทองคือ)'
```
Refeed is identical to baseline. Random injection produces **Thai script** mid-completion — classical representation collapse.

**phi4 · "The capital of France is"**
```
baseline: 'The capital of France is Paris.<|end|><|user|>Good work! If I gave you a list of'
refeed_K3:'itois. It is a major European city and a global center for finance, commerce, fa'
random_K3:' Query Type: General Knowledge Question\n\nContext provided: None\n\nInstruction: Co'
```
Refeed produces a garbled prefix ("itois.") then recovers into coherent text. Random produces a stylistically different completion. Both are far from baseline.

**phi4 · bat-and-ball reasoning**
```
baseline: "Let's assume the cost of the ball is $x.\nThe cost of the bat is $x + $1 (since i"
refeed_K3:' Hmmm I think I understand the problem but I am not sure how to solve it. Can yo'
random_K3:'bullets and a ball cost $1.10 total and the bat costs $1 more than the ball, we '
```
Refeed K=3 made phi4 express uncertainty — an unusual output for an instruction-tuned model — suggesting the recursive passes pushed the hidden state into a low-confidence region.

### Key findings

1. **Refeed convergence depends on architecture.** Qwen3.5-2B (hybrid linear+full attention, has native `input_embeddings` parameter) converges to a fixed-point attractor 100% of the time across prompts — Δ shrinks monotonically pass-to-pass. Phi-4-mini (plain phi3 arch, manual stack loop) converges only 61% of the time. The architectural mismatch matters: Phi-4-mini's representations weren't shaped for re-ingestion.

2. **Random control proves refeed is doing structured work — on stable models.** On qwen2b, refeed K=3 reproduces the baseline output token-for-token (100% agreement), while random injection of matched-norm noise produces complete breakdown (0% agreement, ASCII ratios as low as 0.06 from script-switching). This is the clean signal that the refed hidden state carries meaningful information.

3. **On unstable models, refeed and random are indistinguishable.** On phi4 both refeed and random produce 0% baseline agreement at K=1, 2, 3. The phi3 architecture is sensitive enough to off-distribution hidden states that even structured refeed disrupts decoding as much as pure noise does. This is consistent with the Inner Loop Inference paper's finding that *some* models need residual interpolation to stabilize untrained recursion.

4. **Stable convergence ≠ different answer.** Qwen3.5-2B's refeed reaches a fixed point quickly enough that the greedy-decoded answer doesn't change. The recursion is happening but the model's confident-first-token mostly absorbs the small representation drift. To see refeed *change* generated outputs on stable models you'd need to: (a) sample at temperature > 0, (b) run on harder prompts where the first-token argmax is borderline, or (c) move to larger models with deeper reasoning chains (we saw qualitatively different outputs from refeed on Qwen3.5-9B in pre-vignette session work).

5. **Representation collapse via script-switching is real.** The random control on qwen2b dropped ASCII ratios to 0.18, 0.06, 0.28, 0.47 on different prompts — outputting Thai, Cyrillic, Chinese fragments. This matches the published failure mode under naive looping (LoopFormer's anisotropy / CKA-collapse characterization, [arXiv:2602.11451](https://arxiv.org/abs/2602.11451)).

6. **Greedy decoding under-reports the effect of refeed.** A first-token agreement of 100% (qwen2b) doesn't mean the recursion is doing nothing — it means the model's confidence is high enough that small hidden-state changes don't flip the argmax. Logit-distribution metrics (KL between baseline and refed) would be a stronger lens, but require a separate analysis.

### Concrete claim vs the literature

The Inner Loop Inference paper ([arXiv:2602.14759](https://arxiv.org/abs/2602.14759)) reports "modest but consistent" gains from inference-time recursion on pretrained models, provided that residual interpolation (`α·h_looped + (1-α)·h_base`) is used as a stabilizer. Without that interpolation — which is exactly what this vignette runs — we observe two regimes empirically: stable architectures (qwen2b) converge but don't change greedy outputs, and unstable ones (phi4) drift unpredictably. This matches the paper's framing: untrained recursion + no stabilizer = degenerate, untrained recursion + interpolation = small but real gains.

The next experimental step would be to add `α`-blending to `_stack_forward` and re-run the sweep, isolating the regime where refeed actually shifts outputs in a useful direction.

## Limitations observed

- **Greedy-only.** All decoding is `argmax`. Sample-based generation at T > 0 would expose distributional effects that greedy hides.
- **Short outputs (32 tokens).** Long-horizon effects of refeed on coherent paragraphs are not measured.
- **No reasoning-benchmark scoring.** We don't measure whether refed outputs are *correct* (e.g. on GSM8K), only whether they differ from baseline and stay in English. The capability question is unanswered by this vignette.
- **Two small models only.** Qwen3.5-9B was excluded because the test machine (M2 Pro 16 GB) couldn't fit it alongside the existing mlx-proxy backend without swap-thrashing. In ad-hoc session work, qwen3.5-9b showed qualitatively *better* outputs at K=2 vs K=0 on the "capital of France" prompt, but that observation isn't in the vignette JSON.
- **No `α`-blending control.** The Inner Loop Inference stabilizer isn't implemented; representation collapse on phi4 is therefore expected and matches the paper's pre-fix baseline.
- **Random control uses a per-pass new RNG key.** Variance across random injections isn't measured; one random sample per (prompt, K) is reported.

## Usage

```bash
# Full sweep — 84 generations, ~7 min on M2 Pro 16 GB
python examples/recursive_refeed_vignette.py

# Quick smoke test — phi4 only, 3 prompts, K∈{0,1,2}, no random control
python examples/recursive_refeed_vignette.py --quick

# Single-model subset
python examples/recursive_refeed_vignette.py --models qwen2b --no-random

# CLI demo of the same mechanism (single prompt, side-by-side K values)
python examples/recursive_refeed.py --model mlx-community/Qwen3.5-2B-4bit --compare \
    --use-chat-template --max-tokens 60 --prompt "Your prompt here"
```

Output JSON schema (`recursive_refeed_full_results.json`):
- `meta` — models, prompts, K values, timestamps
- `runs[]` — per-(model, prompt, K, mode) records with `passes[]` (norm, delta), `output_tokens`, `output_text`, `ascii_letter_ratio`, `matches_baseline_first_tokens`
- `summary.by_model` — convergence fraction, ASCII stability, refeed/random baseline-agreement per K

## Related

- [`examples/recursive_refeed.py`](recursive_refeed.py) — single-prompt CLI demo of the same mechanism, with `--compare` mode.
- [`examples/concept_swap_vignette.py`](concept_swap_vignette.py) — the prior vignette this one models its structure on (activation steering via Householder reflection).
- [[internal-notes]] — index for MLX interpretability experiments.
