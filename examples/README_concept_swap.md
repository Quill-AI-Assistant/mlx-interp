---
tags: [workbench, research]
---
# Concept Swap Vignette

An educational demonstration of activation steering using Householder reflections. This vignette applies established techniques from the interpretability literature to swap concept pairs in small language models — no retraining or fine-tuning required.

**This is not novel research.** The techniques used here are well-documented:

- **Turner et al. (2023)** — "Activation Addition: Steering Language Models Without Optimization" — demonstrated concept steering via activation vectors
- **Zou et al. (2023)** — "Representation Engineering: A Top-Down Approach to AI Transparency" — systematically mapped which concepts have linear representations
- **Park et al. (2023)** — "The Linear Representation Hypothesis and the Geometry of Large Language Models" — studied when concepts are linearly separable
- **Elhage et al. (2022)** — "Toy Models of Superposition" (Anthropic) — explains why concrete features are more separable than abstract ones

The value of this vignette is practical: a clean, runnable implementation with an interactive web UI for experimentation and learning. Some concepts resist steering more than others — this is a known property of representation geometry, not a new finding.

## How it works

1. **Contrastive probe extraction** -- Run ~32 diverse prompts per concept through the model (e.g. "The wall is green" vs "The wall is red") and capture activations at each layer.
2. **Logistic regression with L2 regularization** (C=0.1) -- Train a probe per layer to classify concept A vs B. The probe's weight vector becomes the swap direction.
3. **5-fold cross-validation** -- Score each layer's direction for generalization. Select the top 3 layers by CV accuracy (tie-break by norm).
4. **Householder reflection** -- During generation, at each selected layer, project the activation onto the direction and subtract 2x the projection (scaled by alpha). This mirrors the activation across the concept hyperplane.

### L1 vs L2 trade-off

An important finding: L1 regularization (sparse probes) produces interpretable directions that identify specific features, but the resulting vectors are too sparse to actually steer model behavior. L2 regularization produces dense directions that effectively steer outputs. The current implementation uses L2 (C=0.1) for steering capability.

## Verified results

Model: `gemma-2-2b-it-4bit` | 7 concept pairs | ~1400 total generations | Gemini re-review: 9/10 rigor rating (up from 3/10 after evaluation framework improvements)

### Summary table

| Pair | Swap Rate | Confidence | Controls | PPL Ratio | Probe Verification | Random Baseline | Reverse | Alpha |
|---|---|---|---|---|---|---|---|---|
| green/red | 81% | 0.66 | 75% | 4.4x | 0.69 | 8% | 76% | 1.5 |
| blue/yellow | 60% | 0.55 | 100% | 0.8x | 0.71 | 10% | 54% | 1.5 |
| hot/cold | 56% | 0.70 | 100% | 1.6x | 0.91 | 12% | 46% | 1.0 |
| fast/slow | 91% | 0.57 | 88% | 1.7x | 0.45 | 7% | 74% | 2.0 |
| sweet/bitter | 87% | 0.60 | 100% | 0.9x | 0.60 | 7% | 68% | 1.5 |
| happy/sad | 61% | 0.64 | 100% | 1.1x | 0.91 | 7% | 48% | 1.0 |
| big/small | 87% | 0.60 | 75% | 1.8x | 0.66 | 7% | 74% | 1.5 |

**Column definitions:**
- **Swap Rate** — percentage of concept-relevant prompts where the model's output flipped to the opposite concept
- **Confidence** — average logit-based confidence on swapped outputs (0–1)
- **Controls** — percentage of control prompts (e.g. "What is 2+2?") that were unaffected by steering
- **PPL Ratio** — perplexity of steered output / perplexity of normal output (1.0x = no degradation)
- **Probe Verification** — accuracy of independent probe classifying steered outputs as the target concept
- **Random Baseline** — swap rate when using a random direction instead of the extracted one
- **Reverse** — swap rate in the opposite direction (B→A instead of A→B)
- **Alpha** — steering strength used for that pair

### Key findings

1. **Random baselines (4–12%) confirm directions are real.** Every pair's extracted direction dramatically outperforms a random vector of the same norm, ruling out artifacts from magnitude-only effects.

2. **The concrete-vs-abstract claim does not hold cleanly.** fast/slow (91%) and sweet/bitter (87%) both beat green/red (81%). big/small (87%) ties sweet/bitter. The old framing that colors swap best and abstract concepts resist is wrong.

3. **The real predictor is representational binariness.** Pairs where the model maintains a clean binary representation (fast vs slow, big vs small) swap reliably regardless of whether the concept is "concrete" or "abstract." Pairs where the model's representation is more graded or context-dependent (hot/cold at 56%, happy/sad at 61%) resist more.

4. **Perplexity impact varies dramatically.** sweet/bitter (0.9x) and blue/yellow (0.8x) have essentially no quality degradation — the steered text reads as fluently as normal output. green/red (4.4x) has substantial degradation, suggesting the green/red direction is more entangled with general language modeling features.

5. **Reverse directions work but with asymmetry.** Every pair shows positive reverse swap rates (46–76%), confirming the directions are bidirectional. The asymmetry (e.g. fast/slow forward 91% vs reverse 74%) likely reflects uneven concept salience in the training data.

6. **Probe verification catches false positives.** hot/cold has high probe verification (0.91) despite moderate swap rate (56%), meaning when it does swap, it swaps convincingly. fast/slow has low probe verification (0.45) despite high swap rate (91%), suggesting some swaps land in the right neighborhood but not on the exact target token.

### Concept resistance spectrum (revised)

The original hypothesis — that concrete concepts swap cleanly and abstract concepts resist — does not survive contact with the full 7-pair evaluation. The actual pattern, consistent with the representation engineering literature (Zou et al. 2023, Park et al. 2023) and superposition hypothesis (Elhage et al. 2022), is more nuanced:

- **Clean binary representations** swap best — fast/slow (91%), sweet/bitter (87%), big/small (87%). These are pairs where the model maintains a sharp either/or boundary.
- **Context-dependent representations** resist more — hot/cold (56%), happy/sad (61%). Temperature and emotion exist on continuums in the training data, and the model's representation reflects that gradedness.
- **Colors are mixed** — green/red (81%) swaps well but with high perplexity cost; blue/yellow (60%) resists and lands on "golden" instead of "yellow," suggesting the blue/yellow boundary is less cleanly represented than green/red.

The Householder reflection flips one direction. If a concept is distributed across many entangled directions or exists on a continuum rather than a binary, a single-direction flip only partially moves the representation. The key insight is that "concrete vs abstract" is a poor proxy — what actually matters is whether the model learned a binary or graded representation for that particular concept pair.

## Limitations observed

- Some questions resist swapping even at alpha=1.5 (e.g. "What color is broccoli?" stayed Green, "What color is the Hulk?" stayed Green-ish). These tend to be cases where the concept is deeply entangled with other features (Hulk = character identity, not just color).
- Too-high alpha (>2.0) degrades into gibberish. Too-low alpha (<0.5) has no visible effect. The sweet spot is 1.0-1.5 for most concept pairs.
- The blue/yellow swap produces "Golden/Gold" instead of "Yellow" -- the reflection lands in the right conceptual neighborhood but not always on the exact target word.

## Usage

### CLI

```bash
python examples/concept_swap_vignette.py
python examples/concept_swap_vignette.py --alpha 1.5 --test-only
```

### Web UI

```bash
python -m mlux.tools.concept_swap_explorer
# or launch from the command center:
python -m mlux
```

The web UI (default port 5006) lets you:
- Extract directions for any concept pair via the UI
- Test prompts with side-by-side normal vs swapped output
- Adjust alpha with a slider
- Run the full built-in test suite (32 questions + 6 controls)

### API (when server is running)

```bash
# Extract a direction
curl -X POST http://localhost:5006/extract_direction \
  -H "Content-Type: application/json" \
  -d '{"concept_a": "blue", "concept_b": "yellow"}'

# Generate with swap
curl -X POST http://localhost:5006/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What color is the sky?", "alpha": 1.5, "swap_enabled": true, "max_tokens": 5}'
```

## Related
- [[wiki/experiments/mlx-interpretability]]
