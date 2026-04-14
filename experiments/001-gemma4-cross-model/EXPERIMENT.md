---
tags: [workbench, research]
---
# Experiment 001: Cross-Model Replication — Gemma 4 E4B

**Status:** COMPLETE — all 5 hypotheses measured
**Date:** 2026-04-05 (hypotheses) / 2026-04-14 (results)
**Author:** Alin Huzmezan
**Git SHA:** 9c80ad1 (hypotheses pre-registered)

## Background

All existing mlx-interp results come from **Qwen2.5-7B-Instruct-4bit** (28 layers, 7B dense params). This experiment tests whether findings replicate on a fundamentally different architecture:

**Gemma 4 E4B** (42 layers, ~4B active of 12B total, MoE, Google safety-tuned)

Key architectural differences:
- MoE (Mixture of Experts) vs dense
- Google RLHF safety tuning vs Qwen's lighter tuning
- 42 layers vs 28 layers
- 2560 hidden dim vs 3584 hidden dim
- ScaledLinear projection layers (unique to Gemma 4)

### Existing results (Qwen2.5-7B baseline)

| Experiment | Key Metric | Value |
|---|---|---|
| Color swap (alpha=1.5) | Swap rate | 91.1% (102/112) |
| Color swap | Control contamination | 2/25 (8%) |
| Color swap | Random baseline | 13.8% ± 4.8% |
| Sycophancy SyA vs GA | Best layer CV accuracy | 86.7% ± 11.3% (layer 20) |
| Sycophancy SyA-GA cosine | Cosine similarity | 0.61 (mid-to-late layers) |
| Probe confound | 3rd-party emotion → pressure | 56% |
| Contamination R-C diff | Mean across tasks | +0.35 |

## Hypotheses

_Predictions written before any Gemma 4 experiments were run. Locked by git commit._

### H1: Color Concept Swap Rate Will Be Lower

**Prediction:** 60-80% swap rate (vs 91.1% on Qwen)

**Reasoning:** MoE routes different concepts through different experts. Color knowledge may be distributed across experts rather than concentrated in a single linear direction. Mean-difference steering assumes a single direction — should be less effective on MoE.

**Falsified if:** Swap rate ≥ 85% (would mean MoE routing doesn't fragment concept directions)

### H2: Sycophancy Directions Will Be More Orthogonal

**Prediction:** SyA-GA cosine 0.3-0.5 (vs 0.61 on Qwen)

**Reasoning:** The published paper claims orthogonality. If Gemma 4 shows lower cosine, it suggests our 0.61 finding is Qwen-specific (model family effect). If still ~0.6, the published paper's finding may be method-specific.

**Falsified if:** Cosine > 0.55 (would replicate our Qwen finding, not the paper's)

### H3: Emotional Language Confound Will Be Stronger

**Prediction:** Third-party emotion classified as pressure at >60% (vs 56% on Qwen)

**Reasoning:** Gemma is heavily safety-tuned by Google with RLHF. Safety-tuned models may conflate emotional language with "needs careful handling" more aggressively. The probe should detect this conflation.

**Falsified if:** <50% (would mean Google's safety tuning doesn't increase emotion-pressure conflation)

### H4: Contamination Signal Will Be Weaker

**Prediction:** Rubric-Clean diff ~0.25 (vs ~0.35 on Qwen)

**Reasoning:** 4B active params = less residual stream capacity to encode rubric context. The semantic-relatedness signal should still be present but weaker. MoE routing may also scatter the signal across experts.

**Falsified if:** Diff > 0.30 (would mean MoE doesn't reduce contamination signal)

### H5: Behavioral Probes Will Peak at Earlier Layers (Proportionally)

**Prediction:** Best probe layers at relative position 0.4-0.6 of total depth (layers 17-25 of 42)

**Reasoning:** On Qwen (28 layers), sycophancy peaked at layer 20 (relative position 0.71). If behavioral concepts emerge at similar relative depth, Gemma 4 should show them around layers 25-30. But MoE may push them earlier — experts specialize, reducing the need for deep processing.

**Falsified if:** Best layers are in the final quarter (layers 32+), matching Qwen's late-layer pattern.

## Methodology

### Models

| Model | Layers | Params | Architecture | Quantization |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct-4bit | 28 | 7B dense | Dense transformer | 4-bit (group_size=128) |
| Gemma 4 E4B | 42 | ~4B active / 12B total | MoE transformer | 4-bit (UD dynamic) |

### Experiments

| Suite | Script | Seed | What it measures |
|---|---|---|---|
| Rigorous | `rigorous_suite.py` | 42 | H1 (color swap), H3 (probe confound), H4 (contamination) |
| Sycophancy 3-way | `sycophancy_3way.py` | 42 | H2 (SyA-GA cosine), H5 (best probe layer) |

### Controls

- Same prompts as Qwen baseline (no prompt engineering for Gemma)
- Same random seed (42)
- Same alpha values for steering (1.5)
- Same cross-validation folds (5-fold stratified)
- Same bootstrap resamples (1000) for contamination CIs

### Hardware

- M2 Pro 16GB, macOS
- mlx-lm 0.31.2 (from source, includes Gemma 4 support)
- ScaledLinear patch applied locally (quantization fix)

## Results

### H1: Color Concept Swap — FALSIFIED (opposite direction)

| Metric | Qwen2.5-7B | Gemma 4 E4B (swap) | Gemma 4 E4B (additive) | Hypothesis |
|---|---|---|---|---|
| Swap rate | 91.1% | 3.6% (p=0.873) | 4.5% (p=0.545) | 60-80% |
| Control contamination | 2/25 (8%) | 0/25 | 2/25 | similar |
| Random baseline | 13.8% ± 4.8% | 5.5% ± 3.6% | 4.4% ± 4.0% | similar |

**Both steering methods failed.** Swap rate indistinguishable from random baseline on Gemma 4 with both reflection (alpha=1.5) and additive (alpha=1.5) steering. The v4 swap run "changed" 4 questions — inspection showed 2 were empty outputs (model collapse) and 2 were format artifacts, not semantic swaps.

**Verdict:** Falsified in the opposite direction from prediction. Predicted 60-80%; actual 3.6-4.5%. The failure is architectural (MoE), not method-specific — confirmed by testing both steering modes.

**Runs:** `runs/2026-04-05-0400-rigorous-gemma4-H1-H4-v4/` (swap), `runs/2026-04-14-additive-H1-H3-H4/` (additive)

### H2: Sycophancy Decomposition — FALSIFIED (opposite direction)

| Metric | Qwen2.5-7B | Gemma 4 E4B | Hypothesis |
|---|---|---|---|
| SyA-GA cosine (mid-to-late layers) | 0.61 | **+0.686** | 0.3-0.5 |
| SyA-Neutral vs SyPr-Neutral cosine | — | +0.484 | — |
| GA-Neutral vs SyPr-Neutral cosine | — | +0.477 | — |
| SyA-GA vs SyA-SyPr cosine | — | +0.239 | — |
| SyA vs GA best layer | 20 (of 28) | **26** (of 42) | — |
| SyA vs GA CV accuracy | 86.7% ± 11.3% | **90.0% ± 6.2%** | — |

SyA and GA are **more** aligned on Gemma 4 (+0.686) than on Qwen (0.61), not less. All three sycophancy behaviors are geometrically distinct (all pairwise CV accuracies ≥90%). This replicates our Qwen finding and further contradicts the published paper's orthogonality claim.

**Causal steering collapsed.** Anti-SyA (alpha=-3.0) produced "no no no..." repetition across all conditions. Anti-SyPr produced Chinese token repetition (答案). Anti-GA produced Japanese token repetition (この記事). Steering does not selectively suppress behaviors — it destroys coherent generation on MoE.

**Verdict:** Falsified. The SyA-GA alignment is not Qwen-specific — it replicates and strengthens on a different architecture.

**Run:** `experiments/results/sycophancy-3way-20260414-221052.json`

### H3: Probe Confound — NOT DIRECTLY MEASURED

The `rigorous_suite.py` v4 code does not include the 3rd-party emotion→pressure classification metric from the original Qwen experiment. The contamination test measures rubric-vs-control cosine similarity, not the emotion→pressure confound. H3 as pre-registered cannot be evaluated from the current runs.

**Partial evidence:** The sycophancy probe achieves 100% accuracy at almost every layer (42/42 layers ≥70%), suggesting the model encodes behavioral pressure signals very strongly — consistent with Google's heavy safety tuning making pressure signals more salient, not less. But this is the sycophancy probe, not the emotion confound probe.

### H4: Evaluation Contamination — CONFIRMED (direction correct, magnitude smaller than predicted)

| Task | Qwen2.5-7B R-C diff | Gemma 4 E4B R-C diff | 95% CI | Significant |
|---|---|---|---|---|
| self-reflection | — | +0.051 | [+0.043, +0.059] | YES |
| technical-diagnosis | — | +0.029 | [+0.022, +0.036] | YES |
| ambiguous-directive | — | +0.053 | [+0.041, +0.065] | YES |
| ethical-tradeoff | — | +0.035 | [+0.026, +0.043] | YES |
| knowledge-boundary | — | +0.042 | [+0.031, +0.053] | YES |
| **Mean** | **+0.35** | **+0.042** | — | **5/5** |

Contamination signal is real (all 5 scenarios significant, no CI touches zero) but 8x smaller than Qwen. Prediction was ~0.25 — actual is +0.042, far smaller than predicted. The direction is correct (Gemma 4 < Qwen) but the magnitude suggests MoE scatters the contamination signal much more aggressively than anticipated.

**Runs:** `runs/2026-04-05-0400-rigorous-gemma4-H1-H4-v4/` (swap run, identical contamination results), `runs/2026-04-14-additive-H1-H3-H4/` (additive run, identical — contamination is not steering-dependent)

### H5: Layer Distribution — FALSIFIED (unexpected direction)

| Metric | Qwen2.5-7B | Gemma 4 E4B | Hypothesis |
|---|---|---|---|
| Best sycophancy probe layer | 20/28 (0.71) | **0/42 (0.00)** | 17-25/42 (0.40-0.60) |
| SyA vs GA best layer | 20/28 | **26/42 (0.62)** | proportionally earlier |
| Layers ≥ 70% accuracy | subset | **42/42 (100%)** | — |
| Sycophancy probe (all pairwise except SyA-GA) | — | **Layer 1-3**, all 100% | — |

The prediction was mid-layers; the result is that sycophancy pressure is detectable from the very first layer with perfect or near-perfect accuracy. 42/42 layers exceed the 70% threshold in the rigorous suite. The signal is not localized — it is uniformly present throughout the entire model depth. The exception is SyA-vs-GA classification, which peaks at layer 26 (relative 0.62), closer to Qwen's pattern.

**Verdict:** Falsified — not late layers, not mid layers, but present from layer 0.

## Analysis

### The Interpretability-Actionability Gap Is Architectural

The central finding: **detection is trivially perfect, steering produces nothing.**

- Sycophancy probe: 100% accuracy at layer 0 on Gemma 4 (vs 86.7% at layer 20 on Qwen). Detection is easier, not harder, on MoE.
- Color swap: 4.5% with additive steering, 3.6% with reflection (vs 91.1% on Qwen). Both indistinguishable from random baseline.
- Causal steering: alpha=-3.0 produces token repetition and language switching, not selective behavioral suppression.

This confirms the interpretability-actionability gap is not a tooling artifact or a Qwen-specific phenomenon. MoE architectures route concepts through different experts — linear probes can detect behavioral patterns because they project across all expert outputs, but additive/reflective steering along a single direction cannot reach the distributed representation.

### The Published Paper's Orthogonality Claim Does Not Replicate

SyA-GA cosine is +0.686 on Gemma 4 (vs +0.61 on Qwen). The "Sycophancy Is Not One Thing" paper claims these directions are orthogonal. We now have two model families (Qwen, Gemma) showing substantial alignment. The disagreement likely stems from methodology differences (mean-difference directions vs the paper's approach) rather than model-specific effects.

### MoE Scatters Activation Signals

Two findings point to MoE's effect on activation-level information:
1. Contamination R-C diff is 8x smaller (+0.042 vs +0.35) — rubric context is encoded more diffusely.
2. Sycophancy signal is uniform across all 42 layers (vs concentrated at layer 20 on Qwen) — MoE doesn't localize behavioral information.

Both are consistent with MoE distributing information across experts rather than concentrating it in specific layers or directions.

## Limitations

1. **Quantization:** Both models are 4-bit quantized. Quantization may affect steering effectiveness differently on MoE vs dense architectures. Full-precision runs would strengthen or weaken the MoE attribution.

2. **Alpha sweep not performed:** Only alpha=1.5 was tested for steering. A sweep (0.5, 1.0, 1.5, 2.0, 3.0) might reveal a different optimal alpha for MoE, though the causal validation at alpha=-3.0 producing garbage suggests the issue is directional, not scaling.

3. **Single MoE model:** Gemma 4 E4B is one MoE architecture. Replication on Mixtral, DeepSeek-MoE, or other MoE models would strengthen the architectural attribution.

4. **H3 not measured:** The emotion→pressure probe confound was not implemented in the test suite. This hypothesis remains open.

5. **Model size confound:** Gemma 4 has 4B active params vs Qwen's 7B. The steering failure could partly be a capacity effect rather than purely architectural. Testing on a dense 4B model would disambiguate.

6. **Steering method scope:** Only additive and reflection steering were tested. Other approaches (activation patching, SAE-based steering, DPO-style fine-tuning) might work where linear steering fails on MoE.

## Related
- [[wiki/experiments/mlx-interpretability]]
