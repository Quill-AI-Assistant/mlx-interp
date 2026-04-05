# Experiment 001: Cross-Model Replication — Gemma 4 E4B

**Status:** HYPOTHESES LOCKED — awaiting results
**Date:** 2026-04-05
**Author:** Alin Huzmezan
**Git SHA:** (to be filled at commit)

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

_To be filled after experiments complete. Each result references a sealed run directory._

### H1: Color Concept Swap

| Metric | Qwen2.5-7B | Gemma 4 E4B | Hypothesis |
|---|---|---|---|
| Swap rate | 91.1% | _pending_ | 60-80% |
| Control contamination | 2/25 | _pending_ | similar |
| Random baseline | 13.8% ± 4.8% | _pending_ | similar |

**Run:** `runs/2026-04-05-XXXX-rigorous-gemma4-H1-H4-baseline/`

### H2: Sycophancy Decomposition

| Metric | Qwen2.5-7B | Gemma 4 E4B | Hypothesis |
|---|---|---|---|
| SyA-GA cosine | 0.61 | _pending_ | 0.3-0.5 |
| SyA-GA best layer | 20 (of 28) | _pending_ | proportionally earlier |

**Run:** `runs/2026-04-05-XXXX-sycophancy_3way-gemma4-H2-H5/`

### H3: Probe Confound

| Condition | Qwen2.5-7B | Gemma 4 E4B | Hypothesis |
|---|---|---|---|
| 3rd-party emotion → pressure | 56% | _pending_ | >60% |
| Clinical pressure → pressure | 0% | _pending_ | similar |

### H4: Evaluation Contamination

| Metric | Qwen2.5-7B | Gemma 4 E4B | Hypothesis |
|---|---|---|---|
| Mean R-C diff | +0.35 | _pending_ | ~0.25 |

### H5: Layer Distribution

| Metric | Qwen2.5-7B | Gemma 4 E4B | Hypothesis |
|---|---|---|---|
| Best SyA-GA layer | 20/28 (0.71) | _pending_ | 17-25/42 (0.40-0.60) |

## Analysis

_To be written after results are in._

## Limitations

_To be written after results are in._
