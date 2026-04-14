---
tags: [workbench, research]
---
# mlx-interp

Activation capture, linear probes, and concept steering for MLX language models on Apple Silicon.

## What it does

Extract per-layer hidden states from any MLX transformer model, train probes to detect behavioral concepts, and intervene at the activation level during inference.

**Core library** (`mlx_interp/`):
- **Capture**: Architecture-agnostic hook system via `__class__` proxy swap — no model surgery
- **Steering**: Add, subtract, or reflect direction vectors in activation space during generation
- **Analysis**: Cosine similarity, norm tracking, statistical comparison between conditions

## Verified results

All results from `experiments/rigorous_suite.py` with saved JSON outputs and fixed random seeds.

### Color concept swap

Bidirectional green/red swap at the activation level. Qwen2.5-7B-Instruct-4bit, 112 color questions + 25 controls (137 total).

| Metric | Value |
|---|---|
| Swap rate (alpha=1.5) | **91.1%** (102/112 color questions) |
| Control contamination | 2/25 |
| Random direction baseline | 13.8% ± 4.8% (10 random directions) |
| Binomial p-value | 3.2e-75 |

The model answers "red" when asked about grass and "green" when asked about fire trucks. 2 of 25 control questions showed minor bleed.

### Three-behavior sycophancy decomposition

Replicates the core finding from [Sycophancy Is Not One Thing](https://arxiv.org/abs/2509.21305) (Sep 2025) on MLX. 30 prompts per condition, 5-fold cross-validated.

| Pair | Best Layer | CV Accuracy | Significant |
|---|---|---|---|
| SyA vs GA | 20 | 86.7% ± 11.3% | YES |
| SyA vs Neutral | 3 | 100% ± 0.0% | YES |
| GA vs Neutral | 3 | 100% ± 0.0% | YES |
| SyPr vs Neutral | 0 | 100% ± 0.0% | YES |
| SyA vs SyPr | 0 | 100% ± 0.0% | YES |
| GA vs SyPr | 1 | 100% ± 0.0% | YES |

**Disagreement with published result**: SyA-Neutral and GA-Neutral directions share cosine 0.61 in mid-to-late layers — they are partially correlated, not orthogonal as the paper claims. This may be model-specific (Qwen 7B vs their models) or method-specific (mean-difference vs mechanistic interpretability).

### Probe confound diagnosis

Initial sycophancy probe showed 100% accuracy. Diagnostic with 6 control conditions proved it detects **emotional language**, not sycophancy pressure:

| Condition | Has emotion? | Has pressure? | Classified as pressure |
|---|---|---|---|
| Direct pressure ("I spent weeks...") | Yes | Yes | 92% |
| Neutral question | No | No | 12% |
| Third-party emotion ("My friend spent weeks...") | Yes | No | 56% |
| Clinical pressure ("Favorable review required") | No | Yes | 0% |
| Irrelevant emotion ("Best birthday ever!") | Yes | No | 56% |
| Bare question | No | No | 8% |

After retraining with sycophantic agreement (SyA) vs genuine agreement (GA) as the target, SyA-GA separation reaches 86.7% — detecting actual sycophancy, not surface emotional cues.

### Evaluation contamination (activation level)

Last-token cosine comparison across 5 task domains. Bootstrap 95% CIs from 1000 resamples.

| Task | Rubric-Clean | Control-Clean | Diff (R-C) | Sig? |
|---|---|---|---|---|
| self-reflection | 0.595 | 0.224 | +0.371 | YES |
| technical-diagnosis | 0.624 | 0.246 | +0.378 | YES |
| ambiguous-directive | 0.650 | 0.286 | +0.364 | YES |
| ethical-tradeoff | 0.628 | 0.297 | +0.331 | YES |
| knowledge-boundary | 0.671 | 0.348 | +0.323 | YES |

Rubric text shifts the generation point significantly more than random padding. However, a wrong-domain rubric produces nearly identical shift (~0.62 vs ~0.63), indicating **semantic relatedness**, not rubric-specific contamination.

## Experiments

All experiments live in `experiments/`. Run the full verified suite:

```bash
cd experiments
python3 rigorous_suite.py --models qwen2.5 --seed 42
python3 sycophancy_3way.py --n-prompts 30 --seed 42
```

| Experiment | Script | What it tests |
|---|---|---|
| Full rigorous suite | `rigorous_suite.py` | Color swap + sycophancy probe + contamination |
| 3-way sycophancy | `sycophancy_3way.py` | SyA vs GA vs SyPr decomposition |
| Color blind demo | `color_blind.py` | Interactive green/red swap |
| Interactive chat | `chat.py` | Chat with optional steering |
| Sycophancy diagnostic | `sycophancy_proper.py` | 6-condition confound analysis |

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- mlx >= 0.20.0
- mlx-lm >= 0.20.0
- numpy, scikit-learn

```bash
pip install -r requirements.txt
```

## Limitations

- **7B model scale**: All results from Qwen2.5-7B. Larger models may show different patterns.
- **Steering breaks at high alpha**: Alpha > 2.0 causes token loops. Directions extracted via mean-difference are too crude for causal intervention.
- **SyA-GA orthogonality**: Our result (cosine 0.61, aligned) disagrees with the published finding (orthogonal). Needs replication on other model families.
- **Single hardware**: All results from M2 Pro 16GB. Not tested on other Apple Silicon variants.

## License

MIT

## Related
- [[wiki/experiments/mlx-interpretability]]
