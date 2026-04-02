#!/usr/bin/env python3
"""
Concept Swap Explorer - Interactive web UI for concept swapping.

Swap any two concepts (e.g., green/red) at the activation level using
contrastive probe extraction. See side-by-side normal vs swapped outputs
and run a built-in test suite.

Usage:
    python -m tools.concept_swap_explorer
    python -m tools.concept_swap_explorer --model mlx-community/Qwen2.5-7B-Instruct-4bit
"""

import json
import os
import threading
import warnings

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm import load as _load_model
# model options handled inline
from tools.explorer_utils import add_lifecycle_routes, run_explorer, create_arg_parser
from tools.theme import THEME_CSS, THEME_TOGGLE_HTML, THEME_JS

# Server reference for shutdown (single-element list for mutability)
_server_ref = []

# Path to templates directory
_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


# ── Core swap classes (ported from examples/concept_swap_vignette.py) ────────

class CaptureLayer(nn.Module):
    """Captures a layer's full output tensor."""
    def __init__(self, original_layer, idx, capture_dict):
        super().__init__()
        # Bypass nn.Module.__setattr__ so _layer stays in __dict__
        object.__setattr__(self, '_layer', original_layer)
        object.__setattr__(self, '_idx', idx)
        object.__setattr__(self, '_capture', capture_dict)

    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped layer."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            layer = object.__getattribute__(self, '_layer')
            return getattr(layer, name)

    def __call__(self, x, *args, **kwargs):
        out = self._layer(x, *args, **kwargs)
        # Some layers return tuples (hidden_state, attention_weights, ...)
        hidden = out[0] if isinstance(out, tuple) else out
        self._capture[self._idx] = hidden.astype(mx.float32)
        return out


class SwapLayer(nn.Module):
    """Mirrors activations across the concept plane (Householder reflection)."""
    def __init__(self, original_layer, direction, alpha=1.0):
        super().__init__()
        object.__setattr__(self, '_layer', original_layer)
        object.__setattr__(self, '_direction', direction)
        object.__setattr__(self, '_alpha', alpha)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            layer = object.__getattribute__(self, '_layer')
            return getattr(layer, name)

    def __call__(self, x, *args, **kwargs):
        out = self._layer(x, *args, **kwargs)
        # Handle tuple returns (hidden_state, attention_weights, ...)
        if isinstance(out, tuple):
            hidden = out[0]
            h_f = hidden.astype(mx.float32)
            proj = mx.sum(h_f * self._direction, axis=-1, keepdims=True)
            h_modified = h_f - 2.0 * self._alpha * proj * self._direction
            return (h_modified.astype(hidden.dtype),) + out[1:]
        else:
            out_f = out.astype(mx.float32)
            proj = mx.sum(out_f * self._direction, axis=-1, keepdims=True)
            out_modified = out_f - 2.0 * self._alpha * proj * self._direction
            return out_modified.astype(out.dtype)


class TranslateLayer(nn.Module):
    """Translates activations from concept A toward concept B.

    For non-antonymic concept pairs (e.g. apple/banana) where a
    Householder reflection does not make sense. Subtracts the A
    component and adds the B component instead.
    """
    def __init__(self, original_layer, direction_a, direction_b, alpha=1.0):
        super().__init__()
        object.__setattr__(self, '_layer', original_layer)
        object.__setattr__(self, '_dir_a', direction_a)
        object.__setattr__(self, '_dir_b', direction_b)
        object.__setattr__(self, '_alpha', alpha)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            layer = object.__getattribute__(self, '_layer')
            return getattr(layer, name)

    def __call__(self, x, *args, **kwargs):
        out = self._layer(x, *args, **kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
        h_f = hidden.astype(mx.float32)
        # Subtract A component, add B component
        proj_a = mx.sum(h_f * self._dir_a, axis=-1, keepdims=True)
        h_modified = h_f - self._alpha * proj_a * self._dir_a + self._alpha * proj_a * self._dir_b
        result = h_modified.astype(hidden.dtype)
        if isinstance(out, tuple):
            return (result,) + out[1:]
        return result


# ── Helper functions ─────────────────────────────────────────────────────────

def _capture_at_position(model_raw, tokenizer, prompt, target_word, layer_indices):
    """Capture activations at the specific token position of target_word."""
    tokens = tokenizer.encode(prompt)

    # Find position of target word
    target_pos = None
    decoded_so_far = ""
    for i, tok in enumerate(tokens):
        decoded_so_far = tokenizer.decode(tokens[:i + 1])
        if target_word.lower() in decoded_so_far.lower() and target_pos is None:
            target_pos = i
            break

    if target_pos is None:
        target_pos = len(tokens) - 1

    captured = {}
    originals = {}
    for idx in layer_indices:
        originals[idx] = model_raw.layers[idx]
        model_raw.layers[idx] = CaptureLayer(originals[idx], idx, captured)

    token_array = mx.array([tokens])
    out = model_raw(token_array)
    mx.eval(out)

    for idx in layer_indices:
        model_raw.layers[idx] = originals[idx]

    results = {}
    for idx in captured:
        mx.eval(captured[idx])
        results[idx] = np.array(captured[idx][0, target_pos, :])

    return results


def _extract_direction(model_raw, tokenizer, concept_a, concept_b):
    """Extract concept_a <-> concept_b direction using cross-validated probe.

    Returns dict of {layer_idx: direction_unit_vector} and extraction info.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # 100+ diverse templates grouped by usage category.
    # Varied contexts so the probe learns the concept, not just context.

    # Adjective templates (35)
    adjective_templates = [
        "The {c} object sat on the table",
        "A bright {c} light",
        "Everything looked {c}",
        "A deep {c}",
        "A pale {c}",
        "Dark {c}",
        "A {c} surface",
        "A small {c} stone",
        "The {c} glow was unmistakable",
        "A faintly {c} haze",
        "The {c} tint spread evenly",
        "An intensely {c} flash",
        "A strikingly {c} pattern",
        "A subtly {c} shade appeared",
        "The dull {c} finish",
        "A warm {c} undertone",
        "A cool {c} shimmer",
        "A rich {c} hue",
        "A distinctly {c} aura",
        "The muted {c} backdrop",
        "An almost {c} quality",
        "A boldly {c} stripe",
        "A softly {c} blur",
        "A vivid {c} spark",
        "The unmistakably {c} tone",
        "A barely {c} tinge",
        "An overwhelmingly {c} field",
        "The nearly {c} horizon",
        "A uniformly {c} coating",
        "A partially {c} edge",
        "The translucent {c} layer",
        "An opaque {c} slab",
        "A shimmering {c} veil",
        "A solid {c} block",
        "The transparent {c} film",
    ]

    # Noun templates (25)
    noun_templates = [
        "Pure {c}",
        "Vivid {c}",
        "Covered in {c}",
        "The color of {c}",
        "A splash of {c}",
        "Nothing but {c}",
        "A shade of {c}",
        "The essence of {c}",
        "Surrounded by {c}",
        "Bathed in {c}",
        "Drenched in {c}",
        "A pool of {c}",
        "A burst of {c}",
        "Hints of {c}",
        "Traces of {c}",
        "Overwhelmed by {c}",
        "A world of {c}",
        "A touch of {c}",
        "Streaks of {c}",
        "A wall of {c}",
        "A sea of {c}",
        "A ribbon of {c}",
        "Swirls of {c}",
        "A field of {c}",
        "Layers of {c}",
    ]

    # Sentence templates (30)
    sentence_templates = [
        "I see something {c}",
        "It was painted {c}",
        "She chose {c}",
        "The wall is {c}",
        "His shirt was {c}",
        "The car is {c}",
        "The light turned {c}",
        "That flower is {c}",
        "The door was {c}",
        "The ribbon is {c}",
        "They picked {c}",
        "The building is {c}",
        "The paper was {c}",
        "It glowed {c}",
        "The sign was {c}",
        "The sky turned {c}",
        "A {c} bird landed",
        "The water looked {c}",
        "She wore a {c} dress",
        "The leaves were {c}",
        "The room felt {c}",
        "He painted the fence {c}",
        "The curtains were dyed {c}",
        "The old barn looked {c}",
        "Her eyes were {c}",
        "The sunset turned {c}",
        "The sauce looked {c}",
        "The flag was {c}",
        "The forest appeared {c}",
        "The ceiling had a {c} stain",
    ]

    # Abstract / contextual templates (22)
    abstract_templates = [
        "In a {c} mood",
        "The tone was {c}",
        "It felt {c} somehow",
        "The atmosphere was {c}",
        "Everything seemed {c}",
        "A {c} feeling crept in",
        "The vibe was distinctly {c}",
        "An undeniably {c} presence",
        "The memory was {c}",
        "The day felt {c}",
        "The dream was {c}",
        "A {c} sensation washed over",
        "The landscape was {c}",
        "The morning air felt {c}",
        "The texture was {c}",
        "The music sounded {c}",
        "The wind carried a {c} scent",
        "The night sky looked {c}",
        "The reflection appeared {c}",
        "A {c} warmth filled the space",
        "The path ahead looked {c}",
        "The silence felt {c}",
    ]

    templates = adjective_templates + noun_templates + sentence_templates + abstract_templates

    a_pairs = [(t.format(c=concept_a), concept_a) for t in templates]
    b_pairs = [(t.format(c=concept_b), concept_b) for t in templates]

    n_layers = len(model_raw.layers)
    layer_indices = list(range(n_layers // 3, n_layers - 1))

    a_acts = {i: [] for i in layer_indices}
    b_acts = {i: [] for i in layer_indices}

    for prompt, target in a_pairs:
        acts = _capture_at_position(model_raw, tokenizer, prompt, target, layer_indices)
        for idx, act in acts.items():
            a_acts[idx].append(act)

    for prompt, target in b_pairs:
        acts = _capture_at_position(model_raw, tokenizer, prompt, target, layer_indices)
        for idx, act in acts.items():
            b_acts[idx].append(act)

    # Train cross-validated probe per layer
    all_directions = {}
    layer_info = []
    for idx in layer_indices:
        if len(a_acts[idx]) < 10 or len(b_acts[idx]) < 10:
            continue

        X = np.array(a_acts[idx] + b_acts[idx])
        y = np.array([1] * len(a_acts[idx]) + [0] * len(b_acts[idx]))

        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        # L2 reg — dense direction needed to actually steer behavior
        # L1 is more interpretable but too sparse to change model outputs
        probe = LogisticRegression(max_iter=2000, C=0.1, solver='liblinear')

        cv_scores = cross_val_score(probe, X, y, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()

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
            layer_info.append({
                'layer': int(idx),
                'train_acc': float(round(train_acc, 4)),
                'cv_acc': float(round(cv_mean, 4)),
                'nonzero': int(n_nonzero),
                'total_features': int(len(direction)),
                'norm': float(round(norm, 3)),
            })

    if not all_directions:
        return {}, []

    # Select by cross-validation accuracy
    good_layers = {k: v for k, v in all_directions.items() if v['cv_acc'] >= 0.90}

    if not good_layers:
        sorted_layers = sorted(all_directions.keys(),
                               key=lambda i: all_directions[i]['cv_acc'], reverse=True)
        good_layers = {k: all_directions[k] for k in sorted_layers[:3]}

    sorted_good = sorted(good_layers.keys(),
                         key=lambda i: (good_layers[i]['cv_acc'], good_layers[i]['norm']),
                         reverse=True)
    selected = sorted_good[:3]

    directions = {}
    for idx in selected:
        directions[idx] = all_directions[idx]['direction']

    # Mark selected layers in info
    for info in layer_info:
        info['selected'] = info['layer'] in selected

    return directions, layer_info


def _apply_swap(model_raw, directions, alpha=1.0):
    """Patch model layers to swap concepts using Householder reflection."""
    originals = {}
    for idx, direction in directions.items():
        originals[idx] = model_raw.layers[idx]
        d = mx.array(direction)
        model_raw.layers[idx] = SwapLayer(originals[idx], d, alpha)
    return originals


def _apply_translate(model_raw, directions_a, directions_b, alpha=1.0):
    """Patch model layers to translate concept A toward concept B."""
    originals = {}
    for idx in directions_a:
        if idx not in directions_b:
            continue
        originals[idx] = model_raw.layers[idx]
        da = mx.array(directions_a[idx])
        db = mx.array(directions_b[idx])
        model_raw.layers[idx] = TranslateLayer(originals[idx], da, db, alpha)
    return originals


def _restore_model(model_raw, originals):
    """Restore original layers."""
    for idx, orig in originals.items():
        model_raw.layers[idx] = orig


def _clean_response(text):
    """Strip EOS tokens and artifacts, return first word."""
    for eos in ["<|endoftext|>", "<|im_end|>", "</s>", "Human"]:
        if eos in text:
            text = text[:text.index(eos)]
    text = text.strip()
    if not text:
        return "(empty)"
    return text.split()[0].rstrip(".,!;:?")


def _evaluate_swap(normal_text, swapped_text, concept_a, concept_b):
    """Evaluate swap quality using keyword detection and semantic analysis.

    Returns dict with:
    - swapped: bool — did the concept actually flip?
    - confidence: float 0-1 — how confident are we?
    - method: str — which detection method triggered
    """
    n = normal_text.lower()
    s = swapped_text.lower()

    # Method 1: Direct keyword detection
    a_in_normal = concept_a.lower() in n
    b_in_normal = concept_b.lower() in n
    a_in_swapped = concept_a.lower() in s
    b_in_swapped = concept_b.lower() in s

    # Perfect swap: A->B or B->A
    if a_in_normal and b_in_swapped and not a_in_swapped:
        return {"swapped": True, "confidence": 1.0, "method": "keyword_flip_a_to_b"}
    if b_in_normal and a_in_swapped and not b_in_swapped:
        return {"swapped": True, "confidence": 1.0, "method": "keyword_flip_b_to_a"}

    # Partial swap: concept disappeared but target didn't appear
    if a_in_normal and not a_in_swapped and not b_in_swapped:
        return {"swapped": True, "confidence": 0.5, "method": "concept_removed"}
    if b_in_normal and not b_in_swapped and not a_in_swapped:
        return {"swapped": True, "confidence": 0.5, "method": "concept_removed"}

    # Method 2: Synonym/related word detection for known concept pairs
    SYNONYMS = {
        "green": ["emerald", "lime", "olive", "verdant", "jade"],
        "red": ["crimson", "scarlet", "ruby", "cherry", "vermilion"],
        "blue": ["azure", "cobalt", "navy", "sapphire", "cerulean", "indigo"],
        "yellow": ["golden", "gold", "amber", "canary", "lemon", "saffron"],
        "hot": ["warm", "burning", "scorching", "boiling", "heated", "fiery"],
        "cold": ["cool", "freezing", "frigid", "chilly", "icy", "frozen"],
        "fast": ["quick", "rapid", "swift", "speedy"],
        "slow": ["sluggish", "leisurely", "gradual", "unhurried"],
        "sweet": ["sugary", "honeyed", "saccharine"],
        "bitter": ["tart", "acrid", "sharp", "sour"],
        "happy": ["joyful", "cheerful", "glad", "delighted", "pleased"],
        "sad": ["sorrowful", "unhappy", "melancholy", "gloomy", "depressed"],
        "big": ["large", "huge", "enormous", "massive", "giant", "gigantic"],
        "small": ["tiny", "little", "miniature", "minute", "petite"],
    }

    a_syns = set([concept_a.lower()] + SYNONYMS.get(concept_a.lower(), []))
    b_syns = set([concept_b.lower()] + SYNONYMS.get(concept_b.lower(), []))

    a_in_n = any(syn in n for syn in a_syns)
    b_in_n = any(syn in n for syn in b_syns)
    a_in_s = any(syn in s for syn in a_syns)
    b_in_s = any(syn in s for syn in b_syns)

    if a_in_n and b_in_s and not a_in_s:
        return {"swapped": True, "confidence": 0.8, "method": "synonym_flip_a_to_b"}
    if b_in_n and a_in_s and not b_in_s:
        return {"swapped": True, "confidence": 0.8, "method": "synonym_flip_b_to_a"}

    # Method 3: Concept introduced — target concept keyword appears in swapped
    # output but was absent from normal output (regardless of source concept)
    # Uses synonym-expanded detection for broader coverage
    if b_in_s and not b_in_n:
        return {"swapped": True, "confidence": 0.9, "method": "concept_introduced_b"}
    if a_in_s and not a_in_n:
        return {"swapped": True, "confidence": 0.9, "method": "concept_introduced_a"}

    # Method 4: Hedging detection (model refuses to answer directly)
    hedge_phrases = ["great question", "trick question", "it depends", "both", "complex", "tricky"]
    normal_hedges = any(h in n for h in hedge_phrases)
    swapped_hedges = any(h in s for h in hedge_phrases)

    if not normal_hedges and swapped_hedges:
        return {"swapped": False, "confidence": 0.7, "method": "hedging_detected"}

    # Method 5: Fallback string comparison
    if n.split()[:3] != s.split()[:3]:
        return {"swapped": True, "confidence": 0.3, "method": "text_differs_low_confidence"}

    return {"swapped": False, "confidence": 0.8, "method": "no_change_detected"}



def _compute_perplexity(model, tokenizer, text):
    """Compute perplexity of a text string.

    Returns float perplexity, or None if computation fails.
    """
    try:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            return None
        token_array = mx.array([tokens])
        logits = model(token_array)
        mx.eval(logits)

        # Shift logits and targets for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_targets = token_array[:, 1:]

        # Log softmax for numerical stability
        log_probs = shift_logits - mx.logsumexp(shift_logits, axis=-1, keepdims=True)

        # Gather the log probabilities of actual next tokens
        seq_len = shift_targets.shape[1]
        target_log_probs = []
        for i in range(seq_len):
            target_id = shift_targets[0, i].item()
            target_log_probs.append(log_probs[0, i, target_id].item())

        # Clamp individual log probs and average
        target_log_probs = [max(lp, -20.0) for lp in target_log_probs]
        avg_neg_log_prob = -np.mean(target_log_probs)
        avg_neg_log_prob = min(avg_neg_log_prob, 15.0)  # cap at ~3.3M perplexity
        perplexity = float(np.exp(avg_neg_log_prob))
        return perplexity
    except Exception:
        return None


# ── Test suite ───────────────────────────────────────────────────────────────

# Common test subjects per concept category
_TEST_SUBJECTS = {
    "color": {
        "subjects": [
            ("grass", "a"), ("fire truck", "a"), ("leaves in spring", None),
            ("emerald", "an"), ("ruby", "a"), ("blood", None),
            ("lime", "a"), ("cherry", "a"), ("stop sign", "a"),
            ("go signal", "a"), ("the Hulk", None), ("Santa's suit", None),
            ("tomato", "a"), ("broccoli", None), ("strawberry", "a"),
            ("watermelon rind", "a"), ("shamrock", "a"), ("the sky", None),
            ("ocean", "the"), ("banana", "a"),
        ],
        "associations": [
            "What does a {a} traffic light mean?",
            "What does a {b} traffic light mean?",
            "{A} means ___ and {b} means ___.",
            "In finance, {a} means ___ and {b} means ___.",
        ],
        "completions": [
            "Roses are ___.",
            "The grass is always ___ on the other side.",
            "Kermit the Frog said it's not easy being ___.",
        ],
    },
    "temperature": {
        "subjects": [
            ("fire", None), ("ice", None), ("the sun", None),
            ("snow", None), ("lava", None), ("Antarctica", None),
            ("a sauna", None), ("an oven", None), ("a freezer", None),
            ("a desert at noon", None),
        ],
    },
    "speed": {
        "subjects": [
            ("a cheetah", None), ("a turtle", None), ("a snail", None),
            ("light", None), ("a rocket", None), ("a sloth", None),
            ("a bullet", None), ("a glacier", None),
        ],
    },
    "taste": {
        "subjects": [
            ("honey", None), ("coffee", None), ("sugar", None),
            ("dark chocolate", None), ("lemon", None), ("grapefruit", None),
            ("maple syrup", None), ("vinegar", None),
        ],
    },
    "size": {
        "subjects": [
            ("an elephant", None), ("an ant", None), ("the sun", None),
            ("a mouse", None), ("a whale", None), ("a skyscraper", None),
            ("a grain of sand", None), ("Jupiter", None),
        ],
    },
    "emotion": {
        "subjects": [
            ("a birthday party", None), ("a funeral", None),
            ("winning the lottery", None), ("losing a game", None),
            ("a wedding", None), ("a rainy day", None),
            ("a sunny morning", None), ("getting fired", None),
        ],
    },
}

# Map concept pairs to categories
_CONCEPT_CATEGORIES = {
    frozenset({"green", "red"}): "color",
    frozenset({"blue", "yellow"}): "color",
    frozenset({"red", "blue"}): "color",
    frozenset({"hot", "cold"}): "temperature",
    frozenset({"fast", "slow"}): "speed",
    frozenset({"sweet", "bitter"}): "taste",
    frozenset({"big", "small"}): "size",
    frozenset({"happy", "sad"}): "emotion",
}


def _generate_test_questions(concept_a, concept_b):
    """Generate test questions for any concept pair.

    Returns list of (question, type) tuples.
    """
    questions = []
    a, b = concept_a, concept_b

    # Determine category
    pair_key = frozenset({a.lower(), b.lower()})
    category = _CONCEPT_CATEGORIES.get(pair_key, None)

    # Generic templates that work for any concept pair
    generic_concept_templates = [
        f"What is something that is {a}?",
        f"What is something that is {b}?",
        f"Name something {a}.",
        f"Name something {b}.",
        f"Is {a} the opposite of {b}?",
        f"Describe something that is {a}.",
        f"Describe something that is {b}.",
        f"Complete: something very {a} is ___.",
        f"Complete: something very {b} is ___.",
        f"What word is the opposite of {a}?",
        f"What word is the opposite of {b}?",
        f"In one word, {a} or {b}: which describes fire?",
        f"In one word, {a} or {b}: which describes ice?",
        f"If something is not {a}, it might be ___.",
        f"If something is not {b}, it might be ___.",
    ]
    questions.extend([(q, "concept") for q in generic_concept_templates])

    # Category-specific questions
    if category and category in _TEST_SUBJECTS:
        cat_data = _TEST_SUBJECTS[category]

        # Subject-based questions: "What color is X?" / "Is X hot or cold?"
        for subject, article in cat_data.get("subjects", []):
            if category == "color":
                questions.append((f"What color is {subject}?", "concept"))
            else:
                label = subject
                if article:
                    label = f"{article} {subject}"
                questions.append(
                    (f"Is {label} typically {a} or {b}?", "concept")
                )

        # Association questions (with concept substitution)
        for tmpl in cat_data.get("associations", []):
            q = tmpl.replace("{a}", a).replace("{b}", b)
            q = q.replace("{A}", a.capitalize()).replace("{B}", b.capitalize())
            questions.append((q, "concept"))

        # Completion questions
        for q in cat_data.get("completions", []):
            questions.append((q, "concept"))
    else:
        # Unknown category: generate more generic probes
        extra = [
            f"Give an example of something {a}.",
            f"Give an example of something {b}.",
            f"When you think of {a}, what comes to mind?",
            f"When you think of {b}, what comes to mind?",
            f"The feeling of {a} is like ___.",
            f"The feeling of {b} is like ___.",
            f"A {a} thing vs a {b} thing: which is better?",
            f"Name three {a} things.",
            f"Name three {b} things.",
            f"Which is more {a}: a cat or a dog?",
        ]
        questions.extend([(q, "concept") for q in extra])

    # Controls -- no concept involvement (20 diverse questions)
    controls = [
        # Math
        ("What is 2 + 2?", "control"),
        ("What is 5 times 3?", "control"),
        ("What is 100 divided by 10?", "control"),
        # Geography
        ("What planet is closest to the sun?", "control"),
        ("What is the capital of France?", "control"),
        ("What continent is Egypt in?", "control"),
        ("What ocean is the largest?", "control"),
        # History / Literature
        ("Who wrote Romeo and Juliet?", "control"),
        ("What year did World War II end?", "control"),
        ("Who was the first person on the moon?", "control"),
        # Science
        ("What is the boiling point of water in Celsius?", "control"),
        ("How many continents are there?", "control"),
        ("What is the chemical symbol for gold?", "control"),
        ("How many bones does an adult human have?", "control"),
        # Language / General
        ("What language is spoken in Japan?", "control"),
        ("How many days in a week?", "control"),
        ("How many letters in the English alphabet?", "control"),
        # Logic / Common sense
        ("How many legs does a spider have?", "control"),
        ("What comes after Monday?", "control"),
        ("Is the Earth round or flat?", "control"),
    ]
    questions.extend(controls)

    return questions


# ── Probe generation ─────────────────────────────────────────────────────────

def _generate_probes(concept_a, concept_b):
    """Generate diverse contrastive prompts for any concept pair.

    Combines 10 sentence structures with both concepts to produce a large
    set of probe prompts suitable for direction extraction or evaluation.
    """
    a, b = concept_a, concept_b
    probes = []

    # Structure 1: Adjective use
    adjective_frames = [
        "The {c} object sat on the table",
        "A bright {c} light appeared",
        "Something distinctly {c}",
        "A strikingly {c} pattern",
        "The clearly {c} surface",
        "An intensely {c} glow",
        "A subtly {c} shade",
        "The unmistakably {c} tint",
        "A deeply {c} hue",
        "The faintly {c} shimmer",
    ]
    for frame in adjective_frames:
        probes.append({"prompt_a": frame.format(c=a), "prompt_b": frame.format(c=b),
                        "category": "adjective"})

    # Structure 2: Noun use
    noun_frames = [
        "Pure {c}",
        "The color of {c}",
        "A splash of {c}",
        "Surrounded by {c}",
        "Bathed in {c}",
        "A field of {c}",
        "Layers of {c}",
        "A sea of {c}",
        "Traces of {c}",
        "A burst of {c}",
    ]
    for frame in noun_frames:
        probes.append({"prompt_a": frame.format(c=a), "prompt_b": frame.format(c=b),
                        "category": "noun"})

    # Structure 3: Comparison
    comparison_frames = [
        f"Is this more {a} or {b}?",
        f"Between {a} and {b}, this is clearly",
        f"Not {a}, but {b}",
        f"Not {b}, but {a}",
        f"More {a} than {b}",
        f"More {b} than {a}",
        f"Neither {a} nor {b}, but somewhere between",
        f"Definitely {a}, not {b}",
        f"Closer to {a} than to {b}",
        f"Almost {b}, but still {a}",
    ]
    for frame in comparison_frames:
        probes.append({"prompt": frame, "category": "comparison"})

    # Structure 4: Completion
    completion_frames = [
        f"The opposite of {a} is ___.",
        f"The opposite of {b} is ___.",
        f"Something very {a} is ___.",
        f"Something very {b} is ___.",
        f"If not {a}, then ___.",
        f"If not {b}, then ___.",
        f"The most {a} thing I know is ___.",
        f"The most {b} thing I know is ___.",
        f"To be {a} means to be ___.",
        f"To be {b} means to be ___.",
    ]
    for frame in completion_frames:
        probes.append({"prompt": frame, "category": "completion"})

    # Structure 5: Association
    association_frames = [
        f"When I think of {a}, I think of ___.",
        f"When I think of {b}, I think of ___.",
        f"{a.capitalize()} reminds me of ___.",
        f"{b.capitalize()} reminds me of ___.",
        f"A word related to {a}: ___.",
        f"A word related to {b}: ___.",
        f"{a.capitalize()} and ___ go together.",
        f"{b.capitalize()} and ___ go together.",
        f"The feeling of {a} is like ___.",
        f"The feeling of {b} is like ___.",
    ]
    for frame in association_frames:
        probes.append({"prompt": frame, "category": "association"})

    return probes


# ── Flask app ────────────────────────────────────────────────────────────────

def create_app(model_name: str):
    """Create the Flask app for the concept swap explorer."""
    try:
        from flask import Flask, render_template, request, jsonify, Response
    except ImportError:
        raise ImportError("Flask required. Install with: pip install flask")

    app = Flask(__name__, template_folder=_TEMPLATE_DIR)
    model_options = [
        {"id": "mlx-community/gemma-2-2b-it-4bit", "display": "gemma-2-2b-it-4bit"},
        {"id": "mlx-community/Qwen2.5-7B-Instruct-4bit", "display": "Qwen2.5-7B-Instruct-4bit"},
        {"id": "mlx-community/Qwen3-14B-4bit", "display": "Qwen3-14B-4bit"},
        {"id": "mlx-community/gemma-3-4b-it-4bit", "display": "gemma-3-4b-it-4bit"},
        {"id": "mlx-community/Qwen3.5-9B-4bit", "display": "Qwen3.5-9B-4bit"},
    ]

    # Model state
    state = {
        "model_name": model_name,
        "model": None, "tokenizer": None,
        "n_layers": 0,
        "d_model": 0,
        "directions": None,       # {layer_idx: np direction array}
        "directions_b": None,     # For translation mode: B direction per layer
        "layer_info": [],          # extraction results per layer
        "concept_a": "white",
        "concept_b": "black",
        "mode": "reflection",     # "reflection" or "translation"
        "recommended_alpha": 1.5, # auto-tuned alpha recommendation
    }

    # Lock to prevent concurrent model access (MLX is not thread-safe)
    _compute_lock = threading.Lock()

    def load_model(name: str):
        print(f"Loading {name}...")
        model, tokenizer = _load_model(name)
        state["model"] = model
        state["tokenizer"] = tokenizer
        state["model_name"] = name

        model_raw = state["model"].model if hasattr(state["model"], "model") else state["model"]
        if hasattr(model_raw, 'layers'):
            state["n_layers"] = len(model_raw.layers)
        elif hasattr(model_raw, 'args') and hasattr(model_raw.args, 'num_hidden_layers'):
            state["n_layers"] = model_raw.args.num_hidden_layers

        if hasattr(model_raw, 'args'):
            if hasattr(model_raw.args, 'hidden_size'):
                state["d_model"] = model_raw.args.hidden_size
            elif hasattr(model_raw.args, 'dim'):
                state["d_model"] = model_raw.args.dim

        # Reset directions on model swap
        state["directions"] = None
        state["directions_b"] = None
        state["layer_info"] = []

        print(f"Loaded: {state['n_layers']} layers, d_model={state['d_model']}")

    load_model(model_name)

    @app.route('/')
    def index():
        return render_template(
            "concept_swap_explorer.html",
            model_name=state["model_name"],
            n_layers=state["n_layers"],
            model_options=model_options,
            theme_css=THEME_CSS,
            theme_toggle_html=THEME_TOGGLE_HTML,
            theme_js=THEME_JS,
        )

    @app.route('/swap_model', methods=['POST'])
    def swap_model():
        data = request.json
        new_model = data.get('model', '')
        if new_model and new_model != state["model_name"]:
            if not _compute_lock.acquire(blocking=False):
                return jsonify({"error": "Computation in progress, cannot swap model"}), 429
            try:
                load_model(new_model)
            except Exception:
                return jsonify({"success": False, "error": "Model load failed"}), 500
            finally:
                _compute_lock.release()
        return jsonify({
            "model_name": state["model_name"],
            "n_layers": state["n_layers"]
        })

    @app.route('/extract_direction', methods=['POST'])
    def extract_direction():
        if not _compute_lock.acquire(blocking=False):
            return jsonify({"error": "Computation already in progress"}), 429
        try:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler

            data = request.json
            concept_a = data.get('concept_a', 'green').strip()
            concept_b = data.get('concept_b', 'red').strip()
            mode = data.get('mode', 'reflection')
            auto_alpha = data.get('auto_alpha', False)

            if not concept_a or not concept_b:
                return jsonify({"error": "Both concepts required"}), 400

            model_raw = state["model"].model if hasattr(state["model"], "model") else state["model"]
            tokenizer = state["tokenizer"]

            directions, layer_info = _extract_direction(
                model_raw, tokenizer, concept_a, concept_b
            )

            if not directions:
                return jsonify({"error": "No direction found -- concepts may be too similar"}), 400

            state["directions"] = directions
            state["layer_info"] = layer_info
            state["concept_a"] = concept_a
            state["concept_b"] = concept_b
            state["mode"] = mode

            # For translation mode, also extract B->A direction to get B's
            # representation separately
            if mode == "translation":
                directions_b, _ = _extract_direction(
                    model_raw, tokenizer, concept_b, concept_a
                )
                state["directions_b"] = directions_b
            else:
                state["directions_b"] = None

            selected_layers = [int(k) for k in sorted(directions.keys())]

            # Adaptive alpha: grid search over [0.5, 1.0, 1.5, 2.0]
            recommended_alpha = 1.5
            if auto_alpha and directions:
                model_obj = state["model"]
                sampler = make_sampler(temp=0.0)

                # Pick 3 quick probe questions
                probe_qs = [
                    f"What is something that is {concept_a}?",
                    f"Name something {concept_b}.",
                    f"Is {concept_a} the opposite of {concept_b}?",
                ]

                best_alpha = 1.5
                best_score = -1.0

                for test_alpha in [0.5, 1.0, 1.5, 2.0]:
                    score = 0.0
                    originals = _apply_swap(model_raw, directions, test_alpha)
                    try:
                        for pq in probe_qs:
                            prompt = f"Answer in exactly one word, no explanation. {pq}"
                            if hasattr(tokenizer, 'apply_chat_template'):
                                full_prompt = tokenizer.apply_chat_template(
                                    [{'role': 'user', 'content': prompt}],
                                    tokenize=False, add_generation_prompt=True
                                )
                            else:
                                full_prompt = prompt

                            text = ""
                            for resp in stream_generate(
                                model_obj, tokenizer, full_prompt,
                                max_tokens=10, sampler=sampler
                            ):
                                text += resp.text
                            swapped_ans = _clean_response(text)

                            # Generate normal answer for comparison
                            _restore_model(model_raw, originals)
                            text_n = ""
                            for resp in stream_generate(
                                model_obj, tokenizer, full_prompt,
                                max_tokens=10, sampler=sampler
                            ):
                                text_n += resp.text
                            normal_ans = _clean_response(text_n)
                            originals = _apply_swap(model_raw, directions, test_alpha)

                            ev = _evaluate_swap(
                                normal_ans, swapped_ans, concept_a, concept_b
                            )
                            if ev["swapped"]:
                                score += ev["confidence"]
                    finally:
                        _restore_model(model_raw, originals)

                    if score > best_score:
                        best_score = score
                        best_alpha = test_alpha

                recommended_alpha = best_alpha

            state["recommended_alpha"] = recommended_alpha

            return jsonify({
                "success": True,
                "selected_layers": selected_layers,
                "layer_info": layer_info,
                "concept_a": concept_a,
                "concept_b": concept_b,
                "mode": mode,
                "recommended_alpha": recommended_alpha,
            })
        except Exception as e:
            return jsonify({"error": f"Extraction failed: {str(e)}"}), 500
        finally:
            _compute_lock.release()

    @app.route('/generate', methods=['POST'])
    def generate_route():
        """Generate normal and (optionally) swapped responses side by side."""
        if not _compute_lock.acquire(blocking=False):
            return jsonify({"error": "Computation already in progress"}), 429
        try:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler

            data = request.json
            prompt_text = data.get('prompt', '')
            if len(prompt_text) > 10000:
                return jsonify({"error": "Prompt too long (max 10000 chars)"}), 400

            alpha = data.get('alpha', 1.5)
            swap_enabled = data.get('swap_enabled', False)
            max_tokens = data.get('max_tokens', 200)

            model = state["model"]
            tokenizer = state["tokenizer"]
            model_raw = state["model"].model if hasattr(state["model"], "model") else state["model"]
            sampler = make_sampler(temp=0.0)

            # Apply chat template if available
            if hasattr(tokenizer, 'apply_chat_template'):
                full_prompt = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': prompt_text}],
                    tokenize=False, add_generation_prompt=True
                )
            else:
                full_prompt = prompt_text

            # Normal generation
            normal_text = ""
            for resp in stream_generate(model, tokenizer, full_prompt,
                                        max_tokens=max_tokens, sampler=sampler):
                normal_text += resp.text

            # Swapped generation (only if enabled and directions exist)
            swapped_text = None
            if swap_enabled and state["directions"] is not None:
                # Apply based on mode
                if state["mode"] == "translation" and state["directions_b"] is not None:
                    originals = _apply_translate(
                        model_raw, state["directions"], state["directions_b"], alpha
                    )
                else:
                    originals = _apply_swap(model_raw, state["directions"], alpha)
                try:
                    swapped_text = ""
                    for resp in stream_generate(model, tokenizer, full_prompt,
                                                max_tokens=max_tokens, sampler=sampler):
                        swapped_text += resp.text
                finally:
                    _restore_model(model_raw, originals)

            result = {"normal": normal_text}
            if swapped_text is not None:
                result["swapped"] = swapped_text

            return jsonify(result)
        except Exception as e:
            return jsonify({"error": f"Generation failed: {str(e)}"}), 500
        finally:
            _compute_lock.release()

    @app.route('/run_test_suite', methods=['POST'])
    def run_test_suite():
        """Run the full test suite: normal, swapped, random baseline, and reverse."""
        if not _compute_lock.acquire(blocking=False):
            return jsonify({"error": "Computation already in progress"}), 429
        try:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler

            data = request.json
            alpha = data.get('alpha', 1.5)

            if state["directions"] is None:
                return jsonify({"error": "Extract direction first"}), 400

            concept_a = state["concept_a"]
            concept_b = state["concept_b"]
            model = state["model"]
            tokenizer = state["tokenizer"]
            model_raw = state["model"].model if hasattr(state["model"], "model") else state["model"]
            sampler = make_sampler(temp=0.0)

            def apply_chat(prompt_text):
                if hasattr(tokenizer, 'apply_chat_template'):
                    return tokenizer.apply_chat_template(
                        [{'role': 'user', 'content': prompt_text}],
                        tokenize=False, add_generation_prompt=True
                    )
                return prompt_text

            def gen_one(prompt_text):
                formatted = apply_chat(f"Answer in exactly one word, no explanation. {prompt_text}")
                text = ""
                for resp in stream_generate(model, tokenizer, formatted,
                                            max_tokens=10, sampler=sampler):
                    text += resp.text
                return _clean_response(text)

            def gen_full(prompt_text, max_tok=30):
                """Generate longer response for perplexity measurement."""
                formatted = apply_chat(prompt_text)
                text = ""
                for resp in stream_generate(model, tokenizer, formatted,
                                            max_tokens=max_tok, sampler=sampler):
                    text += resp.text
                return text.strip()

            # Generate dynamic test questions for this concept pair
            test_questions = _generate_test_questions(concept_a, concept_b)

            # Build random directions for baseline
            random_dirs = {}
            for idx, d in state["directions"].items():
                rand = np.random.randn(*d.shape)
                random_dirs[idx] = rand / np.linalg.norm(rand)

            # Build reverse directions (negate alpha direction)
            reverse_dirs = {}
            for idx, d in state["directions"].items():
                reverse_dirs[idx] = -d

            modes = [
                ("normal", None),
                ("swapped", state["directions"]),
                ("random", random_dirs),
                ("reverse", reverse_dirs),
            ]

            results = {}
            for mode_name, dirs in modes:
                originals = {}
                if dirs is not None:
                    originals = _apply_swap(model_raw, dirs, alpha)

                results[mode_name] = []
                for q, _ in test_questions:
                    answer = gen_one(q)
                    results[mode_name].append(answer)

                if originals:
                    _restore_model(model_raw, originals)

            # Perplexity measurement: pick a few prompts and measure
            # normal vs swapped perplexity
            perplexity_prompts = [
                f"Tell me about something {concept_a}.",
                f"Describe a {concept_b} object.",
                f"What is the meaning of {concept_a}?",
            ]
            ppl_normal_vals = []
            ppl_swapped_vals = []

            for pp in perplexity_prompts:
                # Normal perplexity
                normal_out = gen_full(pp)
                ppl_n = _compute_perplexity(model, tokenizer, normal_out)
                if ppl_n is not None:
                    ppl_normal_vals.append(ppl_n)

                # Swapped perplexity
                originals = _apply_swap(model_raw, state["directions"], alpha)
                try:
                    swapped_out = gen_full(pp)
                    ppl_s = _compute_perplexity(model, tokenizer, swapped_out)
                    if ppl_s is not None:
                        ppl_swapped_vals.append(ppl_s)
                finally:
                    _restore_model(model_raw, originals)

            avg_ppl_normal = float(np.mean(ppl_normal_vals)) if ppl_normal_vals else None
            avg_ppl_swapped = float(np.mean(ppl_swapped_vals)) if ppl_swapped_vals else None

            # Probe verification: re-run probes on swapped model activations
            # to verify concept has flipped at the representation level
            probe_verification_score = None
            try:
                from sklearn.linear_model import LogisticRegression

                # Pick 3 simple probes per concept
                probe_templates = [
                    "The {c} object", "Pure {c}", "I see something {c}",
                ]
                n_layers = len(model_raw.layers)
                layer_indices = list(state["directions"].keys())

                # Capture swapped model activations for concept A prompts
                originals = _apply_swap(model_raw, state["directions"], alpha)
                swapped_acts = {i: [] for i in layer_indices}
                try:
                    for tmpl in probe_templates:
                        prompt = tmpl.format(c=concept_a)
                        acts = _capture_at_position(
                            model_raw, tokenizer, prompt, concept_a, layer_indices
                        )
                        for idx, act in acts.items():
                            swapped_acts[idx].append(act)
                finally:
                    _restore_model(model_raw, originals)

                # Capture normal model activations for concept B prompts
                normal_b_acts = {i: [] for i in layer_indices}
                for tmpl in probe_templates:
                    prompt = tmpl.format(c=concept_b)
                    acts = _capture_at_position(
                        model_raw, tokenizer, prompt, concept_b, layer_indices
                    )
                    for idx, act in acts.items():
                        normal_b_acts[idx].append(act)

                # Measure: how similar are swapped-A activations to normal-B activations?
                # Use cosine similarity
                similarities = []
                for idx in layer_indices:
                    if swapped_acts[idx] and normal_b_acts[idx]:
                        sa = np.mean(swapped_acts[idx], axis=0)
                        nb = np.mean(normal_b_acts[idx], axis=0)
                        cos_sim = np.dot(sa, nb) / (
                            np.linalg.norm(sa) * np.linalg.norm(nb) + 1e-10
                        )
                        similarities.append(float(cos_sim))
                if similarities:
                    probe_verification_score = float(np.mean(similarities))
            except Exception:
                probe_verification_score = None

            # Classifier verification: train a fresh probe on normal A vs B,
            # then check if steered-A activations are classified as B
            classifier_verification_score = None
            try:
                from sklearn.linear_model import LogisticRegression as LR_verify
                # Capture normal-A activations (need them for training)
                normal_a_acts = {i: [] for i in layer_indices}
                for tmpl in probe_templates:
                    prompt = tmpl.format(c=concept_a)
                    acts = _capture_at_position(
                        model_raw, tokenizer, prompt, concept_a, layer_indices
                    )
                    for idx, act in acts.items():
                        normal_a_acts[idx].append(act)

                # Build training set: normal-A = 0, normal-B = 1
                X_train_parts, y_train_parts = [], []
                for idx in layer_indices:
                    if normal_a_acts[idx] and normal_b_acts[idx]:
                        X_train_parts.extend(normal_a_acts[idx])
                        y_train_parts.extend([0] * len(normal_a_acts[idx]))
                        X_train_parts.extend(normal_b_acts[idx])
                        y_train_parts.extend([1] * len(normal_b_acts[idx]))

                if X_train_parts:
                    X_train = np.array(X_train_parts)
                    y_train = np.array(y_train_parts)
                    verifier = LR_verify(C=0.1, solver='liblinear', max_iter=2000)
                    verifier.fit(X_train, y_train)

                    # Score steered-A activations: should be classified as B (=1)
                    X_steered = []
                    for idx in layer_indices:
                        if swapped_acts[idx]:
                            X_steered.extend(swapped_acts[idx])
                    if X_steered:
                        X_steered = np.array(X_steered)
                        preds = verifier.predict(X_steered)
                        classifier_verification_score = float(np.mean(preds == 1))
            except Exception:
                classifier_verification_score = None

            # Build response table with new evaluation
            rows = []
            concept_swapped = 0
            concept_total = 0
            control_affected = 0
            control_total = 0
            confidence_sum = 0.0
            confidence_count = 0

            for i, (q, qtype) in enumerate(test_questions):
                normal_ans = results["normal"][i]
                swapped_ans = results["swapped"][i]
                random_ans = results["random"][i]
                reverse_ans = results["reverse"][i]

                evaluation = _evaluate_swap(
                    normal_ans, swapped_ans, concept_a, concept_b
                )
                changed = evaluation["swapped"]
                confidence = evaluation["confidence"]
                method = evaluation["method"]

                if qtype == "concept":
                    concept_total += 1
                    if changed:
                        concept_swapped += 1
                    confidence_sum += confidence
                    confidence_count += 1
                else:
                    control_total += 1
                    # Fuzzy control comparison: "Four"=="4", "Seven"=="7", etc.
                    EQUIV = {
                        "four": "4", "4": "4", "seven": "7", "7": "7",
                        "one": "1", "two": "2", "three": "3", "five": "5",
                        "six": "6", "eight": "8", "nine": "9", "ten": "10",
                        "100": "100", "hundred": "100",
                    }
                    n_norm = EQUIV.get(normal_ans.lower(), normal_ans.lower())
                    s_norm = EQUIV.get(swapped_ans.lower(), swapped_ans.lower())
                    control_changed = n_norm != s_norm
                    if control_changed:
                        control_affected += 1

                rows.append({
                    "question": q,
                    "type": qtype,
                    "normal": normal_ans,
                    "swapped": swapped_ans,
                    "random": random_ans,
                    "reverse": reverse_ans,
                    "changed": changed,
                    "confidence": round(confidence, 2),
                    "method": method,
                })

            rand_changed = sum(
                1 for i in range(len(test_questions))
                if results["normal"][i].lower() != results["random"][i].lower()
            )

            reverse_changed = sum(
                1 for i in range(len(test_questions))
                if results["normal"][i].lower() != results["reverse"][i].lower()
            )

            avg_confidence = round(
                confidence_sum / confidence_count, 3
            ) if confidence_count > 0 else 0.0

            swap_rate = round(
                concept_swapped / concept_total, 3
            ) if concept_total > 0 else 0.0

            control_preservation = round(
                1.0 - (control_affected / control_total), 3
            ) if control_total > 0 else 1.0

            return jsonify({
                "rows": rows,
                "summary": {
                    "swap_rate": swap_rate,
                    "avg_confidence": avg_confidence,
                    "concept_swapped": concept_swapped,
                    "concept_total": concept_total,
                    "control_affected": control_affected,
                    "control_total": control_total,
                    "control_preservation_rate": control_preservation,
                    "avg_perplexity_normal": round(avg_ppl_normal, 2) if avg_ppl_normal else None,
                    "avg_perplexity_swapped": round(avg_ppl_swapped, 2) if avg_ppl_swapped else None,
                    "probe_verification_score": round(probe_verification_score, 3) if probe_verification_score is not None else None,
                    "classifier_verification_score": round(classifier_verification_score, 3) if classifier_verification_score is not None else None,
                    "random_baseline_rate": round(
                        rand_changed / len(test_questions), 3
                    ) if test_questions else 0.0,
                    "random_changed": rand_changed,
                    "reverse_swap_rate": round(
                        reverse_changed / len(test_questions), 3
                    ) if test_questions else 0.0,
                    "reverse_changed": reverse_changed,
                    "total_questions": len(test_questions),
                    "recommended_alpha": state["recommended_alpha"],
                },
            })
        except Exception as e:
            return jsonify({"error": f"Test suite failed: {str(e)}"}), 500
        finally:
            _compute_lock.release()

    @app.route('/generate_probes', methods=['POST'])
    def generate_probes_route():
        """Generate diverse contrastive probes for a concept pair."""
        data = request.json
        concept_a = data.get('concept_a', '').strip()
        concept_b = data.get('concept_b', '').strip()

        if not concept_a or not concept_b:
            return jsonify({"error": "Both concepts required"}), 400

        probes = _generate_probes(concept_a, concept_b)
        return jsonify({"probes": probes, "count": len(probes)})

    @app.route('/format_chat', methods=['POST'])
    def format_chat():
        data = request.json
        prompt = data.get('prompt', '')

        try:
            tokenizer = state["tokenizer"]
            if hasattr(tokenizer, 'apply_chat_template'):
                formatted = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': prompt}],
                    tokenize=False, add_generation_prompt=True
                )
                return jsonify({"formatted": formatted})
            else:
                return jsonify({"formatted": prompt, "note": "No chat template available"})
        except Exception:
            return jsonify({"error": "Internal error"}), 500

    # Add /health and /shutdown routes
    add_lifecycle_routes(app, state, "concept-swap", _server_ref)

    return app


def main():
    parser = create_arg_parser("Concept Swap Explorer", default_port=5006)
    parser.set_defaults(model="mlx-community/gemma-2-2b-it-4bit")
    args = parser.parse_args()

    app = create_app(args.model)
    run_explorer(
        app,
        name="Concept Swap Explorer",
        host=args.host,
        port=args.port,
        server_ref=_server_ref,
        open_browser=not args.no_browser
    )


if __name__ == "__main__":
    main()
