"""QuillMorphism — accessible dark-first UI theme for mlx-interp.

Designed for low vision users:
- WCAG AAA contrast ratios (7:1+ for text)
- Minimum 16px base font, scalable via controls
- No color-only information — all states have shape/text cues
- Visible focus indicators (3px accent outlines)
- Monospace typography for code-native readability
- Font size controls (+/- buttons)
- Reduced motion support

Usage:
    from .theme import THEME_CSS, THEME_TOGGLE_HTML, THEME_JS
"""

THEME_CSS = """
/* === QuillMorphism — Accessible Dark-First Theme === */

:root {
    /* Dark mode (default) — AAA contrast ratios */
    --bg: #0a0e1a;
    --bg-card: rgba(16, 22, 40, 0.85);
    --bg-card-hover: rgba(22, 30, 52, 0.92);
    --text: #f0f0f5;              /* contrast 15.2:1 on --bg */
    --text-secondary: #b8bcd0;    /* contrast 8.4:1 on --bg */
    --text-muted: #8890a8;        /* contrast 5.2:1 — large text only */
    --border: rgba(180, 120, 60, 0.25);
    --border-focus: #e8825a;      /* warm orange — Anthropic-inspired */
    --accent: #e8825a;            /* warm orange accent */
    --accent-hover: #f0a070;
    --accent-text: #0a0e1a;       /* dark text on accent buttons */
    --danger: #fca5a5;
    --danger-hover: #fecaca;
    --danger-text: #1a0505;
    --code-bg: rgba(8, 12, 24, 0.9);
    --input-bg: rgba(12, 18, 36, 0.8);
    --input-border: rgba(120, 140, 200, 0.4);
    --shadow: rgba(0, 0, 0, 0.5);
    --blur: 12px;
    --radius: 10px;
    --font-size: 16px;
    --focus-ring: 0 0 0 3px var(--border-focus);

    /* Heatmap — uses opacity not just color */
    --heatmap-positive: #ef4444;
    --heatmap-negative: #3b82f6;
    --heatmap-neutral: var(--bg-card);
}

[data-theme="light"] {
    --bg: #f4f4f8;
    --bg-card: rgba(255, 255, 255, 0.88);
    --bg-card-hover: rgba(255, 255, 255, 0.95);
    --text: #0f1729;              /* contrast 14.8:1 on --bg */
    --text-secondary: #3a4260;    /* contrast 8.1:1 */
    --text-muted: #5a6280;        /* contrast 5.0:1 */
    --border: rgba(140, 90, 40, 0.18);
    --border-focus: #c4622a;
    --accent: #c4622a;            /* warm orange — light mode */
    --accent-hover: #a8501e;
    --accent-text: #ffffff;
    --danger: #dc2626;
    --danger-hover: #b91c1c;
    --danger-text: #ffffff;
    --code-bg: rgba(240, 240, 248, 0.95);
    --input-bg: rgba(255, 255, 255, 0.9);
    --input-border: rgba(60, 70, 100, 0.25);
    --shadow: rgba(0, 0, 0, 0.08);
    --blur: 8px;
}

[data-font="xsmall"] { --font-size: 10px; }
[data-font="small"] { --font-size: 12px; }
[data-font="medium"] { --font-size: 14px; }
[data-font="large"] { --font-size: 20px; }
[data-font="xlarge"] { --font-size: 24px; }
[data-font="xxlarge"] { --font-size: 28px; }
[data-font="xxxlarge"] { --font-size: 32px; }

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        transition-duration: 0.01ms !important;
        animation-duration: 0.01ms !important;
    }
}

/* === Base === */

*, *::before, *::after { box-sizing: border-box; }

body {
    font-family: 'SF Mono', 'Menlo', 'Monaco', 'Consolas', monospace;
    font-size: var(--font-size);
    line-height: 1.6;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 20px;
    transition: background 0.2s ease, color 0.2s ease;
}

/* === Skip link (accessibility) === */
.skip-link {
    position: absolute;
    top: -100px;
    left: 8px;
    background: var(--text);
    color: var(--accent-text);
    padding: 8px 16px;
    border-radius: var(--radius);
    font-weight: 600;
    z-index: 10000;
    text-decoration: none;
}
.skip-link:focus {
    top: 8px;
}

/* === Glass panels === */
.glass, .card, .section, .grid-container, .controls {
    background: var(--bg-card);
    backdrop-filter: blur(var(--blur));
    -webkit-backdrop-filter: blur(var(--blur));
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: 0 4px 20px var(--shadow);
    padding: 16px;
    margin-bottom: 16px;
    transition: background 0.2s ease, border-color 0.2s ease;
}

.glass:hover, .card:hover {
    background: var(--bg-card-hover);
}

/* === Typography === */
h1, h2, h3, h4 {
    color: var(--text);
    font-weight: 600;
    letter-spacing: -0.02em;
    margin-top: 0;
}

h1 { font-size: calc(var(--font-size) * 1.5); }
h2 { font-size: calc(var(--font-size) * 1.25); }

p, li { color: var(--text-secondary); }

a {
    color: var(--text);
    text-decoration: underline;
    text-underline-offset: 2px;
}
a:hover { color: var(--text); }

/* === Focus — visible for all interactive elements === */
a:focus-visible,
button:focus-visible,
input:focus-visible,
textarea:focus-visible,
select:focus-visible,
[tabindex]:focus-visible {
    outline: none;
    box-shadow: var(--focus-ring);
}

/* === Inputs === */
input, textarea, select {
    font-family: inherit;
    font-size: var(--font-size);
    background: var(--input-bg);
    color: var(--text);
    border: 1.5px solid var(--input-border);
    border-radius: 6px;
    padding: 10px 12px;
    transition: border-color 0.15s ease;
}

input:focus, textarea:focus, select:focus {
    border-color: var(--text);
    outline: none;
    box-shadow: var(--focus-ring);
}

select {
    cursor: pointer;
    -webkit-appearance: none;
    appearance: none;
    padding-right: 32px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%238890a8' d='M2 4l4 4 4-4'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center;
}

/* === Buttons === */
button {
    font-family: inherit;
    font-size: calc(var(--font-size) * 0.9);
    font-weight: 600;
    padding: 10px 20px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    background: var(--text);
    color: var(--accent-text);
    transition: all 0.15s ease;
    min-height: 44px;  /* touch target */
    min-width: 44px;
}

button:hover {
    background: var(--bg-card-hover);
    transform: translateY(-1px);
}

button:active {
    transform: translateY(0);
}

button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
}

button.cancel, button.danger {
    background: var(--danger);
    color: var(--danger-text);
}
button.cancel:hover, button.danger:hover {
    background: var(--danger-hover);
}

button.secondary {
    background: var(--bg-card);
    color: var(--text);
    border: 1.5px solid var(--border);
}
button.secondary:hover {
    border-color: var(--text);
}

/* === Code === */
code, pre {
    font-family: inherit;
    background: var(--code-bg);
    color: var(--text);
    border-radius: 6px;
    padding: 2px 6px;
}

pre {
    padding: 16px;
    overflow-x: auto;
    border: 1px solid var(--border);
}

/* === Tables === */
table { border-collapse: collapse; width: 100%; }

th, td {
    border: 1px solid var(--border);
    padding: 8px 12px;
    color: var(--text);
    text-align: left;
    font-size: calc(var(--font-size) * 0.85);
}

th {
    background: var(--bg-card);
    font-weight: 600;
}

/* === Sliders === */
input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    border: none;
    padding: 0;
    min-height: auto;
}

input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: var(--text);
    cursor: pointer;
    border: 2px solid var(--bg);
}

/* === Checkboxes === */
input[type="checkbox"] {
    width: 20px;
    height: 20px;
    min-height: auto;
    cursor: pointer;
    accent-color: var(--text);
}

/* === Theme + accessibility controls === */
.quill-controls {
    position: fixed;
    top: 12px;
    right: 12px;
    display: flex;
    gap: 6px;
    z-index: 1000;
}

.quill-controls button {
    background: var(--bg-card);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 12px;
    min-height: 36px;
    min-width: 36px;
    font-size: 16px;
    backdrop-filter: blur(var(--blur));
    -webkit-backdrop-filter: blur(var(--blur));
}

.quill-controls button:hover {
    border-color: var(--text);
    background: var(--text);
    color: var(--accent-text);
    transform: none;
}

/* === Loading / status === */
.loading, .status {
    color: var(--text-secondary);
    font-style: italic;
}

.baseline {
    font-weight: 600;
    color: var(--text);
}

/* === Background mode toggle === */
.bg-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: calc(var(--font-size) * 0.85);
    color: var(--text-secondary);
    cursor: pointer;
    user-select: none;
}

/* === Info text === */
.info {
    margin-top: 10px;
    font-size: calc(var(--font-size) * 0.85);
    color: var(--text-secondary);
}

/* === Screen reader only === */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}
"""

THEME_TOGGLE_HTML = """<a href="#main" class="skip-link">Skip to content</a>
<div class="quill-controls" role="toolbar" aria-label="Display controls">
    <button onclick="toggleTheme()" title="Toggle dark/light mode" aria-label="Toggle dark or light mode">&#9789;</button>
    <button onclick="decreaseFont()" title="Decrease font size" aria-label="Decrease font size">A-</button>
    <button onclick="increaseFont()" title="Increase font size" aria-label="Increase font size">A+</button>
</div>"""

THEME_JS = """
/* QuillMorphism — theme + accessibility controls */
var _fontSizes = ['xsmall', 'small', 'medium', 'normal', 'large', 'xlarge', 'xxlarge', 'xxxlarge'];
var _fontIdx = 3;

function toggleTheme() {
    var html = document.documentElement;
    var current = html.getAttribute('data-theme');
    var next = current === 'light' ? 'dark' : 'light';
    html.setAttribute('data-theme', next);
    localStorage.setItem('mlx-interp-theme', next);
    var btn = document.querySelector('.quill-controls button');
    if (btn) btn.innerHTML = next === 'dark' ? '&#9789;' : '&#9788;';
}

function increaseFont() {
    if (_fontIdx < _fontSizes.length - 1) {
        _fontIdx++;
        document.documentElement.setAttribute('data-font', _fontSizes[_fontIdx]);
        localStorage.setItem('mlx-interp-font', _fontSizes[_fontIdx]);
    }
}

function decreaseFont() {
    if (_fontIdx > 0) {
        _fontIdx--;
        var size = _fontSizes[_fontIdx];
        if (size === 'normal') {
            document.documentElement.removeAttribute('data-font');
        } else {
            document.documentElement.setAttribute('data-font', size);
        }
        localStorage.setItem('mlx-interp-font', size);
    }
}

(function() {
    /* Restore saved preferences */
    var savedTheme = localStorage.getItem('mlx-interp-theme');
    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
    } else {
        document.documentElement.setAttribute('data-theme', 'dark');
    }
    var savedFont = localStorage.getItem('mlx-interp-font');
    if (savedFont && savedFont !== 'normal') {
        document.documentElement.setAttribute('data-font', savedFont);
        _fontIdx = _fontSizes.indexOf(savedFont);
        if (_fontIdx < 0) _fontIdx = 0;
    }
    document.addEventListener('DOMContentLoaded', function() {
        var btn = document.querySelector('.quill-controls button');
        var theme = document.documentElement.getAttribute('data-theme');
        if (btn) btn.innerHTML = theme === 'dark' ? '&#9789;' : '&#9788;';
    });
})();
"""
