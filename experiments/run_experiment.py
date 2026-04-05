#!/usr/bin/env python3
"""
Sealed experiment runner.

Creates a timestamped run directory under runs/ with:
  config.json   — frozen inputs (model, seed, git SHA, args)
  results.json  — frozen outputs (written by the experiment)
  log.txt       — stdout/stderr capture
  SUMMARY.md    — auto-generated human-readable summary

Usage:
    python run_experiment.py --model gemma4 --suite rigorous
    python run_experiment.py --model qwen2.5 --suite sycophancy_3way
    python run_experiment.py --model gemma4 --suite rigorous --seed 42 --tag "hypothesis-H1"
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

SUITE_MAP = {
    "rigorous": "rigorous_suite.py",
    "sycophancy_3way": "sycophancy_3way.py",
    "sycophancy_proper": "sycophancy_proper.py",
}

MODEL_REGISTRY = {
    "qwen2.5": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "qwen3": "mlx-community/Qwen3-14B-4bit",
    "gemma3": "mlx-community/gemma-3-4b-it-4bit",
    "gemma4": "unsloth/gemma-4-E4B-it-UD-MLX-4bit",
}


def get_git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()[:8]
    except Exception:
        return "unknown"


def create_run_dir(model_key, suite, tag=None):
    ts = datetime.now().strftime("%Y-%m-%d-%H%M")
    name = f"{ts}-{suite}-{model_key}"
    if tag:
        name += f"-{tag}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_config(run_dir, model_key, suite, seed, tag, extra_args):
    config = {
        "timestamp": datetime.now().isoformat(),
        "model_key": model_key,
        "model_name": MODEL_REGISTRY[model_key],
        "suite": suite,
        "seed": seed,
        "tag": tag,
        "git_sha": get_git_sha(),
        "extra_args": extra_args,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    return config


def generate_summary(run_dir, config):
    """Auto-generate SUMMARY.md from config and results."""
    results_file = run_dir / "results.json"
    lines = [
        f"# Run: {run_dir.name}",
        "",
        f"**Model:** {config['model_name']}",
        f"**Suite:** {config['suite']}",
        f"**Seed:** {config['seed']}",
        f"**Git SHA:** {config['git_sha']}",
        f"**Timestamp:** {config['timestamp']}",
    ]
    if config.get("tag"):
        lines.append(f"**Tag:** {config['tag']}")
    lines.append("")

    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        lines.append("## Results")
        lines.append("")
        lines.append("```json")
        # Print top-level keys and their types/values
        for k, v in results.items():
            if isinstance(v, (int, float, str, bool)):
                lines.append(f"  {k}: {v}")
            elif isinstance(v, dict):
                lines.append(f"  {k}: {{...}} ({len(v)} keys)")
            elif isinstance(v, list):
                lines.append(f"  {k}: [...] ({len(v)} items)")
        lines.append("```")
    else:
        lines.append("## Results")
        lines.append("")
        lines.append("_No results.json found. Check log.txt for errors._")

    lines.append("")
    with open(run_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(lines))


def append_to_runs_index(run_dir, config, success):
    """Append one line to RUNS.md."""
    runs_md = REPO_ROOT / "RUNS.md"
    status = "ok" if success else "FAIL"
    date = datetime.now().strftime("%Y-%m-%d")
    line = (
        f"| {date} | `{run_dir.name}` | {config['model_key']} "
        f"| {config['suite']} | {status} |"
    )
    with open(runs_md, "a") as f:
        f.write(line + "\n")


def run(model_key, suite, seed, tag, extra_args):
    script = EXPERIMENTS_DIR / SUITE_MAP[suite]
    run_dir = create_run_dir(model_key, suite, tag)
    config = write_config(run_dir, model_key, suite, seed, tag, extra_args)

    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"{'='*60}")
    print(f"RUN: {run_dir.name}")
    print(f"Model: {config['model_name']}")
    print(f"Suite: {suite} → {script.name}")
    print(f"Seed: {seed}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(script),
        "--models", model_key,
        "--seed", str(seed),
        "--output-dir", str(results_dir),
    ] + extra_args

    log_file = run_dir / "log.txt"
    with open(log_file, "w") as log:
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write(f"Started: {datetime.now().isoformat()}\n")
        log.write("=" * 60 + "\n\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(EXPERIMENTS_DIR),
        )

        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()

        proc.wait()
        log.write(f"\n{'='*60}\n")
        log.write(f"Finished: {datetime.now().isoformat()}\n")
        log.write(f"Exit code: {proc.returncode}\n")

    # Collect results — look for JSON files the suite produced
    result_files = list(results_dir.glob("*.json"))
    if result_files:
        # Merge all result JSONs into one results.json at run root
        merged = {}
        for rf in result_files:
            with open(rf) as f:
                merged[rf.stem] = json.load(f)
        with open(run_dir / "results.json", "w") as f:
            json.dump(merged, f, indent=2)

    success = proc.returncode == 0
    generate_summary(run_dir, config)
    append_to_runs_index(run_dir, config, success)

    if success:
        print(f"\n✓ Run complete: {run_dir}")
    else:
        print(f"\n✗ Run failed (exit {proc.returncode}). See {log_file}")

    return success


def main():
    parser = argparse.ArgumentParser(description="Sealed experiment runner")
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--suite", required=True, choices=SUITE_MAP.keys())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default=None, help="Optional tag (e.g. hypothesis-H1)")
    args, extra = parser.parse_known_args()

    run(args.model, args.suite, args.seed, args.tag, extra)


if __name__ == "__main__":
    main()
