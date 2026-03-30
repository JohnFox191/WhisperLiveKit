#!/usr/bin/env python3
"""Run wlk bench across all backend/model/policy combos and compile results.

Usage:
    python scripts/run_all_benches.py
    python scripts/run_all_benches.py --output-dir results/rtx5090
    python scripts/run_all_benches.py --plot-only results/rtx5090
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class BenchCombo:
    backend: str
    model: str
    policy: Optional[str] = None
    label: str = ""
    color: str = "#4a9eff"
    marker: str = "o"
    size: int = 100

    def __post_init__(self):
        if not self.label:
            parts = [self.backend, self.model]
            if self.policy:
                parts.append("LA" if self.policy == "localagreement" else "SS")
            self.label = " ".join(parts)

    @property
    def filename(self) -> str:
        parts = [self.backend, self.model.replace(":", "-")]
        if self.policy:
            parts.append(self.policy)
        return "_".join(parts) + ".json"


COMBOS = [
    # faster-whisper base
    BenchCombo("faster-whisper", "base", "localagreement",
               "fw LA base", "#4a9eff", "o", 100),
    BenchCombo("faster-whisper", "base", "simulstreaming",
               "fw SS base", "#4a9eff", "s", 100),
    # faster-whisper small
    BenchCombo("faster-whisper", "small", "localagreement",
               "fw LA small", "#4a9eff", "o", 180),
    BenchCombo("faster-whisper", "small", "simulstreaming",
               "fw SS small", "#4a9eff", "s", 180),
    # faster-whisper large-v3
    BenchCombo("faster-whisper", "large-v3", "localagreement",
               "fw LA large-v3", "#4a9eff", "o", 300),
    BenchCombo("faster-whisper", "large-v3", "simulstreaming",
               "fw SS large-v3", "#4a9eff", "s", 300),
    # faster-whisper large-v3-turbo
    BenchCombo("faster-whisper", "large-v3-turbo", "localagreement",
               "fw LA turbo", "#4a9eff", "o", 250),
    BenchCombo("faster-whisper", "large-v3-turbo", "simulstreaming",
               "fw SS turbo", "#4a9eff", "s", 250),
    # voxtral HF
    BenchCombo("voxtral", "base", None,
               "voxtral hf", "#f5a623", "D", 300),
    # qwen3 chunked
    BenchCombo("qwen3", "qwen3:0.6b", None,
               "qwen3 0.6B", "#e056a0", "h", 100),
    BenchCombo("qwen3", "qwen3:1.7b", None,
               "qwen3 1.7B", "#e056a0", "h", 220),
    # qwen3-simul-kv
    BenchCombo("qwen3-simul-kv", "qwen3:0.6b", None,
               "qwen3-kv 0.6B", "#e056a0", "s", 100),
    BenchCombo("qwen3-simul-kv", "qwen3:1.7b", None,
               "qwen3-kv 1.7B", "#e056a0", "s", 220),
]


def run_bench(combo: BenchCombo, output_dir: Path, language: str = "en") -> Optional[dict]:
    """Run a single wlk bench and return parsed JSON results."""
    json_path = (output_dir / combo.filename).resolve()
    # Use the venv wlk entry point to ensure CUDA torch is available
    venv_dir = Path(__file__).resolve().parent.parent / ".venv" / "Scripts"
    wlk_exe = str(venv_dir / "wlk.exe")
    if not Path(wlk_exe).exists():
        wlk_exe = "wlk"
    cmd = [
        wlk_exe, "bench",
        "--backend", combo.backend,
        "--model", combo.model,
        "--languages", language,
        "--json", str(json_path),
    ]
    if combo.policy:
        cmd += ["--backend-policy", combo.policy]

    print(f"\n{'='*70}")
    print(f"  {combo.label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    print(f"  JSON target: {json_path}", flush=True)

    t0 = time.time()
    # Stream output live so we can see progress and errors
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    print(f"\n  Exit code: {result.returncode}, elapsed: {elapsed:.1f}s", flush=True)

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        return None

    if not json_path.exists():
        print(f"  FAILED (no JSON output at {json_path})")
        return None

    with open(json_path) as f:
        data = json.load(f)

    print(f"  Completed in {elapsed:.0f}s")
    return data


def extract_summary(data: dict, combo: BenchCombo) -> dict:
    """Extract key metrics from a bench JSON result."""
    results = data.get("results", [])
    if not results:
        return None

    # Compute weighted WER
    total_ref_words = 0
    total_errors = 0
    total_audio_s = 0
    total_infer_s = 0
    categories = {}

    for r in results:
        ref_words = r.get("ref_words", 0)
        errors = r.get("substitutions", 0) + r.get("insertions", 0) + r.get("deletions", 0)
        total_ref_words += ref_words
        total_errors += errors
        dur = r.get("duration_s", 0)
        total_audio_s += dur

        # Per-inference timing
        durations = r.get("transcription_durations", [])
        if durations:
            total_infer_s += sum(durations)

        # Category
        for tag in r.get("tags", []):
            if tag not in categories:
                categories[tag] = {"ref_words": 0, "errors": 0}
            categories[tag]["ref_words"] += ref_words
            categories[tag]["errors"] += errors

    weighted_wer = total_errors / max(total_ref_words, 1)

    # Average WER (macro)
    wers = []
    for r in results:
        rw = r.get("ref_words", 0)
        if rw > 0:
            e = r.get("substitutions", 0) + r.get("insertions", 0) + r.get("deletions", 0)
            wers.append(e / rw)
    avg_wer = sum(wers) / len(wers) if wers else 0

    # RTF
    rtf = total_infer_s / total_audio_s if total_audio_s > 0 else 0

    # Latency
    latencies = []
    for r in results:
        lat = r.get("avg_latency_ms")
        if lat is not None:
            latencies.append(lat)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    p95_latencies = []
    for r in results:
        p95 = r.get("p95_latency_ms")
        if p95 is not None:
            p95_latencies.append(p95)
    p95_latency = max(p95_latencies) if p95_latencies else 0

    # Category WERs
    cat_wer = {}
    for cat, vals in categories.items():
        if vals["ref_words"] > 0:
            cat_wer[cat] = vals["errors"] / vals["ref_words"]

    return {
        "label": combo.label,
        "backend": combo.backend,
        "model": combo.model,
        "policy": combo.policy or "default",
        "color": combo.color,
        "marker": combo.marker,
        "size": combo.size,
        "weighted_wer": round(weighted_wer * 100, 1),
        "avg_wer": round(avg_wer * 100, 1),
        "rtf": round(rtf, 3),
        "avg_latency_ms": round(avg_latency),
        "p95_latency_ms": round(p95_latency),
        "n_samples": len(results),
        "total_audio_s": round(total_audio_s, 1),
        "category_wer": {k: round(v * 100, 1) for k, v in cat_wer.items()},
    }


def print_tables(summaries: List[dict]):
    """Print comparison tables."""
    if not summaries:
        print("No results to display.")
        return

    print(f"\n{'='*100}")
    print("  BENCHMARK COMPARISON")
    print(f"{'='*100}\n")

    # Main table
    header = f"{'Label':<22} {'W-WER':>6} {'A-WER':>6} {'RTF':>7} {'Lat(avg)':>9} {'Lat(p95)':>9} {'Samples':>8}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(f"{s['label']:<22} {s['weighted_wer']:>5.1f}% {s['avg_wer']:>5.1f}% {s['rtf']:>6.3f}x {s['avg_latency_ms']:>7d}ms {s['p95_latency_ms']:>7d}ms {s['n_samples']:>8d}")

    # Category table
    all_cats = set()
    for s in summaries:
        all_cats.update(s.get("category_wer", {}).keys())
    all_cats = sorted(all_cats)

    if all_cats:
        print(f"\n{'Label':<22}", end="")
        for cat in all_cats:
            print(f" {cat:>10}", end="")
        print()
        print("-" * (22 + 11 * len(all_cats)))
        for s in summaries:
            print(f"{s['label']:<22}", end="")
            for cat in all_cats:
                wer = s.get("category_wer", {}).get(cat)
                if wer is not None:
                    print(f" {wer:>9.1f}%", end="")
                else:
                    print(f" {'N/A':>10}", end="")
            print()

    # Speed ranking
    print(f"\n{'Label':<22} {'RTF':>7} {'Real-time?':>11}")
    print("-" * 42)
    for s in sorted(summaries, key=lambda x: x["rtf"]):
        rt = "Yes" if s["rtf"] < 1.0 else ("Borderline" if s["rtf"] < 1.1 else "No")
        print(f"{s['label']:<22} {s['rtf']:>6.3f}x {rt:>11}")


def generate_scatter(summaries: List[dict], output_path: str):
    """Generate scatter plot of RTF vs WER."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not installed, skipping plot generation")
        return

    fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")
    ax.set_facecolor("#fafafa")

    if not summaries:
        return

    xmax = max(s["rtf"] for s in summaries) * 1.15
    ymax = max(s["weighted_wer"] for s in summaries) * 1.15 + 1
    xmax = max(xmax, 1.15)
    ymax = max(ymax, 8)

    # Sweet spot
    sweet_x = min(1.0, xmax * 0.85)
    sweet_y = min(15, ymax * 0.45)
    rect = plt.Rectangle((0, 0), sweet_x, sweet_y, alpha=0.07, color="#4ecca3",
                          zorder=0, linewidth=0)
    ax.add_patch(rect)
    ax.text(sweet_x - 0.005, sweet_y - 0.15, "sweet spot", ha="right", va="top",
            fontsize=10, color="#2ecc71", fontstyle="italic", fontweight="bold", alpha=0.5)

    # Real-time line
    ax.axvline(x=1.0, color="#e94560", linestyle="--", linewidth=1.5, alpha=0.4, zorder=1)
    ax.text(1.02, ymax * 0.97, "real-time\nlimit", fontsize=8, color="#e94560",
            va="top", alpha=0.6)

    # Plot points
    for s in summaries:
        ax.scatter(s["rtf"], s["weighted_wer"], c=s["color"], marker=s["marker"],
                   s=s["size"], edgecolors="white", linewidths=1.0, zorder=5, alpha=0.85)
        ax.annotate(s["label"], (s["rtf"], s["weighted_wer"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color="#333333", fontweight="medium")

    ax.set_xlim(left=-0.01, right=xmax)
    ax.set_ylim(bottom=0, top=ymax)
    ax.set_xlabel("RTF (lower = faster)", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel("Weighted WER % (lower = more accurate)", fontsize=13, fontweight="bold", labelpad=8)
    ax.grid(True, alpha=0.15, linestyle="-", color="#cccccc")
    ax.tick_params(labelsize=10)

    ax.set_title(f"Speed vs Accuracy — {len(summaries)} configs, wlk bench (quick)",
                 fontsize=14, fontweight="bold", pad=12)

    # Legend — backends
    backend_handles = []
    seen = set()
    for s in summaries:
        if s["backend"] not in seen:
            seen.add(s["backend"])
            backend_handles.append(mpatches.Patch(color=s["color"], label=s["backend"]))

    # Legend — markers
    marker_map = {"o": "LocalAgreement", "s": "SimulStreaming", "D": "Native streaming",
                  "h": "Batch + aligner"}
    active = set(s["marker"] for s in summaries)
    shape_handles = [
        Line2D([0], [0], marker=m, color="#888", label=lbl,
               markerfacecolor="#888", markersize=8, linestyle="None")
        for m, lbl in marker_map.items() if m in active
    ]

    leg1 = ax.legend(handles=backend_handles, loc="upper left", fontsize=9,
                     framealpha=0.95, edgecolor="#ddd", title="Backend", title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, loc="lower right", fontsize=8,
              framealpha=0.95, edgecolor="#ddd", ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    print(f"\nSaved scatter plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run all wlk bench combos and compile results.")
    parser.add_argument("--output-dir", "-o", default="bench_results",
                        help="Output directory for JSON files (default: bench_results)")
    parser.add_argument("--language", default="en", help="Language (default: en)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip running, just compile existing results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        print(f"Running {len(COMBOS)} benchmark combos...")
        print(f"Output: {output_dir.absolute()}\n")

        t_total = time.time()
        for i, combo in enumerate(COMBOS):
            print(f"\n[{i+1}/{len(COMBOS)}]", end="")
            run_bench(combo, output_dir, args.language)
        total_time = time.time() - t_total
        print(f"\n\nAll benchmarks completed in {total_time:.0f}s ({total_time/60:.1f}min)")

    # Compile results
    summaries = []
    for combo in COMBOS:
        json_path = output_dir / combo.filename
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            summary = extract_summary(data, combo)
            if summary:
                summaries.append(summary)

    # Save compiled summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    # Print tables
    print_tables(summaries)

    # Generate scatter plot
    plot_path = str(output_dir / "scatter.png")
    generate_scatter(summaries, plot_path)


if __name__ == "__main__":
    main()
