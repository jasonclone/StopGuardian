#!/usr/bin/env python3
"""
rstats.py

Improved, robust version of your report generator.
- Recursively scans a results root (default: results/rl) for rl_episode_metrics CSVs.
- Groups by algorithm (first folder under root) and seed (next folder or any component containing "seed").
- Computes per-algorithm and per-seed statistics:
    * count, mean, sample std (ddof=1) of cumulative_reward (train / eval)
    * mean episode_length (train / eval)
    * convergence episode per seed (first training episode number where cumulative_reward >= threshold)
    * estimated total training time (hours) from episode_length and seconds_per_step
- Writes a nicely formatted plain-text report and optional CSV summary.
- Usage: python rstats.py --root results/rl --out performance_report.txt --seconds-per-step 0.5
"""

from __future__ import annotations
import argparse
import os
import glob
from collections import defaultdict, OrderedDict
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
import math
import textwrap
import datetime

CSV_GLOB_PATTERNS = [
    "**/rl_episode_metrics.csv",
    "**/rl_episode_metrics*.csv",
    "**/rl_episode_metrics.csv.gz",
]


def find_csv_files(root: str) -> List[str]:
    files = []
    for pat in CSV_GLOB_PATTERNS:
        files.extend(glob.glob(os.path.join(root, pat), recursive=True))
    return sorted(set(files))


def infer_algorithm_and_seed(root: str, filepath: str) -> Tuple[str, str]:
    rel = os.path.relpath(filepath, root)
    parts = rel.split(os.sep)
    algo = parts[0] if parts and parts[0] not in (".", "..") else "unknown_algo"
    seed = "unknown_seed"
    for p in parts[1:6]:
        if not p:
            continue
        pl = p.lower()
        if "seed" in pl or (pl.startswith("s") and pl[1:].isdigit()):
            seed = p
            break
    if seed == "unknown_seed" and len(parts) >= 2:
        seed = parts[1]
    return algo, seed


def read_csv_safe(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df["_source_file"] = path
        return df
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")
        return pd.DataFrame()


def compute_stats_for_series(series: pd.Series) -> Tuple[int, float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = int(s.size)
    if n == 0:
        return (0, float("nan"), float("nan"))
    if n == 1:
        return (1, float(s.iloc[0]), float("nan"))
    return (n, float(s.mean()), float(s.std(ddof=1)))


def compute_stats_grouped(df: pd.DataFrame) -> "OrderedDict[str, Tuple[int,float,float]]":
    results = OrderedDict()
    if df.empty or "mode" not in df.columns or "cumulative_reward" not in df.columns:
        return results
    df["mode"] = df["mode"].astype(str)
    for mode in sorted(df["mode"].unique()):
        series = df.loc[df["mode"] == mode, "cumulative_reward"]
        results[mode] = compute_stats_for_series(series)
    return results


def mean_episode_length_by_mode(df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    if df.empty or "mode" not in df.columns or "episode_length" not in df.columns:
        return out
    df["mode"] = df["mode"].astype(str)
    for mode in sorted(df["mode"].unique()):
        vals = pd.to_numeric(df.loc[df["mode"] == mode, "episode_length"], errors="coerce").dropna()
        out[mode] = float(vals.mean()) if vals.size > 0 else float("nan")
    return out


def compute_convergence_episode(df: pd.DataFrame, threshold: float, mode_filter: str = "train") -> Optional[int]:
    """
    Return the 1-based episode number of the first episode (in the filtered mode order)
    where cumulative_reward >= threshold. If no such episode exists, return None.
    """
    if df.empty or "mode" not in df.columns or "cumulative_reward" not in df.columns:
        return None
    mask = df["mode"].astype(str) == mode_filter
    sub = df.loc[mask].reset_index(drop=True)
    if sub.empty:
        return None
    rewards = pd.to_numeric(sub["cumulative_reward"], errors="coerce")
    for idx, r in enumerate(rewards):
        if pd.isna(r):
            continue
        if r >= threshold:
            # return 1-based episode number
            return int(idx + 1)
    return None


def total_training_time_hours(df: pd.DataFrame, seconds_per_step: float = 0.5, mode_filter: str = "train") -> float:
    if df.empty or "mode" not in df.columns or "episode_length" not in df.columns:
        return float("nan")
    
    TOTAL_EPISODES = 250
    EPISODE_LENGTH = 1000

    total_seconds = TOTAL_EPISODES * EPISODE_LENGTH * float(seconds_per_step)
    return total_seconds / 3600.0


def safe_fmt_mean_std(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "N/A"
    if math.isnan(std):
        return f"{mean:.6f} ± N/A"
    return f"{mean:.6f} ± {std:.6f}"


def format_bytes_as_mb(n_bytes: int) -> str:
    return f"{n_bytes / (1024.0 * 1024.0):.3f} MB"


def write_csv_summary(out_csv: str, per_algo_seed_stats: Dict[str, Dict[str, Dict[str, Tuple[int,float,float]]]]):
    rows = []
    for algo, seed_map in per_algo_seed_stats.items():
        for seed, stats in seed_map.items():
            if not stats:
                rows.append({"algorithm": algo, "seed": seed, "mode": "N/A", "count": 0, "mean": "", "std": ""})
                continue
            for mode, (count, mean, std) in stats.items():
                rows.append({"algorithm": algo, "seed": seed, "mode": mode, "count": count, "mean": mean, "std": std})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)


def build_report_text(
    root: str,
    algo_seed_files: Dict[str, Dict[str, List[str]]],
    seconds_per_step: float,
    convergence_threshold: float,
    rainbow_params: int,
    standard_params: int,
) -> str:
    lines: List[str] = []
    header = "RL Experiment Summary by Algorithm and Seed"
    lines.append(header)
    lines.append("=" * len(header))
    lines.append(f"Generated: {datetime.datetime.now().isoformat(sep=' ', timespec='seconds')}")
    lines.append(f"Root scanned: {root}")
    lines.append("")

    per_algo_stats: Dict[str, Dict[str, Tuple[int, float, float]]] = {}
    per_algo_seed_stats: Dict[str, Dict[str, Dict[str, Tuple[int, float, float]]]] = {}
    per_algo_episode_len: Dict[str, Dict[str, float]] = {}
    per_algo_seed_episode_len: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_algo_convergence: Dict[str, Dict[str, Optional[int]]] = {}
    per_algo_training_time: Dict[str, Dict[str, float]] = {}

    for algo, seed_map in sorted(algo_seed_files.items()):
        algo_seed_stats: Dict[str, Dict[str, Tuple[int, float, float]]] = {}
        algo_seed_ep_len: Dict[str, Dict[str, float]] = {}
        algo_seed_conv: Dict[str, Optional[int]] = {}
        algo_seed_time: Dict[str, float] = {}
        dfs_for_algo: List[pd.DataFrame] = []

        for seed, flist in sorted(seed_map.items()):
            dfs_seed: List[pd.DataFrame] = []
            for f in sorted(flist):
                df = read_csv_safe(f)
                if df.empty:
                    continue
                dfs_seed.append(df)
            if dfs_seed:
                df_seed_all = pd.concat(dfs_seed, ignore_index=True)
                dfs_for_algo.append(df_seed_all)
                algo_seed_stats[seed] = compute_stats_grouped(df_seed_all)
                algo_seed_ep_len[seed] = mean_episode_length_by_mode(df_seed_all)
                algo_seed_conv[seed] = compute_convergence_episode(df_seed_all, threshold=convergence_threshold, mode_filter="train")
                algo_seed_time[seed] = total_training_time_hours(df_seed_all, seconds_per_step=seconds_per_step, mode_filter="train")
            else:
                algo_seed_stats[seed] = {}
                algo_seed_ep_len[seed] = {}
                algo_seed_conv[seed] = None
                algo_seed_time[seed] = float("nan")

        if dfs_for_algo:
            df_algo_all = pd.concat(dfs_for_algo, ignore_index=True)
            per_algo_stats[algo] = compute_stats_grouped(df_algo_all)
            per_algo_episode_len[algo] = mean_episode_length_by_mode(df_algo_all)
            conv_list = [v for v in algo_seed_conv.values() if v is not None]
            per_algo_convergence[algo] = {
                "per_seed": algo_seed_conv,
                "median": int(np.median(conv_list)) if conv_list else None,
                "mean": float(np.mean(conv_list)) if conv_list else None,
            }
            per_algo_training_time[algo] = algo_seed_time
        else:
            per_algo_stats[algo] = {}
            per_algo_episode_len[algo] = {}
            per_algo_convergence[algo] = {"per_seed": algo_seed_conv, "median": None, "mean": None}
            per_algo_training_time[algo] = algo_seed_time

        per_algo_seed_stats[algo] = algo_seed_stats
        per_algo_seed_episode_len[algo] = algo_seed_ep_len
        per_algo_convergence[algo] = per_algo_convergence[algo]
        per_algo_training_time[algo] = algo_seed_time

    # Per-algorithm summary
    for algo in sorted(per_algo_stats.keys()):
        lines.append(f"Algorithm: {algo}")
        algo_stats = per_algo_stats[algo]
        if not algo_stats:
            lines.append("  No valid rows found for this algorithm.")
            lines.append("")
            continue
        for mode, (count, mean, std) in algo_stats.items():
            lines.append(f"  Mode: {mode}")
            lines.append(f"    Count: {count}")
            lines.append(f"    Mean cumulative_reward: {mean:.6f}" if not math.isnan(mean) else "    Mean cumulative_reward: N/A")
            lines.append(f"    Std (sample, ddof=1): {std:.6f}" if not math.isnan(std) else "    Std (sample, ddof=1): N/A")
            ep_len = per_algo_episode_len.get(algo, {}).get(mode, float("nan"))
            lines.append(f"    Mean episode_length: {ep_len:.3f}" if not math.isnan(ep_len) else "    Mean episode_length: N/A")
        lines.append("")
        lines.append(f"  Per-seed breakdown for {algo}:")
        seed_stats = per_algo_seed_stats.get(algo, {})
        for seed, stats in sorted(seed_stats.items()):
            lines.append(f"    Seed: {seed}")
            if not stats:
                lines.append("      No valid rows for this seed.")
            else:
                for mode, (count, mean, std) in stats.items():
                    lines.append(f"      {mode}: count={count}, mean={mean:.6f} std={std:.6f}" if not math.isnan(mean) else f"      {mode}: count={count}, mean=N/A std=N/A")
                seed_ep = per_algo_seed_episode_len.get(algo, {}).get(seed, {})
                for mode in sorted(seed_ep.keys()):
                    val = seed_ep[mode]
                    lines.append(f"        mean episode_length ({mode}): {val:.3f}" if not math.isnan(val) else f"        mean episode_length ({mode}): N/A")
                conv = per_algo_convergence[algo]["per_seed"].get(seed)
                lines.append(f"        convergence_episode (train, threshold={convergence_threshold}): {conv if conv is not None else 'N/A'}")
                t_hours = per_algo_training_time[algo].get(seed, float("nan"))
                lines.append(f"        estimated training time (hours): {t_hours:.4f}" if not math.isnan(t_hours) else "        estimated training time (hours): N/A")
            lines.append("")
        lines.append("")

    # Comparison table (text)
    lines.append("Comparison Summary (computed values where available)")
    lines.append("-" * 60)
    lines.append("")
    def algo_mean_std(algo: str, mode: str = "eval") -> Tuple[float, float]:
        stats = per_algo_stats.get(algo, {})
        if not stats or mode not in stats:
            return (float("nan"), float("nan"))
        _, mean, std = stats[mode]
        return (mean, std)
    def algo_episode_length(algo: str, mode: str = "train") -> float:
        return per_algo_episode_len.get(algo, {}).get(mode, float("nan"))
    r_mean_eval, r_std_eval = algo_mean_std("rainbow", "eval")
    s_mean_eval, s_std_eval = algo_mean_std("standard", "eval")
    r_ep_train = algo_episode_length("rainbow", "train")
    s_ep_train = algo_episode_length("standard", "train")
    r_conv_med = per_algo_convergence.get("rainbow", {}).get("median")
    s_conv_med = per_algo_convergence.get("standard", {}).get("median")
    r_train_hours = np.mean([v for v in per_algo_training_time.get("rainbow", {}).values() if not math.isnan(v)]) if per_algo_training_time.get("rainbow") else None
    s_train_hours = np.mean([v for v in per_algo_training_time.get("standard", {}).values() if not math.isnan(v)]) if per_algo_training_time.get("standard") else None

    # table block
    table = [
        ("Dimension",
         "Rainbow DQN (Primary)",
         "Standard DQN (Baseline)"),
        ("Mean Reward (eval)",
         safe_fmt_mean_std(r_mean_eval, r_std_eval),
         safe_fmt_mean_std(s_mean_eval, s_std_eval)),
        ("Episode Length (mean, train)",
         f"{r_ep_train:.3f}" if not math.isnan(r_ep_train) else "N/A",
         f"{s_ep_train:.3f}" if not math.isnan(s_ep_train) else "N/A"),
        ("Convergence Episodes (median across seeds)",
         str(r_conv_med) if r_conv_med is not None else "N/A",
         str(s_conv_med) if s_conv_med is not None else "N/A"),
        ("Estimated Training Time (avg hours across seeds)",
         f"{r_train_hours:.3f} h" if r_train_hours is not None and not math.isnan(r_train_hours) else "N/A",
         f"{s_train_hours:.3f} h" if s_train_hours is not None and not math.isnan(s_train_hours) else "N/A"),
        ("Model Complexity (params)",
         f"{rainbow_params:,} (~{rainbow_params/1e6:.3f}M)",
         f"{standard_params:,} (~{standard_params/1e6:.3f}M)"),
        ("File Size (float32)",
         format_bytes_as_mb(rainbow_params * 4),
         format_bytes_as_mb(standard_params * 4)),
    ]
    # Render table with aligned columns
    col_widths = [max(len(str(row[c])) for row in table) for c in range(3)]
    for row in table:
        lines.append(f"{row[0]:<{col_widths[0]}}  |  {row[1]:<{col_widths[1]}}  |  {row[2]:<{col_widths[2]}}")
    lines.append("")
    lines.append("Notes:")
    lines.append(" - Convergence episode: first training episode number where cumulative_reward <= threshold (episodes until crossing threshold).")
    lines.append(" - Training time estimated from episode_length * seconds_per_step; measure seconds_per_step for accuracy.")
    lines.append(" - File sizes assume float32 (4 bytes per parameter).")
    lines.append("")
    lines.append("End of report.")
    return "\n".join(lines), per_algo_seed_stats


def main():
    parser = argparse.ArgumentParser(description="Aggregate RL episode metrics by algorithm and seed and produce comparison report")
    parser.add_argument("--root", "-r", default="results/rl", help="Root results directory to scan")
    parser.add_argument("--out", "-o", default="performance_report.txt", help="Output text file path")
    parser.add_argument("--csv-out", default=None, help="Optional CSV summary output path")
    parser.add_argument("--seconds-per-step", type=float, default=0.14786, help="Seconds per simulation step (used to estimate training time). Default 0.14786, colected from hyperpaream tuning   avg_env_step_ms: 96.7172492002137 + avg_train_step_ms: 51.15676169807557,")
    parser.add_argument("--convergence-threshold", type=float, default=-280.0, help="Cumulative reward threshold used to define convergence (default -300.0)")
    parser.add_argument("--rainbow-params", type=int, default=414514, help="Trainable parameter count for Rainbow (override)")
    parser.add_argument("--standard-params", type=int, default=74218, help="Trainable parameter count for Standard DQN (override)")
    args = parser.parse_args()

    files = find_csv_files(args.root)
    if not files:
        print(f"No CSV files found under {args.root}")
        return 1

    algo_seed_files: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for f in files:
        algo, seed = infer_algorithm_and_seed(args.root, f)
        algo_seed_files[algo][seed].append(f)

    report_text, per_algo_seed_stats = build_report_text(
        args.root,
        algo_seed_files,
        seconds_per_step=args.seconds_per_step,
        convergence_threshold=args.convergence_threshold,
        rainbow_params=args.rainbow_params,
        standard_params=args.standard_params,
    )

    try:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report_text)
    except Exception as e:
        print(f"Failed to write report to {args.out}: {e}")
        return 1

    if args.csv_out:
        try:
            write_csv_summary(args.csv_out, per_algo_seed_stats)
        except Exception as e:
            print(f"Failed to write CSV summary to {args.csv_out}: {e}")

    print(f"Report written to {args.out}")
    if args.csv_out:
        print(f"CSV summary written to {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
