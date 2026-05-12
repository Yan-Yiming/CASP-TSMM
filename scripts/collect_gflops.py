#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import sys


def load_results(path):
    files = []
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "results_*.json")))
    else:
        files = [path]
    for filename in files:
        with open(filename, "r", encoding="utf-8") as f:
            yield filename, json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Collect TSMM GFLOPS from benchmark JSON files")
    parser.add_argument("paths", nargs="+", help="result JSON files or directories")
    parser.add_argument("--csv", default="", help="CSV output path")
    parser.add_argument("--json", default="", help="summary JSON output path")
    args = parser.parse_args()

    rows = []
    for path in args.paths:
        for filename, data in load_results(path):
            layout = data.get("layout", "")
            timestamp = data.get("timestamp", "")
            blas = data.get("blas", "")
            threads = data.get("n_threads", "")
            for problem in data.get("problems", []):
                for impl in problem.get("impls", []):
                    rows.append({
                        "file": filename,
                        "timestamp": timestamp,
                        "layout": layout,
                        "blas": blas,
                        "threads": threads,
                        "problem": problem.get("name", ""),
                        "m": problem.get("m", ""),
                        "n": problem.get("n", ""),
                        "k": problem.get("k", ""),
                        "required": problem.get("required", ""),
                        "impl": impl.get("name", ""),
                        "time_ms": impl.get("time_ms", ""),
                        "gflops": impl.get("gflops", ""),
                        "speedup": impl.get("speedup", ""),
                        "correct": impl.get("correct", ""),
                    })

    fieldnames = [
        "file", "timestamp", "layout", "blas", "threads",
        "problem", "m", "n", "k", "required",
        "impl", "time_ms", "gflops", "speedup", "correct",
    ]

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = {}
    for row in rows:
        if row["correct"] is not True:
            continue
        key = f'{row["layout"]}/{row["problem"]}'
        current = best.get(key)
        if current is None or float(row["gflops"]) > float(current["gflops"]):
            best[key] = row

    summary = {"rows": len(rows), "best_gflops": best}
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
