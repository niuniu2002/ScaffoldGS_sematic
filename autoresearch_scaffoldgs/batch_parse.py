#!/usr/bin/env python3
"""Batch parse existing experiment results and append to results.tsv."""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

OUTPUT_ROOT = Path("/mnt/data/liufengyang/data/Scaffold-GSLFY/output")
TSV_PATH = Path("/mnt/data/liufengyang/data/Scaffold-GSLFY/autoresearch_scaffoldgs/results.tsv")


def main():
    # Find all experiment directories with evaluation or log files
    exp_dirs = sorted(d for d in OUTPUT_ROOT.iterdir() if d.is_dir() and d.name.startswith("dronev4_2_"))

    # Read existing entries to avoid duplicates
    existing = set()
    if TSV_PATH.exists():
        with open(TSV_PATH) as f:
            next(f, None)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    existing.add((parts[1], parts[2]))  # (exp_name, iter)

    lines_added = 0
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name.replace("dronev4_2_", "")

        # Find all myvideo_iou_iter*.txt files
        iou_files = sorted(exp_dir.glob("myvideo_iou_iter*.txt"))
        log_file = exp_dir / "outputs.log"

        # If no iou files but has log, try to infer iter from log
        if not iou_files and log_file.exists():
            iou_files = [None]

        for iou_file in iou_files:
            if iou_file is not None:
                iter_str = iou_file.stem.replace("myvideo_iou_iter", "")
            else:
                iter_str = "30000"

            if (exp_name, iter_str) in existing:
                continue

            cmd = [
                sys.executable,
                "parse_metrics.py",
                "--exp_name", exp_name,
                "--iter", iter_str,
            ]
            if iou_file is not None:
                cmd += ["--iou_file", str(iou_file)]
            if log_file.exists():
                cmd += ["--log_file", str(log_file)]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
            if result.returncode != 0:
                print(f"[WARN] Failed to parse {exp_name} @ {iter_str}: {result.stderr.strip()}")
                continue

            # parse_metrics.py prints to stdout and optionally appends to TSV.
            # We manually prepend date and append.
            metrics = {}
            for line in result.stdout.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    metrics[key.strip().lower().replace(" ", "_")] = val.strip()

            date_str = datetime.now().strftime("%Y-%m-%d")
            line = (
                f"{date_str}\t{exp_name}\t{iter_str}\t"
                f"{metrics.get('psnr', 'NA')}\t"
                f"{metrics.get('bg_iou', 'NA')}\t"
                f"{metrics.get('fg_iou', 'NA')}\t"
                f"{metrics.get('miou', 'NA')}\t"
                f"{metrics.get('ssim', 'NA')}\t"
                f"{metrics.get('lpips', 'NA')}\t"
                f"completed\t\n"
            )
            with open(TSV_PATH, "a") as f:
                f.write(line)
            lines_added += 1
            print(f"[ADDED] {exp_name} @ {iter_str}")

    print(f"\nDone. Added {lines_added} new rows to {TSV_PATH}")


if __name__ == "__main__":
    main()
