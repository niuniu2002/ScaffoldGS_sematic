#!/usr/bin/env python3
"""
Parse metrics from training logs and eval outputs.

Supports:
- myvideo_iou_iter*.txt  (from eval_myvideo.py)
- outputs.log / training logs (from train.py)
"""

import argparse
import re
import sys
from pathlib import Path


def parse_myvideo_iou(path: str) -> dict:
    """Parse myvideo_iou_iter*.txt files."""
    text = Path(path).read_text()
    result = {
        "psnr": "NA",
        "bg_iou": "NA",
        "fg_iou": "NA",
        "miou": "NA",
        "ssim": "NA",
        "lpips": "NA",
    }
    # Mean PSNR   : 23.1889
    m = re.search(r"Mean PSNR\s*:\s*([0-9.]+)", text)
    if m:
        result["psnr"] = m.group(1)
    # Mean BG IoU : 0.9798
    m = re.search(r"Mean BG IoU\s*:\s*([0-9.]+)", text)
    if m:
        result["bg_iou"] = m.group(1)
    # Mean FG IoU : 0.6178
    m = re.search(r"Mean FG IoU\s*:\s*([0-9.]+)", text)
    if m:
        result["fg_iou"] = m.group(1)
    # Mean mIoU   : 0.7988 (79.88%)
    m = re.search(r"Mean mIoU\s*:\s*([0-9.]+)", text)
    if m:
        result["miou"] = m.group(1)
    return result


def parse_outputs_log(path: str) -> dict:
    """Parse outputs.log for PSNR, mIoU, SSIM, LPIPS."""
    text = Path(path).read_text()
    result = {
        "psnr": "NA",
        "bg_iou": "NA",
        "fg_iou": "NA",
        "miou": "NA",
        "ssim": "NA",
        "lpips": "NA",
    }

    # Find last occurrence of test PSNR / mIoU
    # [ITER 30000] 检阅 test: PSNR 24.63 dB, mIoU 0.7885
    matches = list(re.finditer(r"\[ITER\s+(\d+)\]\s+检阅\s+test:\s+PSNR\s+([0-9.]+)\s+dB,\s+mIoU\s+([0-9.]+)", text))
    if matches:
        last = matches[-1]
        result["psnr"] = last.group(2)
        result["miou"] = last.group(3)

    # SSIM :    0.6872310
    m = re.search(r"SSIM\s*:\s*([0-9.]+)", text)
    if m:
        result["ssim"] = m.group(1)
    # PSNR :   24.6281528
    m = re.search(r"PSNR\s*:\s*([0-9.]+)", text)
    if m:
        # Only overwrite if not already found from test line, preferring eval PSNR
        result["psnr"] = m.group(1)
    # LPIPS:    0.2961478
    m = re.search(r"LPIPS\s*:\s*([0-9.]+)", text)
    if m:
        result["lpips"] = m.group(1)

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse metrics from experiment outputs.")
    parser.add_argument("--iou_file", type=str, default=None, help="Path to myvideo_iou_iter*.txt")
    parser.add_argument("--log_file", type=str, default=None, help="Path to outputs.log")
    parser.add_argument("--tsv_out", type=str, default=None, help="Append results to TSV file")
    parser.add_argument("--exp_name", type=str, default="unknown", help="Experiment name")
    parser.add_argument("--iter", type=str, default="30000", help="Iteration number")
    args = parser.parse_args()

    merged = {
        "psnr": "NA",
        "bg_iou": "NA",
        "fg_iou": "NA",
        "miou": "NA",
        "ssim": "NA",
        "lpips": "NA",
    }

    if args.iou_file:
        iou_metrics = parse_myvideo_iou(args.iou_file)
        for k in merged:
            if iou_metrics.get(k) != "NA":
                merged[k] = iou_metrics[k]

    if args.log_file:
        log_metrics = parse_outputs_log(args.log_file)
        for k in merged:
            if log_metrics.get(k) != "NA":
                merged[k] = log_metrics[k]

    print(f"Experiment: {args.exp_name} @ iter {args.iter}")
    print(f"  PSNR   : {merged['psnr']}")
    print(f"  BG IoU : {merged['bg_iou']}")
    print(f"  FG IoU : {merged['fg_iou']}")
    print(f"  mIoU   : {merged['miou']}")
    print(f"  SSIM   : {merged['ssim']}")
    print(f"  LPIPS  : {merged['lpips']}")

    if args.tsv_out:
        line = f"\t{args.exp_name}\t{args.iter}\t{merged['psnr']}\t{merged['bg_iou']}\t{merged['fg_iou']}\t{merged['miou']}\t{merged['ssim']}\t{merged['lpips']}\tcompleted\t\n"
        with open(args.tsv_out, "a") as f:
            f.write(line)
        print(f"Appended to {args.tsv_out}")


if __name__ == "__main__":
    main()
