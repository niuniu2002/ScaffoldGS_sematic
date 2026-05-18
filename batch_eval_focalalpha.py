#!/usr/bin/env python3
"""
Batch evaluate all focal_alpha ablation experiments on both:
1. Original dataset (dronev4_2) masks
2. myvideo dataset masks
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(__file__))
from eval_myvideo import evaluate_on_myvideo

OUTPUT_ROOT = "/mnt/data/liufengyang/data/Scaffold-GSLFY/output"
EXP_PATTERN = "dronev4_2_focalalpha0.*"

DATASETS = {
    "dronev4_2": "/mnt/data/liufengyang/data/Scaffold-GSLFY/data/dronev4_2",
    "myvideo": "/mnt/data/liufengyang/data/myvideo",
}


def main():
    exp_dirs = sorted(glob.glob(os.path.join(OUTPUT_ROOT, EXP_PATTERN)))
    print(f"Found {len(exp_dirs)} experiments to evaluate")
    print("=" * 60)

    summary = []

    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        cfg_path = os.path.join(exp_dir, "cfg_args")

        # Read cfg_args for white_background, appearance_dim, use_per_gaussian_seg, num_classes
        white_background = False
        appearance_dim = 32
        use_per_gaussian_seg = False
        num_classes = 1
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                content = f.read()
                if "white_background=True" in content:
                    white_background = True
                if "use_per_gaussian_seg=True" in content:
                    use_per_gaussian_seg = True
                import re

                m = re.search(r"appearance_dim=(\d+)", content)
                if m:
                    appearance_dim = int(m.group(1))
                m = re.search(r"num_classes=(\d+)", content)
                if m:
                    num_classes = int(m.group(1))

        print(f"\n>>> {exp_name}")
        print(f"    white_background={white_background}, appearance_dim={appearance_dim}, use_per_gaussian_seg={use_per_gaussian_seg}, num_classes={num_classes}")

        exp_results = {"exp": exp_name}

        for dataset_name, source_path in DATASETS.items():
            print(f"    Evaluating on {dataset_name} ...")
            try:
                mean_miou, mean_fg, mean_bg, mean_psnr = evaluate_on_myvideo(
                    model_path=exp_dir,
                    source_path=source_path,
                    iteration=30000,
                    white_background=white_background,
                    appearance_dim=appearance_dim,
                    use_per_gaussian_seg=use_per_gaussian_seg,
                    num_classes=num_classes,
                )
                exp_results[f"{dataset_name}_miou"] = mean_miou
                exp_results[f"{dataset_name}_psnr"] = mean_psnr
                print(f"    {dataset_name} => mIoU={mean_miou:.4f}, PSNR={mean_psnr:.2f}")
            except Exception as e:
                print(f"    {dataset_name} => FAILED: {e}")
                exp_results[f"{dataset_name}_miou"] = None
                exp_results[f"{dataset_name}_psnr"] = None

        summary.append(exp_results)

    # Final summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<40} {'dronev4_2 mIoU':>14} {'myvideo mIoU':>14} {'dronev4_2 PSNR':>14} {'myvideo PSNR':>14}")
    print("-" * 80)
    for r in summary:
        d_miou = f"{r['dronev4_2_miou']:.4f}" if r["dronev4_2_miou"] is not None else "FAIL"
        m_miou = f"{r['myvideo_miou']:.4f}" if r["myvideo_miou"] is not None else "FAIL"
        d_psnr = f"{r['dronev4_2_psnr']:.2f}" if r["dronev4_2_psnr"] is not None else "FAIL"
        m_psnr = f"{r['myvideo_psnr']:.2f}" if r["myvideo_psnr"] is not None else "FAIL"
        print(f"{r['exp']:<40} {d_miou:>14} {m_miou:>14} {d_psnr:>14} {m_psnr:>14}")
    print("=" * 80)

    # Save summary to file
    summary_file = os.path.join(OUTPUT_ROOT, "focalalpha_ablation_summary.txt")
    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Focal Alpha Ablation Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Experiment':<40} {'dronev4_2 mIoU':>14} {'myvideo mIoU':>14} {'dronev4_2 PSNR':>14} {'myvideo PSNR':>14}\n")
        f.write("-" * 80 + "\n")
        for r in summary:
            d_miou = f"{r['dronev4_2_miou']:.4f}" if r["dronev4_2_miou"] is not None else "FAIL"
            m_miou = f"{r['myvideo_miou']:.4f}" if r["myvideo_miou"] is not None else "FAIL"
            d_psnr = f"{r['dronev4_2_psnr']:.2f}" if r["dronev4_2_psnr"] is not None else "FAIL"
            m_psnr = f"{r['myvideo_psnr']:.2f}" if r["myvideo_psnr"] is not None else "FAIL"
            f.write(f"{r['exp']:<40} {d_miou:>14} {m_miou:>14} {d_psnr:>14} {m_psnr:>14}\n")
        f.write("=" * 80 + "\n")
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
