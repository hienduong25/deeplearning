"""Summarize evaluation CSV results across multiple experiments into a Markdown ablation table.
Calculates Mean and Standard Deviation for PSNR, SSIM, and RMSE.

Usage:
    python Shadow_Removal/tools/summarize_results.py --results_dir results --out results/ablation_summary.md
"""
import os
import glob
import argparse
import csv
import numpy as np
import yaml


def summarize_results(results_dir='results', out_file=None):
    csv_paths = sorted(glob.glob(os.path.join(results_dir, '*', 'metrics.csv')))
    if not csv_paths:
        csv_paths = sorted(glob.glob(os.path.join(results_dir, '*.csv')))

    if not csv_paths:
        print(f"[Warning] No metrics.csv files found in '{results_dir}'.")
        return ""

    print(f"Found {len(csv_paths)} evaluation CSV files to summarize:")

    rows = []
    for p in csv_paths:
        exp_folder = os.path.basename(os.path.dirname(p)) if os.path.basename(p) == 'metrics.csv' else os.path.splitext(os.path.basename(p))[0]

        psnrs, ssims, rmses, times = [], [], [], []
        with open(p, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                psnrs.append(float(r['psnr']))
                ssims.append(float(r['ssim']))
                rmses.append(float(r['rmse']))
                if 'time_sec' in r:
                    times.append(float(r['time_sec']))

        n = len(psnrs)
        if n == 0:
            continue

        mean_p, std_p = np.mean(psnrs), (np.std(psnrs, ddof=1) if n > 1 else 0.0)
        mean_s, std_s = np.mean(ssims), (np.std(ssims, ddof=1) if n > 1 else 0.0)
        mean_r, std_r = np.mean(rmses), (np.std(rmses, ddof=1) if n > 1 else 0.0)
        mean_t = np.mean(times) if times else 0.0

        rows.append({
            'name': exp_folder,
            'n_samples': n,
            'psnr': f"{mean_p:.2f} ± {std_p:.2f}",
            'ssim': f"{mean_s:.4f} ± {std_s:.4f}",
            'rmse': f"{mean_r:.2f} ± {std_r:.2f}",
            'speed': f"{mean_t:.3f}s"
        })

    # Render Markdown table
    md_lines = []
    md_lines.append("# BẢNG TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ (ABLATION SUMMARY)\n")
    md_lines.append("| Thí nghiệm / Mô hình | Số ảnh | PSNR (dB) ↑ | SSIM ↑ | RMSE ↓ | Giây / ảnh |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for row in rows:
        md_lines.append(f"| **{row['name']}** | {row['n_samples']} | {row['psnr']} | {row['ssim']} | {row['rmse']} | {row['speed']} |")

    table_md = "\n".join(md_lines)
    print("\n" + table_md + "\n")

    if out_file:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(table_md)
        print(f"Summary saved to: {out_file}")

    return table_md


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results', help='Directory containing results')
    parser.add_argument('--out', type=str, default='results/ablation_summary.md', help='Output Markdown file')
    args = parser.parse_args()
    summarize_results(args.results_dir, args.out)
