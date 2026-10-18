"""Independent Testing and Evaluation Script for Document Shadow Removal.
Performs:
  - Network instantiation from YAML
  - Weight loading (.pth)
  - Reflection padding to multiple of 8
  - Inference (CPU or GPU)
  - Exact metric evaluation (PSNR, SSIM, RMSE) via basicsr.metrics
  - CSV output and image saving

Usage:
    python Shadow_Removal/test.py --opt Shadow_Removal/Options/1_Restormer_baseline.yml --weights experiments/1_Restormer_baseline/models/net_g_latest.pth --split val --out results/exp1_val
"""
import os
import sys
sys.path.insert(0, '.')
import time
import argparse
import yaml
import csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.models.archs import define_network
from basicsr.utils import img2tensor, tensor2img
from basicsr.metrics import calculate_psnr, calculate_ssim, calculate_rmse


def test_pipeline(opt_path, weights_path, split='val', data_dir=None, out_dir=None):
    with open(opt_path, 'r') as f:
        opt = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print(f"RUNNING EVALUATION ON [{device}]")
    print(f"Config   : {opt_path}")
    print(f"Weights  : {weights_path}")
    print(f"Split    : {split}")
    print("=" * 70)

    # 1. Instantiate network
    net_opt = dict(opt['network_g'])
    net = define_network(net_opt)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Network created: {opt['network_g']['type']} ({n_params:,} parameters)")

    # 2. Load weights
    assert os.path.exists(weights_path), f"Weights file not found: {weights_path}"
    checkpoint = torch.load(weights_path, map_location=device)
    if 'params' in checkpoint:
        net.load_state_dict(checkpoint['params'], strict=True)
    elif 'params_ema' in checkpoint:
        net.load_state_dict(checkpoint['params_ema'], strict=True)
    else:
        net.load_state_dict(checkpoint, strict=True)

    net.to(device)
    net.eval()

    # 3. Resolve dataset paths
    if data_dir is None:
        if split == 'val' and 'val' in opt['datasets']:
            lq_dir = opt['datasets']['val']['dataroot_lq']
            gt_dir = opt['datasets']['val']['dataroot_gt']
        else:
            # Fallback to local RDD_data
            lq_dir = f"./RDD_data/{split}/img"
            gt_dir = f"./RDD_data/{split}/gt"
    else:
        lq_dir = os.path.join(data_dir, split, 'img')
        gt_dir = os.path.join(data_dir, split, 'gt')

    if not os.path.isdir(lq_dir) or not os.path.isdir(gt_dir):
        # Fallback check for smoke data
        if os.path.isdir(f"./RDD_data/smoke/{split}/img"):
            lq_dir = f"./RDD_data/smoke/{split}/img"
            gt_dir = f"./RDD_data/smoke/{split}/gt"
            print(f"[Notice] Using smoke dataset at: {lq_dir}")
        else:
            raise FileNotFoundError(f"Dataset folders not found: {lq_dir} / {gt_dir}")

    # Resolve output directory
    if out_dir is None:
        exp_name = opt.get('name', 'evaluation')
        out_dir = os.path.join('results', f"{exp_name}_{split}")
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, 'restored')
    os.makedirs(images_dir, exist_ok=True)

    filenames = sorted(os.listdir(lq_dir))
    print(f"Total images to evaluate: {len(filenames)}")

    results = []
    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
    total_time = 0.0

    csv_file = os.path.join(out_dir, 'metrics.csv')

    with open(csv_file, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['filename', 'psnr', 'ssim', 'rmse', 'time_sec'])

        for idx, fname in enumerate(filenames):
            lq_path = os.path.join(lq_dir, fname)
            gt_path = os.path.join(gt_dir, fname)
            if not os.path.exists(gt_path):
                continue

            # Read images with OpenCV (BGR)
            img_lq = cv2.imread(lq_path, cv2.IMREAD_COLOR)
            img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
            assert img_lq is not None, f"Failed to load: {lq_path}"
            assert img_gt is not None, f"Failed to load: {gt_path}"

            # Convert to Tensor (B, C, H, W) normalized to [0, 1]
            img_lq_tensor = img2tensor(img_lq, bgr2rgb=True, float32=True).unsqueeze(0).to(device) / 255.0

            # Padding to multiple of 8
            H, W = img_lq_tensor.shape[-2:]
            pad_h = (8 - H % 8) % 8
            pad_w = (8 - W % 8) % 8
            if pad_h > 0 or pad_w > 0:
                img_lq_padded = F.pad(img_lq_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            else:
                img_lq_padded = img_lq_tensor

            t0 = time.time()
            with torch.no_grad():
                out_tensor = net(img_lq_padded)
            elapsed = time.time() - t0

            # Unpad back to original H, W
            out_tensor = out_tensor[:, :, :H, :W]

            # Convert back to uint8 BGR image
            out_img = tensor2img(out_tensor, rgb2bgr=True)

            # Compute exact metrics via basicsr.metrics
            psnr_val = calculate_psnr(out_img, img_gt, crop_border=0, test_y_channel=False)
            ssim_val = calculate_ssim(out_img, img_gt, crop_border=0, test_y_channel=False)
            rmse_val = calculate_rmse(out_img, img_gt, crop_border=0, test_y_channel=False)

            total_psnr += psnr_val
            total_ssim += ssim_val
            total_rmse += rmse_val
            total_time += elapsed

            results.append((fname, psnr_val, ssim_val, rmse_val, elapsed))
            writer.writerow([fname, f"{psnr_val:.4f}", f"{ssim_val:.4f}", f"{rmse_val:.4f}", f"{elapsed:.4f}"])

            # Save restored image
            cv2.imwrite(os.path.join(images_dir, fname), out_img)

            if (idx + 1) % max(1, len(filenames) // 5) == 0 or idx == len(filenames) - 1:
                print(f"  [{idx + 1}/{len(filenames)}] {fname:25s} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | RMSE: {rmse_val:.2f} | Time: {elapsed:.3f}s")

    n_items = len(results)
    avg_psnr = total_psnr / n_items if n_items > 0 else 0.0
    avg_ssim = total_ssim / n_items if n_items > 0 else 0.0
    avg_rmse = total_rmse / n_items if n_items > 0 else 0.0
    avg_sec = total_time / n_items if n_items > 0 else 0.0

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Results CSV : {csv_file}")
    print(f"Images Dir  : {images_dir}")
    print(f"Total Evaluated : {n_items} images")
    print(f"Average PSNR    : {avg_psnr:.4f} dB")
    print(f"Average SSIM    : {avg_ssim:.4f}")
    print(f"Average RMSE    : {avg_rmse:.4f}")
    print(f"Average Speed   : {avg_sec:.4f} sec/image")
    print("=" * 70 + "\n")

    return {
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'avg_rmse': avg_rmse,
        'avg_sec': avg_sec,
        'csv_file': csv_file,
        'n_params': n_params
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to options yaml file')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained weights .pth')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Dataset split')
    parser.add_argument('--data_dir', type=str, default=None, help='Override dataset directory')
    parser.add_argument('--out', type=str, default=None, help='Output directory for metrics and images')
    args = parser.parse_args()
    test_pipeline(args.opt, args.weights, args.split, args.data_dir, args.out)
