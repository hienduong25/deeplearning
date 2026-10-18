"""Simulate progressive learning schedule and learning rate trajectory without training.
Verifies schedule consistency and reproduces exact logic from basicsr/train.py.

Usage:
    python Shadow_Removal/tools/check_progressive.py [opt_path]
"""
import os
import sys
sys.path.insert(0, '.')
import yaml
import numpy as np
import torch
import torch.nn as nn
from basicsr.models.lr_scheduler import CosineAnnealingRestartCyclicLR


def check_progressive(opt_path):
    print(f"\n{'='*75}")
    print(f"Checking Progressive Learning Config: {opt_path}")
    print(f"{'='*75}")

    with open(opt_path, 'r') as f:
        opt = yaml.safe_load(f)

    train_ds = opt['datasets']['train']
    iters = train_ds['iters']
    mini_batch_sizes = train_ds['mini_batch_sizes']
    mini_gt_sizes = train_ds['gt_sizes']
    total_iter = opt['train']['total_iter']
    periods = opt['train']['scheduler']['periods']

    # Assert sum(iters) == total_iter == sum(periods)
    assert sum(iters) == total_iter, (
        f"sum(iters) {sum(iters)} != total_iter {total_iter}")
    assert sum(periods) == total_iter, (
        f"sum(periods) {sum(periods)} != total_iter {total_iter}")
    print(f"Schedule Assertion PASSED: sum(iters) == total_iter == sum(periods) == {total_iter}")

    # Simulation matching exact logic in basicsr/train.py:
    # groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])
    # j = ((current_iter > groups) != True).nonzero()[0]
    # bs_j = len(groups) - 1 if len(j) == 0 else j[0]
    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    # Build stages summary
    print("\n--- Progressive Learning Stages ---")
    header = f"{'Giai đoạn':<12} | {'Iter bắt đầu':<14} | {'Iter kết thúc':<14} | {'Patch (px)':<10} | {'Batch':<6} | {'Điểm ảnh/iter':<14}"
    print(header)
    print("-" * len(header))

    start_iter = 1
    for stage_idx in range(len(iters)):
        end_iter = groups[stage_idx]
        patch_size = mini_gt_sizes[stage_idx]
        batch_size = mini_batch_sizes[stage_idx]
        pixels_per_iter = batch_size * patch_size * patch_size
        print(f"Stage {stage_idx + 1:<6} | {start_iter:<14} | {end_iter:<14} | {patch_size:<10} | {batch_size:<6} | {pixels_per_iter:<14,d}")
        start_iter = end_iter + 1

    # Verify simulation across iterations
    # Test boundary and middle iterations
    check_points = [1, 10000, 20000, 20001, 34000, 34001, 48000, 48001, 60000]
    print("\n--- Point-by-Point Stage Verification (train.py logic) ---")
    for it in check_points:
        j = ((it > groups) != True).nonzero()[0]
        bs_j = len(groups) - 1 if len(j) == 0 else j[0]
        print(f"  Iter {it:6d} -> Stage {bs_j + 1} (Patch: {mini_gt_sizes[bs_j]}, Batch: {mini_batch_sizes[bs_j]})")

    # Real learning rate scheduler simulation
    dummy_param = nn.Parameter(torch.zeros(1))
    opt_g = torch.optim.AdamW([dummy_param], lr=float(opt['train']['optim_g']['lr']))
    sched_opt = opt['train']['scheduler']
    scheduler = CosineAnnealingRestartCyclicLR(
        opt_g,
        periods=sched_opt['periods'],
        restart_weights=sched_opt['restart_weights'],
        eta_mins=sched_opt['eta_mins']
    )

    print("\n--- Real Learning Rate at Key Checkpoints ---")
    lr_checkpoints = [0, 20000, 20001, 39999, 60000]
    for target_iter in lr_checkpoints:
        scheduler.last_epoch = target_iter
        lr = scheduler.get_lr()[0]
        print(f"  Iter {target_iter:6d} -> Learning Rate: {lr:.8f}")

    print(f"\nAll progressive checks PASSED for {opt_path}!")


def main():
    if len(sys.argv) > 1:
        configs = sys.argv[1:]
    else:
        configs = [
            'Shadow_Removal/Options/1_Restormer_baseline.yml',
            'Shadow_Removal/Options/2_Restormer_fft.yml',
            'Shadow_Removal/Options/3_DocShadowNetV2_fp_freqavg.yml',
            'Shadow_Removal/Options/4_DocShadowNetV2_hf.yml',
            'Shadow_Removal/Options/5_DocShadowNetV2_full.yml',
        ]

    for cfg in configs:
        check_progressive(cfg)


if __name__ == '__main__':
    main()
