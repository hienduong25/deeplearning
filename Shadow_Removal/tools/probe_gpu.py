"""GPU Probing Tool for Colab Execution.
Measures peak VRAM, throughput (sec/iter), and estimates total training time across
progressive stages. Detects OOM and suggests optimal mini-batch sizes.

Usage on Colab:
    python Shadow_Removal/tools/probe_gpu.py --opt Shadow_Removal/Options/5_DocShadowNetV2_full.yml
"""
import os
import sys
sys.path.insert(0, '.')
import time
import argparse
import yaml
import torch
import torch.nn as nn
from basicsr.models import create_model
from basicsr.data import create_dataloader, create_dataset


def probe(opt_path, n_iters=20):
    with open(opt_path, 'r') as f:
        opt = yaml.safe_load(f)

    opt['is_train'] = True
    opt['dist'] = False

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        gpu_name = torch.cuda.get_device_name(0).replace(' ', '_').replace('/', '_')
        device_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        gpu_name = 'CPU'
        device_name = 'CPU (Local environment - for GPU run on Colab)'
        total_vram_gb = 0.0

    print("=" * 70)
    print(f"PROBING HARDWARE & THROUGHPUT: {device_name}")
    print(f"Model Config: {opt['name']} ({opt['network_g']['type']})")
    print("=" * 70)

    train_opt = opt['datasets']['train']
    iters = train_opt['iters']
    mini_batch_sizes = train_opt['mini_batch_sizes']
    mini_gt_sizes = train_opt['gt_sizes']

    # Set up dummy or real dataset
    # If dataroot not found, fallback to synthetic data for offline validation
    use_synthetic = not os.path.isdir(train_opt['dataroot_gt'])
    if use_synthetic:
        print("[Notice] Real data folder not found locally. Using synthetic batches for probing.")

    # Create model
    model = create_model(opt)

    stage_reports = []
    total_est_hours = 0.0

    for stage_idx in range(len(iters)):
        stage_num = stage_idx + 1
        st_iters = iters[stage_idx]
        patch_size = mini_gt_sizes[stage_idx]
        orig_batch = mini_batch_sizes[stage_idx]

        curr_batch = orig_batch
        success = False
        oom_occurred = False

        while curr_batch >= 1:
            try:
                if is_cuda:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()

                # Warmup iterations
                for _ in range(3):
                    dummy_lq = torch.randn(curr_batch, 3, patch_size, patch_size)
                    dummy_gt = torch.randn(curr_batch, 3, patch_size, patch_size)
                    if is_cuda:
                        dummy_lq = dummy_lq.cuda()
                        dummy_gt = dummy_gt.cuda()
                    model.feed_train_data({'lq': dummy_lq, 'gt': dummy_gt})
                    model.optimize_parameters(1)

                # Benchmark iterations
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.time()
                for benchmark_step in range(n_iters):
                    dummy_lq = torch.randn(curr_batch, 3, patch_size, patch_size)
                    dummy_gt = torch.randn(curr_batch, 3, patch_size, patch_size)
                    if is_cuda:
                        dummy_lq = dummy_lq.cuda()
                        dummy_gt = dummy_gt.cuda()
                    model.feed_train_data({'lq': dummy_lq, 'gt': dummy_gt})
                    model.optimize_parameters(benchmark_step + 1)
                if is_cuda:
                    torch.cuda.synchronize()
                elapsed = time.time() - t0

                sec_per_iter = elapsed / n_iters
                stage_hours = (st_iters * sec_per_iter) / 3600.0
                total_est_hours += stage_hours
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if is_cuda else 0.0

                stage_reports.append({
                    'stage': stage_num,
                    'patch': patch_size,
                    'batch': curr_batch,
                    'iters': st_iters,
                    'peak_vram_mb': peak_vram_mb,
                    'sec_per_iter': sec_per_iter,
                    'hours': stage_hours,
                    'status': 'OK' if not oom_occurred else f'Adjusted from {orig_batch}'
                })
                success = True
                print(f"Stage {stage_num} (Patch {patch_size}, Batch {curr_batch}): {sec_per_iter:.3f} s/iter, Peak VRAM: {peak_vram_mb:.1f} MB, Est: {stage_hours:.2f} hrs")
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if 'out of memory' in str(e).lower():
                    oom_occurred = True
                    print(f"Stage {stage_num} OOM with batch {curr_batch}! Trying batch {curr_batch // 2}...")
                    curr_batch //= 2
                    if is_cuda:
                        torch.cuda.empty_cache()
                else:
                    raise e

        if not success:
            stage_reports.append({
                'stage': stage_num,
                'patch': patch_size,
                'batch': orig_batch,
                'iters': st_iters,
                'peak_vram_mb': -1,
                'sec_per_iter': -1,
                'hours': -1,
                'status': 'FAILED OOM even with batch 1'
            })

    # Generate markdown report
    os.makedirs('Shadow_Removal/colab', exist_ok=True)
    report_file = f'Shadow_Removal/colab/probe_{gpu_name}.md'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# BÁO CÁO PROBE GPU: {device_name}\n\n")
        f.write(f"- **Mô hình**: `{opt['name']}` (`{opt['network_g']['type']}`)\n")
        f.write(f"- **VRAM phần cứng**: {total_vram_gb:.2f} GB\n")
        f.write(f"- **Tổng thời gian dự kiến (60k iters)**: **{total_est_hours:.2f} giờ**\n\n")
        f.write("### Chi tiết từng giai đoạn Progressive Learning\n\n")
        f.write("| Giai đoạn | Patch | Batch | Số iters | VRAM đỉnh (MB) | Giây / iter | Dự kiến (giờ) | Trạng thái |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for rep in stage_reports:
            f.write(f"| Stage {rep['stage']} | {rep['patch']}x{rep['patch']} | {rep['batch']} | {rep['iters']:,} | {rep['peak_vram_mb']:.1f} | {rep['sec_per_iter']:.3f} | {rep['hours']:.2f} | {rep['status']} |\n")

    print(f"\nReport written to: {report_file}")
    return report_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='Shadow_Removal/Options/5_DocShadowNetV2_full.yml')
    parser.add_argument('--iters', type=int, default=5)
    args = parser.parse_args()
    probe(args.opt, n_iters=args.iters)
