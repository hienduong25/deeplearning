import sys
sys.path.insert(0, '.')
import os
import copy
import torch

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils.options import parse


def run_pipeline_test(opt_path, exp_name):
    print(f"\n--- Testing pipeline for {exp_name} ({opt_path}) ---")
    opt = parse(opt_path, is_train=True)
    opt['name'] = f'test_{exp_name}'
    opt['num_gpu'] = 0  # CPU mode for local verification
    opt['dist'] = False
    opt['rank'] = 0
    opt['world_size'] = 1
    opt['datasets']['train']['num_worker_per_gpu'] = 0
    opt['val']['max_minibatch'] = 1

    # 1. Create train dataset & dataloader
    train_set = create_dataset(opt['datasets']['train'])
    print(f"  Train dataset loaded: {len(train_set)} pairs")
    train_loader = create_dataloader(
        train_set,
        opt['datasets']['train'],
        num_gpu=0,
        dist=False,
        sampler=None
    )

    # 2. Create val dataset & dataloader
    val_set = create_dataset(opt['datasets']['val'])
    print(f"  Val dataset loaded: {len(val_set)} pairs")
    val_loader = create_dataloader(
        val_set,
        opt['datasets']['val'],
        num_gpu=0,
        dist=False,
        sampler=None
    )

    # 3. Create model
    model = create_model(opt)
    print(f"  Model created: {model.__class__.__name__}")

    # 4. Run 2 train iterations
    data_iter = iter(train_loader)
    for step in range(1, 3):
        data = next(data_iter)
        model.feed_data(data)
        model.optimize_parameters(step)
        loss_dict = model.get_current_log()
        loss_str = ', '.join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
        print(f"  Train step {step}: {loss_str}")

    # 5. Run 1 validation step (cropped to 128x128 for fast CPU verification)
    val_data = next(iter(val_loader))
    val_data['lq'] = val_data['lq'][:, :, :128, :128]
    if 'gt' in val_data:
        val_data['gt'] = val_data['gt'][:, :, :128, :128]
    model.feed_data(val_data)
    model.pad_test(8)
    visuals = model.get_current_visuals()
    print(f"  Val step: result keys = {list(visuals.keys())}, output shape = {visuals['result'].shape}", flush=True)

    print(f"  {exp_name} pipeline test PASSED!")


def main():
    configs = [
        ('Shadow_Removal/Options/Restormer_RDD_baseline_smoke.yml', 'Restormer_baseline'),
        ('Shadow_Removal/Options/Restormer_RDD_fft_smoke.yml', 'Restormer_fft'),
        ('Shadow_Removal/Options/DocShadowNet_RDD_smoke.yml', 'DocShadowNet'),
    ]

    for opt_path, name in configs:
        run_pipeline_test(opt_path, name)

    print("\n" + "=" * 50)
    print("ALL 3 CONFIGURATION PIPELINES PASSED END-TO-END!")
    print("=" * 50)


if __name__ == '__main__':
    main()
