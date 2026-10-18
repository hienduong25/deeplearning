"""Generate scaled budget training configurations.
Scales 5 baseline/ablation configs to a target total_iter N with preserved ratios.

Ratios:
  iters:   [1/3, 7/30, 7/30, 1/5]
  periods: [1/3, 2/3]

Usage:
    python Shadow_Removal/tools/make_budget_configs.py --total 30000
"""
import os
import sys
import glob
import argparse
import yaml


def make_budget_configs(total_n, ckpt_freq=None):
    if ckpt_freq is None:
        # Choose reasonable checkpoint freq (around 10% to 20% of total)
        if total_n >= 40000:
            ckpt_freq = 5000
        elif total_n >= 20000:
            ckpt_freq = 2500
        elif total_n >= 10000:
            ckpt_freq = 1000
        else:
            ckpt_freq = max(100, total_n // 10)

    # 1. Compute iters: [1/3, 7/30, 7/30, 1/5]
    it1 = int(round(total_n * (1.0 / 3.0)))
    it2 = int(round(total_n * (7.0 / 30.0)))
    it3 = int(round(total_n * (7.0 / 30.0)))
    # Enforce exact sum
    it4 = total_n - (it1 + it2 + it3)
    iters = [it1, it2, it3, it4]

    # 2. Compute periods: [1/3, 2/3]
    p1 = int(round(total_n * (1.0 / 3.0)))
    p2 = total_n - p1
    periods = [p1, p2]

    assert sum(iters) == total_n, f"sum(iters) {sum(iters)} != total_n {total_n}"
    assert sum(periods) == total_n, f"sum(periods) {sum(periods)} != total_n {total_n}"

    print(f"Generating Budget Configs for total_iter = {total_n}:")
    print(f"  iters   : {iters} (sum = {sum(iters)})")
    print(f"  periods : {periods} (sum = {sum(periods)})")
    print(f"  val_freq & save_checkpoint_freq: {ckpt_freq}")

    out_dir = os.path.join('Shadow_Removal', 'Options', f'budget_{total_n}')
    os.makedirs(out_dir, exist_ok=True)

    source_configs = sorted(glob.glob('Shadow_Removal/Options/[1-5]_*.yml'))
    assert len(source_configs) == 5, f"Expected 5 source configs, found {len(source_configs)}"

    generated = []
    for src_path in source_configs:
        with open(src_path, 'r') as f:
            opt = yaml.safe_load(f)

        base_name = os.path.basename(src_path)
        opt['name'] = f"{opt['name']}_budget_{total_n}"
        opt['datasets']['train']['iters'] = iters
        opt['train']['total_iter'] = total_n
        opt['train']['scheduler']['periods'] = periods
        opt['val']['val_freq'] = ckpt_freq
        opt['logger']['save_checkpoint_freq'] = ckpt_freq

        dest_path = os.path.join(out_dir, base_name)
        with open(dest_path, 'w') as f:
            yaml.dump(opt, f, sort_keys=False)

        generated.append(dest_path)
        print(f"  Saved: {dest_path}")

    print(f"\nSuccessfully created {len(generated)} budget configs in: {out_dir}")
    return generated


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total', type=int, default=30000, help='Target total_iter')
    parser.add_argument('--ckpt_freq', type=int, default=None, help='Checkpoint & val frequency')
    args = parser.parse_args()
    make_budget_configs(args.total, args.ckpt_freq)
