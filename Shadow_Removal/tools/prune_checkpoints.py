"""Prune older training checkpoints to prevent Google Drive storage overflow.
Keeps:
  - N newest .state files
  - N newest model .pth files
  - net_g_latest.pth
Deletes all older iterations.

Usage:
    python Shadow_Removal/tools/prune_checkpoints.py --exp 1_Restormer_baseline --keep 2
"""
import os
import re
import glob
import argparse


def extract_iter(filepath):
    """Extract iteration number from filename like 'net_g_25000.pth' or '25000.state'."""
    basename = os.path.basename(filepath)
    match = re.search(r'(\d+)', basename)
    return int(match.group(1)) if match else -1


def prune_checkpoints(exp_name, keep=2, base_dir='experiments'):
    exp_dir = os.path.join(base_dir, exp_name)
    if not os.path.isdir(exp_dir):
        print(f"Experiment directory not found: {exp_dir}. Nothing to prune.")
        return

    models_dir = os.path.join(exp_dir, 'models')
    states_dir = os.path.join(exp_dir, 'training_states')

    deleted_bytes = 0
    deleted_files = 0

    # 1. Prune models
    if os.path.isdir(models_dir):
        model_files = glob.glob(os.path.join(models_dir, 'net_g_*.pth'))
        # Exclude 'net_g_latest.pth' from candidate deletion
        numbered_models = [f for f in model_files if not f.endswith('net_g_latest.pth')]
        numbered_models.sort(key=extract_iter)

        if len(numbered_models) > keep:
            to_delete = numbered_models[:-keep]
            for f in to_delete:
                deleted_bytes += os.path.getsize(f)
                os.remove(f)
                deleted_files += 1
                print(f"  Deleted old model: {os.path.basename(f)}")

    # 2. Prune training states
    if os.path.isdir(states_dir):
        state_files = glob.glob(os.path.join(states_dir, '*.state'))
        state_files.sort(key=extract_iter)

        if len(state_files) > keep:
            to_delete_states = state_files[:-keep]
            for f in to_delete_states:
                deleted_bytes += os.path.getsize(f)
                os.remove(f)
                deleted_files += 1
                print(f"  Deleted old state: {os.path.basename(f)}")

    freed_mb = deleted_bytes / (1024 ** 2)
    print(f"\nPrune summary for '{exp_name}':")
    print(f"  Files deleted : {deleted_files}")
    print(f"  Storage freed : {freed_mb:.2f} MB")
    print(f"  Retained newest {keep} models and states (+ net_g_latest.pth).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True, help='Experiment name')
    parser.add_argument('--keep', type=int, default=2, help='Number of newest checkpoints to keep')
    parser.add_argument('--base_dir', type=str, default='experiments', help='Base experiments dir')
    args = parser.parse_args()
    prune_checkpoints(args.exp, args.keep, args.base_dir)
