"""Apply fixed validation split to RDD dataset by moving files specified in val_files.txt.
Idempotent and safe to run multiple times without duplicating or corrupting data.

Usage:
    python Shadow_Removal/tools/apply_split.py --data_dir ./RDD_data
"""
import os
import shutil
import argparse


def apply_split(data_dir='./RDD_data', split_file='Shadow_Removal/splits/val_files.txt'):
    assert os.path.exists(split_file), f"Split file not found: {split_file}"
    with open(split_file, 'r', encoding='utf-8') as f:
        val_filenames = [line.strip() for line in f if line.strip()]

    print(f"Applying split from {split_file} ({len(val_filenames)} entries)...")
    train_img = os.path.join(data_dir, 'train', 'img')
    train_gt = os.path.join(data_dir, 'train', 'gt')
    val_img = os.path.join(data_dir, 'val', 'img')
    val_gt = os.path.join(data_dir, 'val', 'gt')

    os.makedirs(val_img, exist_ok=True)
    os.makedirs(val_gt, exist_ok=True)

    moved_count = 0
    already_val_count = 0

    for fname in val_filenames:
        src_i = os.path.join(train_img, fname)
        src_g = os.path.join(train_gt, fname)
        dst_i = os.path.join(val_img, fname)
        dst_g = os.path.join(val_gt, fname)

        # Move img if in train
        if os.path.exists(src_i):
            shutil.move(src_i, dst_i)
        elif os.path.exists(dst_i):
            already_val_count += 1

        # Move gt if in train
        if os.path.exists(src_g):
            shutil.move(src_g, dst_g)
            moved_count += 1

    print(f"Split applied successfully:")
    print(f"  Moved from train to val: {moved_count}")
    print(f"  Already in val:          {already_val_count}")
    print(f"  Total val items:         {len(os.listdir(val_img))} img, {len(os.listdir(val_gt))} gt")
    print(f"  Total train items:       {len(os.listdir(train_img))} img, {len(os.listdir(train_gt))} gt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./RDD_data')
    parser.add_argument('--split_file', type=str, default='Shadow_Removal/splits/val_files.txt')
    args = parser.parse_args()
    apply_split(args.data_dir, args.split_file)
