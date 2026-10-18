"""Split ~200 pairs from RDD train into a val set (symlinks/copies).
Does NOT modify the original train folder.

Usage:
    python Shadow_Removal/create_val_split.py --data_dir ./RDD_data --n_val 200 --seed 42
"""
import os
import argparse
import random
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./RDD_data')
    parser.add_argument('--n_val', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_img = os.path.join(args.data_dir, 'train', 'img')
    train_gt = os.path.join(args.data_dir, 'train', 'gt')

    val_img = os.path.join(args.data_dir, 'val', 'img')
    val_gt = os.path.join(args.data_dir, 'val', 'gt')

    assert os.path.isdir(train_img), f'{train_img} not found'
    assert os.path.isdir(train_gt), f'{train_gt} not found'

    # Get common filenames
    img_files = set(os.listdir(train_img))
    gt_files = set(os.listdir(train_gt))
    common = sorted(img_files & gt_files)

    n_val = min(args.n_val, len(common))
    random.seed(args.seed)
    val_names = random.sample(common, n_val)

    os.makedirs(val_img, exist_ok=True)
    os.makedirs(val_gt, exist_ok=True)

    for fname in val_names:
        src_img = os.path.join(train_img, fname)
        src_gt = os.path.join(train_gt, fname)
        dst_img = os.path.join(val_img, fname)
        dst_gt = os.path.join(val_gt, fname)

        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)
        if not os.path.exists(dst_gt):
            shutil.copy2(src_gt, dst_gt)

    print(f'Created val split: {n_val} pairs')
    print(f'  val/img: {val_img}')
    print(f'  val/gt:  {val_gt}')
    print(f'  Seed: {args.seed}')


if __name__ == '__main__':
    main()
