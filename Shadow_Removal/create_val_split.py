"""Split ~200 pairs from RDD train into a val set by MOVING them.
Ensures no overlap between train and val.

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

    # If val already exists and has files that are also in train, remove them from train
    os.makedirs(val_img, exist_ok=True)
    os.makedirs(val_gt, exist_ok=True)

    val_existing = set(os.listdir(val_img))
    if val_existing:
        print(f"Detected existing val set with {len(val_existing)} images. Cleaning up any duplicates in train...")
        for fname in val_existing:
            dup_img = os.path.join(train_img, fname)
            dup_gt = os.path.join(train_gt, fname)
            if os.path.exists(dup_img):
                os.remove(dup_img)
            if os.path.exists(dup_gt):
                os.remove(dup_gt)

    # Get common filenames currently in train
    img_files = set(os.listdir(train_img))
    gt_files = set(os.listdir(train_gt))
    common = sorted(img_files & gt_files)

    needed_val = args.n_val - len(os.listdir(val_img))
    if needed_val > 0:
        random.seed(args.seed)
        val_names = random.sample(common, min(needed_val, len(common)))

        for fname in val_names:
            src_img = os.path.join(train_img, fname)
            src_gt = os.path.join(train_gt, fname)
            dst_img = os.path.join(val_img, fname)
            dst_gt = os.path.join(val_gt, fname)

            shutil.move(src_img, dst_img)
            shutil.move(src_gt, dst_gt)

    final_train_img = set(os.listdir(train_img))
    final_val_img = set(os.listdir(val_img))
    overlap = final_train_img & final_val_img

    print(f"Val split ready:")
    print(f"  val/img count  : {len(final_val_img)}")
    print(f"  train/img count: {len(final_train_img)}")
    print(f"  Overlap count  : {len(overlap)} (MUST be 0)")
    assert len(overlap) == 0, f"Error: Train and Val overlap by {len(overlap)} files!"


if __name__ == '__main__':
    main()
