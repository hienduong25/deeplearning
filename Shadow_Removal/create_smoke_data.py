"""Create a small smoke-test dataset (~20 pairs) from RDD train.
Does NOT modify the original train folder.

Usage:
    python Shadow_Removal/create_smoke_data.py --data_dir ./RDD_data --n_samples 20 --seed 42
"""
import os
import argparse
import random
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./RDD_data')
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_img = os.path.join(args.data_dir, 'train', 'img')
    train_gt = os.path.join(args.data_dir, 'train', 'gt')

    smoke_train_img = os.path.join(args.data_dir, 'smoke', 'train', 'img')
    smoke_train_gt = os.path.join(args.data_dir, 'smoke', 'train', 'gt')
    smoke_val_img = os.path.join(args.data_dir, 'smoke', 'val', 'img')
    smoke_val_gt = os.path.join(args.data_dir, 'smoke', 'val', 'gt')

    assert os.path.isdir(train_img), f'{train_img} not found'
    assert os.path.isdir(train_gt), f'{train_gt} not found'

    img_files = set(os.listdir(train_img))
    gt_files = set(os.listdir(train_gt))
    common = sorted(img_files & gt_files)

    random.seed(args.seed)
    chosen = random.sample(common, min(args.n_samples, len(common)))

    # Use first 15 for train, 5 for val
    train_samples = chosen[:15]
    val_samples = chosen[15:]

    for d in [smoke_train_img, smoke_train_gt, smoke_val_img, smoke_val_gt]:
        os.makedirs(d, exist_ok=True)

    for fname in train_samples:
        shutil.copy2(os.path.join(train_img, fname), os.path.join(smoke_train_img, fname))
        shutil.copy2(os.path.join(train_gt, fname), os.path.join(smoke_train_gt, fname))

    for fname in val_samples:
        shutil.copy2(os.path.join(train_img, fname), os.path.join(smoke_val_img, fname))
        shutil.copy2(os.path.join(train_gt, fname), os.path.join(smoke_val_gt, fname))

    print(f'Created smoke dataset: {len(train_samples)} train pairs, {len(val_samples)} val pairs')
    print(f'  Train: {smoke_train_img}')
    print(f'  Val:   {smoke_val_img}')


if __name__ == '__main__':
    main()
