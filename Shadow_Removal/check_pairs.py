"""Check RDD dataset pairing: verify img/ and gt/ have matching files.

Usage:
    python Shadow_Removal/check_pairs.py --data_dir ./RDD_data
"""
import os
import argparse
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    from PIL import Image


def check_split(split_dir, split_name, min_size=128, max_check=0):
    img_dir = os.path.join(split_dir, 'img')
    gt_dir = os.path.join(split_dir, 'gt')

    if not os.path.isdir(img_dir):
        print(f'[{split_name}] ERROR: {img_dir} does not exist')
        return False
    if not os.path.isdir(gt_dir):
        print(f'[{split_name}] ERROR: {gt_dir} does not exist')
        return False

    img_files = set(os.listdir(img_dir))
    gt_files = set(os.listdir(gt_dir))

    only_img = img_files - gt_files
    only_gt = gt_files - img_files
    common = img_files & gt_files

    print(f'\n=== {split_name} ===')
    print(f'  img/ files : {len(img_files)}')
    print(f'  gt/  files : {len(gt_files)}')
    print(f'  matched    : {len(common)}')

    ok = True
    if only_img:
        print(f'  ONLY in img/ ({len(only_img)}): {sorted(only_img)[:10]}...')
        ok = False
    if only_gt:
        print(f'  ONLY in gt/  ({len(only_gt)}): {sorted(only_gt)[:10]}...')
        ok = False

    small_imgs = []
    size_mismatches = []
    check_list = sorted(common)
    if max_check > 0:
        check_list = check_list[:max_check]
        print(f'  (Checking first {max_check} pairs for dimensions and readability)')

    for fname in check_list:
        img_path = os.path.join(img_dir, fname)
        gt_path = os.path.join(gt_dir, fname)

        if HAS_CV2:
            img = cv2.imread(img_path)
            gt = cv2.imread(gt_path)
            if img is None:
                print(f'  WARNING: cannot read {img_path}')
                ok = False
                continue
            if gt is None:
                print(f'  WARNING: cannot read {gt_path}')
                ok = False
                continue
            img_shape = img.shape
            gt_shape = gt.shape
        else:
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    img_shape = (h, w, len(im.getbands()))
                with Image.open(gt_path) as gm:
                    w, h = gm.size
                    gt_shape = (h, w, len(gm.getbands()))
            except Exception as e:
                print(f'  WARNING: error reading {fname}: {e}')
                ok = False
                continue

        if img_shape != gt_shape:
            size_mismatches.append((fname, img_shape, gt_shape))

        h, w = img_shape[:2]
        if h < min_size or w < min_size:
            small_imgs.append((fname, h, w))

    if size_mismatches:
        print(f'  SIZE MISMATCH ({len(size_mismatches)}):')
        for name, s1, s2 in size_mismatches[:5]:
            print(f'    {name}: img={s1} vs gt={s2}')
        ok = False

    if small_imgs:
        print(f'  Small images < {min_size}px ({len(small_imgs)}):')
        for name, h, w in small_imgs[:5]:
            print(f'    {name}: {h}x{w}')

    # Sample image info
    if common:
        sample = sorted(common)[0]
        if HAS_CV2:
            s_img = cv2.imread(os.path.join(img_dir, sample))
            print(f'  Sample: {sample}, shape={s_img.shape}, dtype={s_img.dtype}')
        else:
            with Image.open(os.path.join(img_dir, sample)) as im:
                print(f'  Sample: {sample}, size={im.size}, mode={im.mode}')

    if ok:
        print(f'  STATUS: OK')
    else:
        print(f'  STATUS: ISSUES FOUND')
    return ok


def main():
    parser = argparse.ArgumentParser(description='Check RDD dataset pairs')
    parser.add_argument('--data_dir', type=str, default='./RDD_data',
                        help='Root of RDD data (contains train/, test/)')
    parser.add_argument('--min_size', type=int, default=128,
                        help='Minimum image dimension to report')
    parser.add_argument('--max_check', type=int, default=0,
                        help='Max images to read content for (0 = all)')
    args = parser.parse_args()

    all_ok = True
    for split in ['train', 'test']:
        split_dir = os.path.join(args.data_dir, split)
        if os.path.isdir(split_dir):
            ok = check_split(split_dir, split, args.min_size, args.max_check)
            all_ok = all_ok and ok
        else:
            print(f'WARNING: {split_dir} not found, skipping')

    print(f'\n{"="*40}')
    print(f'Overall: {"ALL OK" if all_ok else "ISSUES FOUND"}')


if __name__ == '__main__':
    main()
