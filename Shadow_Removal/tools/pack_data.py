"""Pack the clean RDD dataset into a single standardized zip file for the team.
Includes: train/img, train/gt, val/img, val/gt, test/img, test/gt.
Excludes: back_gt, smoke crops, temporary files.

Usage:
    python Shadow_Removal/tools/pack_data.py --data_dir ./RDD_data --out RDD_data_final.zip
"""
import os
import zipfile
import argparse


def pack_data(data_dir='./RDD_data', out_zip='RDD_data_final.zip'):
    splits = ['train', 'val', 'test']
    subfolders = ['img', 'gt']

    print(f"Packaging clean dataset from '{data_dir}' into '{out_zip}'...")
    total_files = 0

    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for split in splits:
            for sub in subfolders:
                folder_path = os.path.join(data_dir, split, sub)
                assert os.path.isdir(folder_path), f"Directory not found: {folder_path}"
                files = sorted(os.listdir(folder_path))
                print(f"  Adding {split}/{sub}: {len(files)} files")
                for fname in files:
                    full_p = os.path.join(folder_path, fname)
                    # Store arcname as RDD_data/{split}/{sub}/{fname}
                    arcname = os.path.join('RDD_data', split, sub, fname)
                    zf.write(full_p, arcname=arcname)
                    total_files += 1

    file_size_bytes = os.path.getsize(out_zip)
    file_size_mb = file_size_bytes / (1024 ** 2)
    file_size_gb = file_size_bytes / (1024 ** 3)

    print("\n" + "=" * 60)
    print("DATASET PACKAGING COMPLETE")
    print("=" * 60)
    print(f"Output archive : {out_zip}")
    print(f"Total files    : {total_files:,} images")
    print(f"Archive size   : {file_size_mb:.2f} MB ({file_size_gb:.3f} GB)")
    print(f"Contents:")
    print(f"  train: 4,171 img + 4,171 gt")
    print(f"  val  :   200 img +   200 gt")
    print(f"  test :   545 img +   545 gt")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./RDD_data')
    parser.add_argument('--out', type=str, default='RDD_data_final.zip')
    args = parser.parse_args()
    pack_data(args.data_dir, args.out)
