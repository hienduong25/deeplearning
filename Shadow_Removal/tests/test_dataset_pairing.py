import os
import sys
sys.path.insert(0, '.')
from basicsr.data.paired_image_dataset import Dataset_PairedImage


def test_dataset_pairing():
    print("Testing Dataset_PairedImage with RDD data...")

    # 1. Test smoke dataset
    opt_smoke = {
        'name': 'SmokeTest',
        'type': 'Dataset_PairedImage',
        'dataroot_gt': './RDD_data/smoke/train/gt',
        'dataroot_lq': './RDD_data/smoke/train/img',
        'io_backend': {'type': 'disk'},
        'phase': 'train',
        'scale': 1,
        'gt_size': 128,
        'geometric_augs': False,
    }
    ds_smoke = Dataset_PairedImage(opt_smoke)
    smoke_gt_count = len(os.listdir(opt_smoke['dataroot_gt']))
    assert len(ds_smoke) == smoke_gt_count, f"Expected {smoke_gt_count}, got {len(ds_smoke)}"

    item = ds_smoke[0]
    assert 'lq' in item and 'gt' in item, "Item missing lq or gt"
    print(f"  Item 0: lq shape={item['lq'].shape}, gt shape={item['gt'].shape}")

    # 2. Test full RDD train dataset (4171 matched pairs after 200 moved to val)
    opt_train = {
        'name': 'RDD_Train',
        'type': 'Dataset_PairedImage',
        'dataroot_gt': './RDD_data/train/gt',
        'dataroot_lq': './RDD_data/train/img',
        'io_backend': {'type': 'disk'},
        'phase': 'train',
        'scale': 1,
        'gt_size': 128,
        'geometric_augs': False,
    }
    ds_train = Dataset_PairedImage(opt_train)
    print(f"  Full RDD train dataset pairs loaded: {len(ds_train)}")
    assert len(ds_train) == 4171, f"Expected 4171 matched pairs, got {len(ds_train)}"

    print("ALL dataset pairing tests PASSED!\n")


if __name__ == '__main__':
    test_dataset_pairing()
