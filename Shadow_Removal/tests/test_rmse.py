import sys
sys.path.insert(0, '.')
import numpy as np
import torch
from basicsr.metrics import calculate_rmse


def test_rmse():
    print("Testing calculate_rmse...")

    # 1. Identical images -> RMSE = 0
    img1 = np.ones((128, 128, 3), dtype=np.uint8) * 128
    img2 = np.ones((128, 128, 3), dtype=np.uint8) * 128
    rmse = calculate_rmse(img1, img2, crop_border=0)
    print(f"  Identical uint8 RMSE: {rmse:.6f}")
    assert rmse == 0.0, f"Expected 0.0, got {rmse}"

    # 2. Known difference
    diff = 10.0
    img_diff = (img1.astype(np.float64) + diff).clip(0, 255)
    rmse_diff = calculate_rmse(img1, img_diff, crop_border=0)
    print(f"  Known diff RMSE (expected {diff}): {rmse_diff:.6f}")
    assert abs(rmse_diff - diff) < 1e-4, f"Expected {diff}, got {rmse_diff}"

    # 3. Torch Tensor input
    t1 = torch.ones((1, 3, 64, 64)) * 0.5
    t2 = torch.ones((1, 3, 64, 64)) * 0.6
    rmse_torch = calculate_rmse(t1, t2, crop_border=0, input_order='HWC')
    print(f"  Torch tensor RMSE: {rmse_torch:.6f}")
    assert abs(rmse_torch - 0.1) < 1e-4, f"Expected 0.1, got {rmse_torch}"

    # 4. Crop border
    rmse_crop = calculate_rmse(img1, img_diff, crop_border=4)
    assert abs(rmse_crop - diff) < 1e-4, "Crop border RMSE mismatch"
    print("  Crop border test: OK")

    print("ALL calculate_rmse tests PASSED!\n")


if __name__ == '__main__':
    test_rmse()
