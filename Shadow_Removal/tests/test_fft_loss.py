import sys
sys.path.insert(0, '.')
import torch
import torch.nn as nn
from basicsr.models.losses import FFTLoss


def test_fft_loss():
    print("Testing FFTLoss...")
    criterion = FFTLoss(loss_weight=0.1)

    # 1. Forward test
    pred = torch.randn(2, 3, 128, 128, requires_grad=True)
    target = torch.randn(2, 3, 128, 128)

    loss = criterion(pred, target)
    print(f"  Forward loss value: {loss.item():.6f}")
    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert loss.item() > 0, "Loss should be strictly positive for random inputs"

    # 2. Perfect match test (loss should be 0)
    loss_zero = criterion(target, target)
    print(f"  Zero loss value (identical inputs): {loss_zero.item():.8f}")
    assert loss_zero.item() < 1e-6, "Loss should be zero for identical inputs"

    # 3. Backward test
    loss.backward()
    assert pred.grad is not None, "Gradient should be computed for pred"
    assert not torch.isnan(pred.grad).any(), "Gradient should not contain NaNs"
    print("  Backward pass: OK (gradients computed, no NaNs)")

    # 4. Odd dimensions test
    pred_odd = torch.randn(1, 3, 127, 131, requires_grad=True)
    target_odd = torch.randn(1, 3, 127, 131)
    loss_odd = criterion(pred_odd, target_odd)
    loss_odd.backward()
    assert pred_odd.grad is not None, "Odd dimensions backward failed"
    print("  Odd dimensions test: OK")

    print("ALL FFTLoss tests PASSED!\n")


if __name__ == '__main__':
    test_fft_loss()
