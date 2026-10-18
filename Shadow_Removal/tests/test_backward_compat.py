import sys
sys.path.insert(0, '.')
import torch
from basicsr.models.archs import define_network
from basicsr.models.losses import L1Loss, FFTLoss
import basicsr.models.losses as loss_module


def test_backward_compat():
    print("Testing backward compatibility...")

    # 1. Original Restormer arch can be loaded dynamically
    restormer_opt = {
        'type': 'Restormer',
        'inp_channels': 3,
        'out_channels': 3,
        'dim': 16,
        'num_blocks': [1, 1, 1, 1],
        'num_refinement_blocks': 1,
        'heads': [1, 1, 1, 1],
        'ffn_expansion_factor': 2.0,
    }
    net = define_network(restormer_opt)
    x = torch.randn(1, 3, 32, 32)
    out = net(x)
    assert out.shape == x.shape, "Original Restormer forward failed"
    print("  Original Restormer instantiation and forward: OK")

    # 2. Loss module exports
    assert hasattr(loss_module, 'L1Loss'), "L1Loss missing from loss_module"
    assert hasattr(loss_module, 'CharbonnierLoss'), "CharbonnierLoss missing from loss_module"
    assert hasattr(loss_module, 'FFTLoss'), "FFTLoss missing from loss_module"
    print("  Loss module exports check: OK")

    # 3. Verify losses.py original classes untouched
    l1 = loss_module.L1Loss()
    l1_val = l1(x, x)
    assert l1_val.item() == 0.0, "L1Loss behavior changed"
    print("  Original L1Loss behavior: OK")

    print("ALL backward compatibility tests PASSED!\n")


if __name__ == '__main__':
    test_backward_compat()
