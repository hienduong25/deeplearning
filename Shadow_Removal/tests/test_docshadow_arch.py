import sys
sys.path.insert(0, '.')
import torch
from basicsr.models.archs import define_network
from basicsr.models.archs.docshadow_arch import DocShadowNet


def test_docshadow_architecture():
    print("Testing DocShadowNet...")

    # 1. Instantiate via define_network (verifies dynamic discovery)
    opt = {
        'type': 'DocShadowNet',
        'inp_channels': 3,
        'out_channels': 3,
        'dim': 16,  # small dim for quick test
        'num_blocks': [1, 1, 1, 1],
        'num_refinement_blocks': 1,
        'heads': [1, 1, 1, 1],
        'ffn_expansion_factor': 2.0,
        'bias': False,
        'LayerNorm_type': 'WithBias',
        'dual_pixel_task': False,
        'reduce_top': False,
    }

    net = define_network(opt)
    print("  Dynamically loaded DocShadowNet via define_network: OK")

    # 2. Forward pass with even dimensions
    x = torch.randn(2, 3, 64, 64)
    out = net(x)
    print(f"  Forward pass (B=2, C=3, H=64, W=64): Output shape = {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    # 3. Backward pass
    loss = out.sum()
    loss.backward()
    print("  Backward pass: OK")

    # 4. Test reduce_top=True
    net_reduce = DocShadowNet(
        inp_channels=3,
        out_channels=3,
        dim=16,
        num_blocks=[1, 1, 1, 1],
        num_refinement_blocks=1,
        heads=[1, 1, 1, 1],
        ffn_expansion_factor=2.0,
        reduce_top=True,
    )
    out_reduce = net_reduce(x)
    assert out_reduce.shape == x.shape, f"reduce_top shape mismatch: {out_reduce.shape}"
    print("  reduce_top=True forward pass: OK")

    print("ALL DocShadowNet tests PASSED!\n")


if __name__ == '__main__':
    test_docshadow_architecture()
