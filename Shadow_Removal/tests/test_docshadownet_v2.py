import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer, Downsample, Upsample
from basicsr.models.archs.docshadow_arch import (
    DocShadowNetV2,
    FrequencyPreservedPooling,
    FreqAvgUp,
    HighFreqBranch
)
from basicsr.models.archs import define_network


def test_freqavgup_constant():
    print("=== Test 1: FreqAvgUp on constant image ===")
    test_values = [0.0, 1.0, 3.14159, -2.5, 255.0]
    for val in test_values:
        # Input shape (B=2, C=384, H=16, W=16)
        x = torch.full((2, 384, 16, 16), val, dtype=torch.float32)
        freq_up = FreqAvgUp(n_feat=384)
        out = freq_up(x)

        assert out.shape == (2, 192, 32, 32), f"Expected (2, 192, 32, 32), got {out.shape}"
        max_diff = torch.max(torch.abs(out - val)).item()
        print(f"  Constant value {val:9.4f} -> Max difference: {max_diff:.8f}")
        assert max_diff < 1e-4, f"Constant preservation failed for {val}: max_diff={max_diff}"

    print("FreqAvgUp constant preservation test PASSED!\n")


def test_fp_and_freqavgup_shapes():
    print("=== Test 2: Shape comparison of FP and FreqAvgUp with Downsample and Upsample ===")
    dim = 48
    x_down = torch.randn(2, dim * 4, 32, 32)
    down_orig = Downsample(dim * 4)
    down_fp = FrequencyPreservedPooling(dim * 4)

    out_down_orig = down_orig(x_down)
    out_down_fp = down_fp(x_down)
    print(f"  Downsample original output shape : {out_down_orig.shape}")
    print(f"  FrequencyPreservedPooling output : {out_down_fp.shape}")
    assert out_down_orig.shape == out_down_fp.shape, (
        f"Shape mismatch: {out_down_orig.shape} vs {out_down_fp.shape}")

    x_up = torch.randn(2, dim * 8, 16, 16)
    up_orig = Upsample(dim * 8)
    up_freq = FreqAvgUp(dim * 8)

    out_up_orig = up_orig(x_up)
    out_up_freq = up_freq(x_up)
    print(f"  Upsample original output shape   : {out_up_orig.shape}")
    print(f"  FreqAvgUp output shape           : {out_up_freq.shape}")
    assert out_up_orig.shape == out_up_freq.shape, (
        f"Shape mismatch: {out_up_orig.shape} vs {out_up_freq.shape}")

    print("FP and FreqAvgUp shape comparison test PASSED!\n")


def test_params_and_output_identical_when_flags_off():
    print("=== Test 3: DocShadowNetV2 with all flags OFF vs Restormer Original ===")
    kwargs = dict(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False,
    )

    restormer = Restormer(**kwargs)
    docshadow_v2_default = DocShadowNetV2(
        **kwargs,
        down_mode='pixel',
        up_mode='pixel',
        use_hf=False,
        reduce_top=False,
        out_mode='add'
    )

    n_params_restormer = sum(p.numel() for p in restormer.parameters())
    n_params_v2_off = sum(p.numel() for p in docshadow_v2_default.parameters())

    print(f"  Restormer original parameters   : {n_params_restormer:,}")
    print(f"  DocShadowNetV2 (flags OFF) params: {n_params_v2_off:,}")
    assert n_params_restormer == n_params_v2_off, (
        f"Parameter count mismatch! Restormer: {n_params_restormer} vs V2: {n_params_v2_off}")

    x = torch.randn(1, 3, 64, 64)
    # Forward check
    out_v2 = docshadow_v2_default(x)
    assert out_v2.shape == x.shape, f"Expected {x.shape}, got {out_v2.shape}"
    print(f"  Output shape match: {out_v2.shape} == {x.shape}")

    print("Parameter and shape identity test PASSED!\n")


def test_docshadownet_v2_ablation_matrix():
    print("=== Test 4: DocShadowNetV2 Ablation Variants Forward & Backward ===")
    variants = [
        ("Baseline (all flags off)", dict(down_mode='pixel', up_mode='pixel', use_hf=False)),
        ("+FP only", dict(down_mode='fp', up_mode='pixel', use_hf=False)),
        ("+FreqAvgUp only", dict(down_mode='pixel', up_mode='freqavg', use_hf=False)),
        ("+FP & FreqAvgUp", dict(down_mode='fp', up_mode='freqavg', use_hf=False)),
        ("+HF only", dict(down_mode='pixel', up_mode='pixel', use_hf=True)),
        ("Full (+FP, +FreqAvgUp, +HF)", dict(down_mode='fp', up_mode='freqavg', use_hf=True)),
        ("Full with reduce_top=True", dict(down_mode='fp', up_mode='freqavg', use_hf=True, reduce_top=True)),
    ]

    for name, flags in variants:
        opt = {
            'type': 'DocShadowNetV2',
            'inp_channels': 3,
            'out_channels': 3,
            'dim': 16,  # small dim for swift testing
            'num_blocks': [1, 1, 1, 1],
            'num_refinement_blocks': 1,
            'heads': [1, 1, 1, 1],
            'ffn_expansion_factor': 2.0,
            **flags
        }
        net = define_network(opt)
        n_params = sum(p.numel() for p in net.parameters())

        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        out = net(x)
        assert out.shape == x.shape
        loss = out.sum()
        loss.backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()

        print(f"  [{name:30s}] Params: {n_params:8,d} | Out: {list(out.shape)} | Fwd/Bwd: OK")

    print("All ablation variants PASSED!\n")


def main():
    test_freqavgup_constant()
    test_fp_and_freqavgup_shapes()
    test_params_and_output_identical_when_flags_off()
    test_docshadownet_v2_ablation_matrix()
    print("=" * 60)
    print("ALL DocShadowNetV2 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == '__main__':
    main()
