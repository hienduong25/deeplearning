## DocShadowNet: Frequency-Aware Transformer for Document Shadow Removal
## Designed for document restoration tasks (preserving text edges while removing low-frequency shadows)
## Extends Restormer with frequency-aware low-pass prior and high-frequency edge enhancement

import torch
import torch.nn as nn
import torch.nn.functional as F

from .restormer_arch import (
    LayerNorm,
    FeedForward,
    Attention,
    TransformerBlock,
    OverlapPatchEmbed,
    Downsample,
    Upsample,
)


class FFTLowPassFilter(nn.Module):
    """Simple 2D FFT Low-Pass Filter without 5/2 spectral padding.
    Extracts the low-frequency illumination profile (shadows) from feature maps.

    Args:
        cutoff (float): Normalized frequency cutoff in range (0, 0.5].
                        Lower values retain only coarse illumination.
    """

    def __init__(self, cutoff=0.2):
        super(FFTLowPassFilter, self).__init__()
        self.cutoff = cutoff

    def forward(self, x):
        B, C, H, W = x.shape
        x_f32 = x.float()

        # Real FFT 2D
        fft = torch.fft.rfft2(x_f32, norm='backward')

        # Normalized frequency coordinates
        freq_y = torch.fft.fftfreq(H, device=x.device)  # [-0.5, 0.5)
        freq_x = torch.fft.rfftfreq(W, device=x.device)  # [0, 0.5]

        # Radial distance in frequency plane
        dist = torch.sqrt(freq_y.unsqueeze(1) ** 2 + freq_x.unsqueeze(0) ** 2)

        # Low-pass mask
        mask = (dist <= self.cutoff).float().unsqueeze(0).unsqueeze(0)

        # Filtered spectrum and inverse transform
        fft_low = fft * mask
        x_low = torch.fft.irfft2(fft_low, s=(H, W), norm='backward')

        return x_low.to(x.dtype)


class FreqIllumBottleneck(nn.Module):
    """Frequency-guided Illumination Modulation at the latent bottleneck.
    Estimates affine transformation parameters (gamma, beta) from the low-frequency
    illumination field to normalize shadowed features.
    """

    def __init__(self, dim, cutoff=0.2):
        super(FreqIllumBottleneck, self).__init__()
        self.low_pass = FFTLowPassFilter(cutoff=cutoff)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        )

    def forward(self, x):
        # Extract low-frequency shadow/illumination representation
        x_low = self.low_pass(x)
        gamma_beta = self.conv(x_low)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = torch.tanh(gamma)
        return x * (1.0 + gamma) + beta


class FreqSkipEnhance(nn.Module):
    """Frequency-Aware Skip Connection Enhancement.
    Decomposes skip features into low-pass (shadow) and high-pass (text edge) components,
    and amplifies high-frequency text details to prevent character blur.
    """

    def __init__(self, dim, cutoff=0.2):
        super(FreqSkipEnhance, self).__init__()
        self.low_pass = FFTLowPassFilter(cutoff=cutoff)
        self.alpha = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        x_low = self.low_pass(x)
        x_high = x - x_low
        return x + self.alpha * x_high


class DocShadowNet(nn.Module):
    """DocShadowNet: Frequency-Aware Transformer Architecture for Document Shadow Removal.

    Matches Restormer's interface and U-Net depth while incorporating:
    1. Low-frequency illumination modulation at the bottleneck (FreqIllumBottleneck).
    2. High-frequency text stroke preservation at skip connections (FreqSkipEnhance).
    3. Option `reduce_top` for level 1 decoder (default False to match Restormer).

    Args:
        inp_channels (int): Input image channels (default 3).
        out_channels (int): Output clean image channels (default 3).
        dim (int): Base feature channels (default 48).
        num_blocks (list[int]): Number of transformer blocks per encoder/decoder level.
        num_refinement_blocks (int): Blocks in the refinement stage.
        heads (list[int]): Attention heads per level.
        ffn_expansion_factor (float): Feed-forward network expansion factor.
        bias (bool): Whether to use bias in linear/conv projections.
        LayerNorm_type (str): 'WithBias' or 'BiasFree'.
        dual_pixel_task (bool): True for DP defocus deblurring (default False).
        reduce_top (bool): Whether to use a 1x1 conv at level 1 decoder to reduce
                           channels from 2*dim to dim. Default: False (Restormer convention).
        cutoff (float): Cutoff frequency for FFT low-pass filtering. Default: 0.2.
    """

    def __init__(
        self,
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
        reduce_top=False,
        cutoff=0.2,
    ):
        super(DocShadowNet, self).__init__()

        self.reduce_top = reduce_top
        self.dual_pixel_task = dual_pixel_task

        # Patch Embedding
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # Encoder Level 1
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[0])
        ])

        # Downsample 1 -> 2
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])
        ])

        # Downsample 2 -> 3
        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[2])
        ])

        # Downsample 3 -> 4
        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**3),
                num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[3])
        ])

        # Frequency-aware bottleneck illumination modulation
        self.freq_bottleneck = FreqIllumBottleneck(int(dim * 2**3), cutoff=cutoff)

        # Frequency skip enhancers for high-frequency text details
        self.skip_enhance3 = FreqSkipEnhance(int(dim * 2**2), cutoff=cutoff)
        self.skip_enhance2 = FreqSkipEnhance(int(dim * 2**1), cutoff=cutoff)
        self.skip_enhance1 = FreqSkipEnhance(dim, cutoff=cutoff)

        # Decoder Level 3
        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[2])
        ])

        # Decoder Level 2
        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])
        ])

        # Decoder Level 1
        self.up2_1 = Upsample(int(dim * 2**1))

        if self.reduce_top:
            self.reduce_chan_level1 = nn.Conv2d(int(dim * 2**1), dim, kernel_size=1, bias=bias)
            dec1_dim = dim
        else:
            dec1_dim = int(dim * 2**1)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dec1_dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[0])
        ])

        # Refinement Stage
        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=dec1_dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_refinement_blocks)
        ])

        # Dual pixel task support (backward compatibility)
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dec1_dim, kernel_size=1, bias=bias)

        # Output projection
        self.output = nn.Conv2d(dec1_dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # Level 1 Encoder
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        # Level 2 Encoder
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        # Level 3 Encoder
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        # Bottleneck Latent
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # Frequency Illumination Bottleneck Modulation
        latent = self.freq_bottleneck(latent)

        # Level 3 Decoder with frequency-enhanced skip
        inp_dec_level3 = self.up4_3(latent)
        skip3 = self.skip_enhance3(out_enc_level3)
        inp_dec_level3 = torch.cat([inp_dec_level3, skip3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        # Level 2 Decoder with frequency-enhanced skip
        inp_dec_level2 = self.up3_2(out_dec_level3)
        skip2 = self.skip_enhance2(out_enc_level2)
        inp_dec_level2 = torch.cat([inp_dec_level2, skip2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        # Level 1 Decoder with frequency-enhanced skip
        inp_dec_level1 = self.up2_1(out_dec_level2)
        skip1 = self.skip_enhance1(out_enc_level1)
        inp_dec_level1 = torch.cat([inp_dec_level1, skip1], 1)
        if self.reduce_top:
            inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # Refinement
        out_dec_level1 = self.refinement(out_dec_level1)

        # Output projection + global residual
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out = self.output(out_dec_level1)
        else:
            out = self.output(out_dec_level1) + inp_img

        return out


##########################################################################
## Frequency Modules for DocShadowNetV2

class FrequencyPreservedPooling(nn.Module):
    """Frequency-Preserved Pooling from Level 3 to Level 4 (latent).
    Applies anti-aliasing low-pass frequency filtering prior to downsampling,
    preserving global illumination structure and matching Downsample's output shape.
    Input: (B, C, H, W) -> Output: (B, C*2, H//2, W//2)
    """

    def __init__(self, n_feat, cutoff=0.25):
        super(FrequencyPreservedPooling, self).__init__()
        self.cutoff = cutoff
        self.conv = nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x):
        B, C, H, W = x.shape
        x_f32 = x.float()
        fft = torch.fft.rfft2(x_f32)
        freq_y = torch.fft.fftfreq(H, device=x.device)
        freq_x = torch.fft.rfftfreq(W, device=x.device)
        dist = torch.sqrt(freq_y.unsqueeze(1) ** 2 + freq_x.unsqueeze(0) ** 2)
        mask = (dist <= self.cutoff).float().unsqueeze(0).unsqueeze(0)
        mask[:, :, 0, 0] = 1.0  # Preserve DC component

        x_lp = torch.fft.irfft2(fft * mask, s=(H, W)).to(x.dtype)
        out = self.conv(x_lp)
        out = self.unshuffle(out)
        return out


class FreqAvgUp(nn.Module):
    """Frequency Average Upsampling from Level 4 to Level 3.
    Maps (B, C, H, W) -> (B, C//2, 2*H, 2*W).
    Guaranteed: on a constant image c, the output is identically c.
    Output shape matches Upsample(n_feat).
    """

    def __init__(self, n_feat, cutoff=0.25):
        super(FreqAvgUp, self).__init__()
        self.n_feat = n_feat
        self.out_channels = n_feat // 2
        self.cutoff = cutoff

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.n_feat, f"Expected {self.n_feat} channels, got {C}"

        # 1. Channel reduction C -> C // 2 via pairwise averaging (preserves constant c)
        x_chan = 0.5 * (x[:, 0::2, :, :] + x[:, 1::2, :, :])

        # 2. 2x spatial upsampling via bilinear interpolation (preserves constant c)
        x_up = F.interpolate(x_chan, scale_factor=2, mode='bilinear', align_corners=False)

        # 3. Frequency low-pass filtering with DC gain = 1.0 (preserves constant c)
        H_up, W_up = x_up.shape[-2:]
        x_f32 = x_up.float()
        fft = torch.fft.rfft2(x_f32)
        freq_y = torch.fft.fftfreq(H_up, device=x.device)
        freq_x = torch.fft.rfftfreq(W_up, device=x.device)
        dist = torch.sqrt(freq_y.unsqueeze(1) ** 2 + freq_x.unsqueeze(0) ** 2)
        mask = (dist <= self.cutoff).float().unsqueeze(0).unsqueeze(0)
        mask[:, :, 0, 0] = 1.0  # DC component untouched

        out = torch.fft.irfft2(fft * mask, s=(H_up, W_up)).to(x.dtype)
        return out


class HighFreqBranch(nn.Module):
    """Lightweight CNN extracting high-frequency features from input document image.
    Extracts high-pass residual (x - low_pass(x)) followed by a 3-layer CNN.
    """

    def __init__(self, inp_channels=3, out_channels=3, hidden_dim=16, cutoff=0.15):
        super(HighFreqBranch, self).__init__()
        self.cutoff = cutoff
        self.conv = nn.Sequential(
            nn.Conv2d(inp_channels, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        # Initialize final layer weights to 0 for safe residual addition
        nn.init.zeros_(self.conv[-1].weight)

    def forward(self, x):
        x_f32 = x.float()
        fft = torch.fft.rfft2(x_f32)
        H, W = x.shape[-2:]
        freq_y = torch.fft.fftfreq(H, device=x.device)
        freq_x = torch.fft.rfftfreq(W, device=x.device)
        dist = torch.sqrt(freq_y.unsqueeze(1) ** 2 + freq_x.unsqueeze(0) ** 2)
        lp_mask = (dist <= self.cutoff).float().unsqueeze(0).unsqueeze(0)
        x_low = torch.fft.irfft2(fft * lp_mask, s=(H, W)).to(x.dtype)
        x_high = x - x_low
        return self.conv(x_high)


class DocShadowNetV2(nn.Module):
    """DocShadowNetV2: Full specification matching DocumentRestoration-DL diagram.

    Features:
    - Downsample at Level 3->4: controlled by down_mode ('pixel' | 'fp')
      'pixel': standard Downsample (Conv + PixelUnshuffle)
      'fp': FrequencyPreservedPooling
    - Upsample at Level 4->3: controlled by up_mode ('pixel' | 'freqavg')
      'pixel': standard Upsample (Conv + PixelShuffle)
      'freqavg': FreqAvgUp
    - High-frequency branch: controlled by use_hf (bool)
      If True: lightweight CNN processing high-frequency detail from input image I
    - Level 1 decoder reduction: controlled by reduce_top (bool, default False)
      If False: Level 1 decoder operates on 2*dim channels (exact match to Restormer)
      If True: 1x1 conv reduces 2*dim -> dim
    - Output composition: out_mode ('add')
      out = R + I + (HF if use_hf else 0)

    When down_mode='pixel', up_mode='pixel', use_hf=False, reduce_top=False:
      Exact identical parameter count and architecture to original Restormer!
    """

    def __init__(
        self,
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
        down_mode='pixel',      # 'pixel' | 'fp'
        up_mode='pixel',        # 'pixel' | 'freqavg'
        use_hf=False,           # bool
        reduce_top=False,       # bool (default False to match Restormer)
        out_mode='add',         # 'add'
        cutoff=0.25,
    ):
        super(DocShadowNetV2, self).__init__()

        assert down_mode in ['pixel', 'fp'], f"Unknown down_mode: {down_mode}"
        assert up_mode in ['pixel', 'freqavg'], f"Unknown up_mode: {up_mode}"
        assert out_mode == 'add', f"Only out_mode='add' supported, got {out_mode}"

        self.down_mode = down_mode
        self.up_mode = up_mode
        self.use_hf = use_hf
        self.reduce_top = reduce_top
        self.dual_pixel_task = dual_pixel_task

        # Patch Embedding
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # Encoder Level 1
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[0])
        ])

        # Downsample 1 -> 2 (PixelUnshuffle)
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])
        ])

        # Downsample 2 -> 3 (PixelUnshuffle)
        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[2])
        ])

        # Downsample 3 -> 4: PixelUnshuffle or FrequencyPreservedPooling
        if self.down_mode == 'fp':
            self.down3_4 = FrequencyPreservedPooling(int(dim * 2**2), cutoff=cutoff)
        else:
            self.down3_4 = Downsample(int(dim * 2**2))

        # Level 4 Latent
        self.latent = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**3),
                num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[3])
        ])

        # Upsample 4 -> 3: PixelShuffle or FreqAvgUp
        if self.up_mode == 'freqavg':
            self.up4_3 = FreqAvgUp(int(dim * 2**3), cutoff=cutoff)
        else:
            self.up4_3 = Upsample(int(dim * 2**3))

        # Decoder Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[2])
        ])

        # Decoder Level 2
        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])
        ])

        # Decoder Level 1
        self.up2_1 = Upsample(int(dim * 2**1))

        if self.reduce_top:
            self.reduce_chan_level1 = nn.Conv2d(int(dim * 2**1), dim, kernel_size=1, bias=bias)
            dec1_dim = dim
        else:
            dec1_dim = int(dim * 2**1)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dec1_dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[0])
        ])

        # Refinement Stage
        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=dec1_dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_refinement_blocks)
        ])

        # High-Frequency Branch (optional)
        if self.use_hf:
            self.hf_branch = HighFreqBranch(inp_channels=inp_channels, out_channels=out_channels, cutoff=cutoff)
        else:
            self.hf_branch = None

        # Dual pixel task support (backward compatibility)
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dec1_dim, kernel_size=1, bias=bias)

        # Output projection
        self.output = nn.Conv2d(dec1_dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # Level 1 Encoder
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        # Level 2 Encoder
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        # Level 3 Encoder
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        # Bottleneck Latent (Level 4)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # Level 3 Decoder
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        # Level 2 Decoder
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        # Level 1 Decoder
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        if self.reduce_top:
            inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # Refinement
        out_dec_level1 = self.refinement(out_dec_level1)

        # Residual R from decoder
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            residual_r = self.output(out_dec_level1)
        else:
            residual_r = self.output(out_dec_level1)

        # Final summation: R + I + (HF if enabled)
        if self.use_hf and self.hf_branch is not None:
            hf_feat = self.hf_branch(inp_img)
            out = residual_r + inp_img + hf_feat
        else:
            out = residual_r + inp_img

        return out
