# 📋 Giai đoạn 0: Báo cáo Đọc Code

## 1. Kiến trúc mạng — [`restormer_arch.py`](file:///c:/Restormer/Restormer-main/basicsr/models/archs/restormer_arch.py)

### Các lớp chính
| Lớp | Chức năng |
|-----|-----------|
| `TransformerBlock` | `__init__(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)` — Norm→MDTA→Norm→GDFN |
| `OverlapPatchEmbed` | Conv2d 3×3, `__init__(in_c=3, embed_dim=48, bias=False)` |
| `Downsample` | Conv 3×3 (n_feat → n_feat//2) rồi `PixelUnshuffle(2)` → đầu ra **2×n_feat, H/2, W/2** |
| `Upsample` | Conv 3×3 (n_feat → n_feat*2) rồi `PixelShuffle(2)` → đầu ra **n_feat//2, 2H, 2W** |
| `Restormer` | UNet 4 tầng, xem chi tiết bên dưới |

### `Restormer.__init__` nhận:
```
inp_channels=3, out_channels=3, dim=48,
num_blocks=[4,6,6,8], num_refinement_blocks=4,
heads=[1,2,4,8], ffn_expansion_factor=2.66,
bias=False, LayerNorm_type='WithBias',
dual_pixel_task=False
```

### Luồng forward:
```
inp_img → patch_embed → encoder_level1 (dim=48)
  ↓ down1_2 → encoder_level2 (dim=96)
  ↓ down2_3 → encoder_level3 (dim=192)
  ↓ down3_4 → latent (dim=384)
  ↑ up4_3 → cat(skip3) → reduce_chan_level3 (1×1: 384→192) → decoder_level3
  ↑ up3_2 → cat(skip2) → reduce_chan_level2 (1×1: 192→96)  → decoder_level2
  ↑ up2_1 → cat(skip1) → decoder_level1 (dim=96, KHÔNG có reduce_chan)
  → refinement (dim=96)
  → output Conv 3×3 (96→3) + inp_img  (residual learning)
```

> [!IMPORTANT]
> Tầng 1 decoder hoạt động trên **2×dim = 96 kênh** (concat skip 48 + upsample 48), KHÔNG có 1×1 conv giảm kênh. Refinement cũng 96 kênh. Đây là khác biệt chính với tầng 2, 3.

### Đăng ký mạng — [`archs/__init__.py`](file:///c:/Restormer/Restormer-main/basicsr/models/archs/__init__.py)
- **Tự động scan** tất cả file `*_arch.py` trong thư mục `archs/`, import hết.
- `define_network(opt)`: pop `type` → dùng `dynamic_instantiation` → `getattr(module, cls_type)` → `cls_(**opt)`.
- ✅ Chỉ cần tạo file `docshadow_arch.py` trong thư mục `archs/` là tự nhận `DocShadowNet`.

---

## 2. Loss — [`losses.py`](file:///c:/Restormer/Restormer-main/basicsr/models/losses/losses.py) & [`losses/__init__.py`](file:///c:/Restormer/Restormer-main/basicsr/models/losses/__init__.py)

### Các loss có sẵn
- `L1Loss(loss_weight=1.0, reduction='mean')` — dùng `@weighted_loss` decorator
- `MSELoss`, `PSNRLoss`, `CharbonnierLoss`

### Export bằng `__init__.py`
```python
from .losses import (L1Loss, MSELoss, PSNRLoss, CharbonnierLoss)
```
- **KHÔNG tự động scan** — phải thêm tay vào `__init__.py` khi thêm loss mới.

### Cách model gọi loss
Trong `init_training_settings` ([L102-109](file:///c:/Restormer/Restormer-main/basicsr/models/image_restoration_model.py#L102-L109)):
```python
pixel_type = train_opt['pixel_opt'].pop('type')
cri_pix_cls = getattr(loss_module, pixel_type)
self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
```
- `loss_module = importlib.import_module('basicsr.models.losses')` — lấy từ `__init__.py`.

---

## 3. Model Training — [`image_restoration_model.py`](file:///c:/Restormer/Restormer-main/basicsr/models/image_restoration_model.py)

### Lớp `ImageCleanModel(BaseModel)`

| Method | Hành vi |
|--------|---------|
| `__init__` | Mixing augment (nếu bật), `define_network`, load pretrained, `init_training_settings` |
| `init_training_settings` | EMA (nếu có), pixel loss (bắt buộc), optimizers, schedulers |
| `optimize_parameters` | `preds = net_g(lq)` → nếu không phải list thì wrap thành `[preds]` → `output = preds[-1]` → cộng L1 trên **tất cả** preds → **backward trực tiếp trên `l_pix`** → grad clip → step |
| `nondist_validation` | Dùng `pad_test(window_size=8)` hoặc `nonpad_test` → `tensor2img` → gọi metric qua `getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)` |

> [!NOTE]
> `optimize_parameters` hiện backward **trực tiếp** trên `l_pix`, không qua biến `l_total`. Khi thêm FFT loss, cần backward trên tổng.

---

## 4. Metrics — [`psnr_ssim.py`](file:///c:/Restormer/Restormer-main/basicsr/metrics/psnr_ssim.py) & [`metrics/__init__.py`](file:///c:/Restormer/Restormer-main/basicsr/metrics/__init__.py)

### `calculate_psnr(img1, img2, crop_border, input_order='HWC', test_y_channel=False)`
- Nhận **numpy ndarray HWC [0,255]** hoặc **torch.Tensor CHW [0,1]** (tự convert Tensor→numpy).
- Range tự detect: `max_value = 1 if img1.max() <= 1 else 255`.

### `calculate_ssim` — cùng signature, dùng 3D Gaussian SSIM trên GPU.

### Export:
```python
from .psnr_ssim import calculate_psnr, calculate_ssim
```
- **KHÔNG tự động scan** — phải thêm `calculate_rmse` vào `__init__.py` thủ công.

---

## 5. Dataset — [`paired_image_dataset.py`](file:///c:/Restormer/Restormer-main/basicsr/data/paired_image_dataset.py)

### `Dataset_PairedImage`
- Nhận `dataroot_gt`, `dataroot_lq` → ghép cặp bằng [`paired_paths_from_folder`](file:///c:/Restormer/Restormer-main/basicsr/data/data_util.py#L208-L251).
- `filename_tmpl`: mặc định `'{}'` — tên file img và gt phải **giống nhau** (chỉ cần cùng basename).
- Ghép cặp: scan cả 2 folder → assert **cùng số file** → dùng sorted order + template.
- Phase train: `padding` → `paired_random_crop(gt_size)` → `random_augmentation` (flip/rot).
- Return: `{'lq': tensor, 'gt': tensor, 'lq_path', 'gt_path'}`.
- `io_backend.type: disk` → dùng `FileClient`.

> [!WARNING]
> `paired_paths_from_folder` assert `len(input_paths) == len(gt_paths)` — RDD data phải đảm bảo img/ và gt/ có **cùng số file**.

---

## 6. Training Loop — [`train.py`](file:///c:/Restormer/Restormer-main/basicsr/train.py)

### Progressive learning keys (tên khoá THẬT):
```yaml
datasets.train:
  iters: [92000,64000,48000,36000,36000,24000]     # thời lượng mỗi giai đoạn
  mini_batch_sizes: [8,5,4,2,1,1]                    # batch size tương ứng
  gt_size: 384                                       # kích thước patch tối đa (dùng cho DataLoader)
  gt_sizes: [128,160,192,256,320,384]                # patch size theo giai đoạn
```

### Logic ([L212-269](file:///c:/Restormer/Restormer-main/basicsr/train.py#L212-L269)):
```python
groups = np.array([sum(iters[0:i+1]) for i in range(len(iters))])  # cumulative
j = ((current_iter > groups) != True).nonzero()[0]
bs_j = j[0] if len(j) > 0 else len(groups)-1
mini_gt_size = mini_gt_sizes[bs_j]      # ← gt_sizes, KHÔNG PHẢI gt_size
mini_batch_size = mini_batch_sizes[bs_j]
```
- Nếu `mini_batch_size < batch_size`: subsample indices.
- Nếu `mini_gt_size < gt_size`: random crop nhỏ hơn từ patch đã crop.

### Scheduler, val, checkpoint:
```yaml
train.total_iter: 300000
train.scheduler.periods: [92000, 208000]    # sum = 300000
val.val_freq: 4000
logger.save_checkpoint_freq: 4000
```

### Auto-resume:
- Tự tìm file `*.state` mới nhất trong `experiments/{name}/training_states/`.

---

## 7. Test Script — [`Motion_Deblurring/test.py`](file:///c:/Restormer/Restormer-main/Motion_Deblurring/test.py)
- Load YAML → pop `type` → `Restormer(**x['network_g'])` **(hardcode class)**.
- Load checkpoint: `checkpoint['params']`.
- Padding lên bội số 8 → forward → unpad → clamp → save.
- **KHÔNG tính metric** — chỉ lưu ảnh.

---

## 8. Kiểm chứng suy đoán

| Suy đoán của bạn | Thực tế | Đúng/Sai |
|---|---|---|
| Tên lớp `Restormer` | ✅ Đúng | ✅ |
| `TransformerBlock` | ✅ Đúng | ✅ |
| `Downsample` dùng PixelUnshuffle | ✅ Đúng (`nn.PixelUnshuffle(2)`) | ✅ |
| `Upsample` dùng PixelShuffle | ✅ Đúng (`nn.PixelShuffle(2)`) | ✅ |
| `OverlapPatchEmbed` | ✅ Đúng | ✅ |
| `Dataset_PairedImage` ghép theo tên file | ✅ Đúng (qua `filename_tmpl='{}'`) | ✅ |
| `filename_tmpl` | ✅ Đúng, mặc định `'{}'` | ✅ |
| `L1Loss` trong losses | ✅ Đúng | ✅ |
| Loss tìm theo `getattr(loss_module, type)` | ✅ Đúng | ✅ |
| `calculate_psnr` nhận ndarray | ✅ Đúng, **cũng nhận Tensor** (auto convert) | ✅ |
| Network đăng ký bằng `network_g.type` | ✅ Đúng (pop `type` → `dynamic_instantiation`) | ✅ |
| Progressive keys: `gt_sizes`, `mini_batch_sizes`, `iters` | ✅ Đúng — chính xác | ✅ |
| Model class `ImageCleanModel` | ✅ Đúng (trong yml là `model_type: ImageCleanModel`) | ✅ |
| `init_training_settings` có pixel loss | ✅ Đúng, **bắt buộc** (raise nếu None) | ✅ |
| `optimize_parameters` gọi `net_g(lq)` → `preds` | ✅ Đúng, wrap list nếu cần | ✅ |
| Validation dùng `metric_module` + `getattr` | ✅ Đúng | ✅ |
| `test.py` pad bội 8 | ✅ Đúng (`factor = 8`) | ✅ |

> [!TIP]
> Tất cả 16 suy đoán đều **ĐÚNG**. Không có khác biệt nào cần điều chỉnh.

---

## 9. Quan sát quan trọng cho Giai đoạn 1–3

1. **Tầng 1 decoder dùng 2×dim kênh** — `decoder_level1` và `refinement` cùng dùng `dim*2=96` kênh, KHÔNG có 1×1 conv giảm kênh.
2. **`optimize_parameters` backward trực tiếp** trên `l_pix` — cần sửa thành backward trên `l_total = l_pix + l_fft`.
3. **Loss registry là thủ công** — phải thêm FFTLoss vào `losses/__init__.py`.
4. **Metric registry thủ công** — thêm `calculate_rmse` vào `metrics/__init__.py`.
5. **Arch registry tự động** — chỉ cần file `*_arch.py` trong thư mục `archs/`.
6. **`paired_paths_from_folder` assert cùng số file** — cần `check_pairs.py` xác nhận.
7. **`test.py` hardcode `Restormer`** — shadow test.py phải import đúng lớp hoặc dùng `define_network`.
8. **`train.py` đọc `gt_sizes`** (KHÔNG PHẢI `gt_size` cho progressive) — `gt_size` là kích thước patch DataLoader (cần ≥ max(gt_sizes)).

---

## 10. Câu hỏi mở

1. **RDD data format**: File ảnh là `.png` hay `.jpg`? Cần xác nhận để đảm bảo `paired_paths_from_folder` scan đúng (nó dùng `scandir` scan mọi file).
2. **Scale**: Restormer gốc dùng `scale: 1` (không super-resolution). Shadow removal cũng `scale: 1` — OK?
3. **Val window_size**: Giữ `window_size: 8` cho validation? (pad lên bội 8).
4. **`use_image: true`** trong val: dùng uint8 numpy để tính metric (range [0,255]). Hay dùng tensor float [0,1]?

---

*Giai đoạn 0 hoàn tất. Chưa sửa/tạo file nào. Chờ xác nhận để sang Giai đoạn 1.*
