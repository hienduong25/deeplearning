# BÁO CÁO NGHIỆM THU DỰ ÁN RESTORMER DOCUMENT SHADOW REMOVAL (V3)
**Ngày thực hiện:** 18/10/2026  
**Môi trường thử nghiệm:** Windows local (PyTorch 2.6.0 CPU) & Chuẩn hóa sẵn sàng cho Google Colab (1 GPU).

---

## 1. BẢNG ĐỐI CHIẾU FP / FREQAVGUP VỚI ĐẶC TẢ & BẢNG THAM SỐ (DIM=48)

### 1.1. Bảng đối chiếu đặc tả chi tiết của từng khối
| Module | Có Conv học được? | Có đường cao? | Có hệ số trộn học được? | Số tham số so với khối gốc (dim=48) |
| :--- | :---: | :---: | :---: | :---: |
| **Downsample (gốc)** | Có (192->96, 3x3) | Không | Không | 165,888 |
| **FP (bản v2 trước)** | Có (192->96, 3x3) | Không | Không | 165,888 (bằng gốc, thiếu $\alpha$ và đường cao) |
| **FP (đặc tả chuẩn v3)** | **Có (192->96, 3x3)** | **Có ($x_{conv} - x_{low}$ -> unshuffle)** | **Có ($\alpha$ khởi tạo 0.3)** | **165,889 (+1 tham số)** |
| **Upsample (gốc)** | Có (384->768, 3x3) | Không | Không | 2,654,208 |
| **FreqAvgUp (bản v2 trước)** | Không | Không | Không | 0 (MẤT 2,654,208 tham số!) |
| **FreqAvgUp (đặc tả chuẩn v3)** | **Có (384->768, 3x3)** | **Có ($x_{conv} - x_{mean}$ -> PixelShuffle)** | **Có ($\beta$ khởi tạo 0.3)** | **2,654,209 (+1 tham số)** |

### 1.2. Bảng tham số DocShadowNetV2 tại chiều thực (`dim=48`)
*(Khi tắt toàn bộ cờ, `DocShadowNetV2` trùng khớp tuyệt đối với Restormer gốc 26,126,644 tham số)*

| Biến thể / Cấu hình | Cờ kích hoạt | Số tham số thực (`dim=48`) | Chênh lệch so với Baseline |
| :--- | :--- | :---: | :---: |
| **Baseline (Flags OFF)** | `down='pixel'`, `up='pixel'`, `use_hf=False` | **26,126,644** | Gốc (0) |
| **+FP only** | `down='fp'`, `up='pixel'`, `use_hf=False` | **26,126,645** | +1 tham số ($\alpha$) |
| **+FreqAvgUp only** | `down='pixel'`, `up='freqavg'`, `use_hf=False` | **26,126,645** | +1 tham số ($\beta$) |
| **+FP & FreqAvgUp** | `down='fp'`, `up='freqavg'`, `use_hf=False` | **26,126,646** | +2 tham số ($\alpha, \beta$) |
| **+HF only** | `down='pixel'`, `up='pixel'`, `use_hf=True` | **26,129,812** | +3,168 tham số (CNN 3 lớp) |
| **Full (+FP, +FreqAvgUp, +HF)** | `down='fp'`, `up='freqavg'`, `use_hf=True` | **26,129,814** | +3,170 tham số |
| **Full with reduce_top=True** | Full + `reduce_top=True` | **25,440,390** | -686,254 tham số |

---

## 2. DANH SÁCH FILE ĐÃ XOÁ / CHUYỂN VÀ FILE ĐÃ TẠO

### 2.1. File đã xoá (`git rm`)
1. `Shadow_Removal/Options/DocShadowNet_RDD.yml`
2. `Shadow_Removal/Options/DocShadowNet_RDD_smoke.yml`
3. `Shadow_Removal/Options/Restormer_RDD_baseline.yml`
4. `Shadow_Removal/Options/Restormer_RDD_baseline_smoke.yml`
5. `Shadow_Removal/Options/Restormer_RDD_fft.yml`
6. `Shadow_Removal/Options/Restormer_RDD_fft_smoke.yml`
7. `Shadow_Removal/tests/test_docshadow_arch.py`
8. `Shadow_Removal/tests/test_train_pipeline.py`

### 2.2. File tài liệu đã chuyển vào `docs/`
- `docs/archive/phase0_report.md` (từ phase 0)
- `docs/archive/final_report.md` (từ phase 1-3 v1)
- `docs/final_report_v2.md` (báo cáo v2)

### 2.3. File mới tạo
- `Shadow_Removal/splits/val_files.txt` (danh sách cố định 200 tệp tập Val)
- `Shadow_Removal/tools/apply_split.py` (chuyển 200 tệp từ train sang val an toàn, idempotent)
- `Shadow_Removal/tools/pack_data.py` (đóng gói tệp zip chuẩn hoá `RDD_data_final.zip`)
- `Shadow_Removal/tools/check_progressive.py` (mô phỏng progressive learning và LR từ train.py)
- `Shadow_Removal/tools/probe_gpu.py` (thăm dò phần cứng GPU, đo VRAM, s/iter, dự đoán thời gian trên Colab)
- `Shadow_Removal/tools/make_budget_configs.py` (sinh 5 cấu hình theo tỉ lệ cho total_iter bất kỳ)
- `Shadow_Removal/tools/make_resume_yml.py` (sinh cấu hình resume tự động từ file .state)
- `Shadow_Removal/tools/prune_checkpoints.py` (tỉa checkpoint cũ, giữ lại N bản mới nhất trên Drive)
- `Shadow_Removal/tools/summarize_results.py` (tổng hợp bảng Ablation Markdown từ CSV)
- `Shadow_Removal/colab/setup_colab.sh` (tự động hoá môi trường Colab)
- `Shadow_Removal/colab/run_experiment.sh` (tự động symlink Drive, auto-resume và chạy train.py)
- `Shadow_Removal/colab/README_COLAB.md` (hướng dẫn chi tiết từng bước cho nhóm)
- `Shadow_Removal/colab/ASSIGNMENT.md` (phân công 2 đợt cho 3 người)
- `Shadow_Removal/test.py` (kiểm thử độc lập với padding bội số của 8 và ghi CSV)
- `Shadow_Removal/Options/budget_30000/` (5 cấu hình budget cho 30,000 steps)

---

## 3. NGUYÊN VĂN GIT DIFF TỪ COMMIT GỐC CỦA REPO

**Lệnh xác định commit gốc:**
```bash
$ git rev-list --max-parents=0 HEAD
608e8280a73cbe96026302919214c185269036d6
```

### 3.1. `git diff 608e8280a73cbe96026302919214c185269036d6 --stat -- basicsr/`
```text
 basicsr/data/data_util.py                 |   8 +-
 basicsr/metrics/__init__.py               |   4 +-
 basicsr/metrics/psnr_ssim.py              |  58 ++++-
 basicsr/models/archs/docshadow_arch.py    | 420 ++++++++++++++++++++++++++++++
 basicsr/models/image_restoration_model.py |  19 +-
 basicsr/models/losses/__init__.py         |   4 +-
 basicsr/models/losses/losses.py           |  26 ++
 basicsr/models/lr_scheduler.py            |   1 +
 basicsr/train.py                          |  13 +-
 9 files changed, 538 insertions(+), 15 deletions(-)
```

> [!IMPORTANT]
> `basicsr/models/archs/restormer_arch.py` **HOÀN TOÀN KHÔNG BỊ SỬA ĐỔI** (0 dòng diff).  
> 4 thư mục tác vụ gốc (`Motion_Deblurring`, `Defocus_Deblurring`, `Denoising`, `Deraining`) **HOÀN TOÀN NGUYÊN BẢN**.

### 3.2. Diff chi tiết các file đã sửa đổi trong `basicsr/`

```diff
diff --git a/basicsr/data/data_util.py b/basicsr/data/data_util.py
--- a/basicsr/data/data_util.py
+++ b/basicsr/data/data_util.py
@@ -232,18 +232,10 @@ def paired_paths_from_folder(folders, keys, filename_tmpl):
     input_paths = list(scandir(input_folder))
     gt_paths = list(scandir(gt_folder))
     if len(input_paths) != len(gt_paths):
-        # Graceful fallback: match by identical filename if present
-        common_names = sorted(list(set(input_paths) & set(gt_paths)))
-        if len(common_names) > 0:
-            paths = []
-            for name in common_names:
-                paths.append(
-                    dict([(f'{input_key}_path', osp.join(input_folder, name)),
-                          (f'{gt_key}_path', osp.join(gt_folder, name))]))
-            return paths
-        assert len(input_paths) == len(gt_paths), (
-            f'{input_key} and {gt_key} datasets have different number of images: '
-            f'{len(input_paths)}, {len(gt_paths)}.')
+        raise ValueError(
+            f'Mismatch in image counts: {input_key} folder "{input_folder}" has {len(input_paths)} images, '
+            f'while {gt_key} folder "{gt_folder}" has {len(gt_paths)} images. '
+            f'Dataset must be strictly 1-to-1 paired!')
     paths = []
     for idx in range(len(gt_paths)):
         gt_path = gt_paths[idx]

diff --git a/basicsr/metrics/__init__.py b/basicsr/metrics/__init__.py
--- a/basicsr/metrics/__init__.py
+++ b/basicsr/metrics/__init__.py
@@ -1,4 +1,4 @@
 from .niqe import calculate_niqe
-from .psnr_ssim import calculate_psnr, calculate_ssim
+from .psnr_ssim import calculate_psnr, calculate_ssim, calculate_rmse
 
-__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe']
+__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate_rmse']

diff --git a/basicsr/metrics/psnr_ssim.py b/basicsr/metrics/psnr_ssim.py
--- a/basicsr/metrics/psnr_ssim.py
+++ b/basicsr/metrics/psnr_ssim.py
@@ -160,10 +160,11 @@ def _ssim_3d(img1, img2, max_value):
     img1 = img1.astype(np.float64)
     img2 = img2.astype(np.float64)
 
-    kernel = _generate_3d_gaussian_kernel().cuda()
+    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
+    kernel = _generate_3d_gaussian_kernel().to(device)
 
-    img1 = torch.tensor(img1).float().cuda()
-    img2 = torch.tensor(img2).float().cuda()
+    img1 = torch.tensor(img1).float().to(device)
+    img2 = torch.tensor(img2).float().to(device)
 
 
     mu1 = _3d_gaussian_calculator(img1, kernel)
@@ -301,3 +302,54 @@ def calculate_ssim(img1,
     return np.array(ssims).mean()
+
+def calculate_rmse(img1,
+                   img2,
+                   crop_border,
+                   input_order='HWC',
+                   test_y_channel=False):
+    """Calculate RMSE (Root Mean Squared Error)."""
+    assert img1.shape == img2.shape, (
+        f'Image shapes are different: {img1.shape}, {img2.shape}.')
+    if input_order not in ['HWC', 'CHW']:
+        raise ValueError(
+            f'Wrong input_order {input_order}. Supported input_orders are '
+            '"HWC" and "CHW"')
+    if type(img1) == torch.Tensor:
+        if len(img1.shape) == 4:
+            img1 = img1.squeeze(0)
+        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
+    if type(img2) == torch.Tensor:
+        if len(img2.shape) == 4:
+            img2 = img2.squeeze(0)
+        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
+
+    img1 = reorder_image(img1, input_order=input_order)
+    img2 = reorder_image(img2, input_order=input_order)
+    img1 = img1.astype(np.float64)
+    img2 = img2.astype(np.float64)
+
+    if crop_border != 0:
+        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
+        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
+
+    if test_y_channel:
+        img1 = to_y_channel(img1)
+        img2 = to_y_channel(img2)
+
+    mse = np.mean((img1 - img2) ** 2)
+    return np.sqrt(mse)

diff --git a/basicsr/models/image_restoration_model.py b/basicsr/models/image_restoration_model.py
--- a/basicsr/models/image_restoration_model.py
+++ b/basicsr/models/image_restoration_model.py
@@ -108,6 +108,14 @@ class ImageCleanModel(BaseModel):
         else:
             raise ValueError('pixel loss are None.')
 
+        # optional FFT loss (backward-compatible: absent key → no FFT loss)
+        self.cri_fft = None
+        if train_opt.get('fft_opt'):
+            fft_type = train_opt['fft_opt'].pop('type')
+            cri_fft_cls = getattr(loss_module, fft_type)
+            self.cri_fft = cri_fft_cls(**train_opt['fft_opt']).to(
+                self.device)
+
         # set up optimizers and schedulers
         self.setup_optimizers()
         self.setup_schedulers()
@@ -161,8 +169,17 @@ class ImageCleanModel(BaseModel):
             l_pix += self.cri_pix(pred, self.gt)
 
         loss_dict['l_pix'] = l_pix
+        l_total = l_pix
+
+        # optional FFT loss
+        if self.cri_fft is not None:
+            l_fft = 0.
+            for pred in preds:
+                l_fft += self.cri_fft(pred, self.gt)
+            loss_dict['l_fft'] = l_fft
+            l_total = l_total + l_fft
 
-        l_pix.backward()
+        l_total.backward()
         if self.opt['train']['use_grad_clip']:
             torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
         self.optimizer_g.step()

diff --git a/basicsr/models/losses/__init__.py b/basicsr/models/losses/__init__.py
--- a/basicsr/models/losses/__init__.py
+++ b/basicsr/models/losses/__init__.py
@@ -1,5 +1,5 @@
-from .losses import (L1Loss, MSELoss, PSNRLoss, CharbonnierLoss)
+from .losses import (L1Loss, MSELoss, PSNRLoss, CharbonnierLoss, FFTLoss)
 
 __all__ = [
-    'L1Loss', 'MSELoss', 'PSNRLoss', 'CharbonnierLoss',
+    'L1Loss', 'MSELoss', 'PSNRLoss', 'CharbonnierLoss', 'FFTLoss',
 ]

diff --git a/basicsr/models/losses/losses.py b/basicsr/models/losses/losses.py
--- a/basicsr/models/losses/losses.py
+++ b/basicsr/models/losses/losses.py
@@ -120,3 +120,29 @@ class CharbonnierLoss(nn.Module):
         loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
         return loss
+
+class FFTLoss(nn.Module):
+    """L1 loss between frequency spectra (real + imaginary)."""
+    def __init__(self, loss_weight=0.1, reduction='mean'):
+        super(FFTLoss, self).__init__()
+        self.loss_weight = loss_weight
+        self.criterion = nn.L1Loss(reduction=reduction)
+
+    def forward(self, pred, target, **kwargs):
+        pred_f32 = pred.float()
+        target_f32 = target.float()
+        pf = torch.view_as_real(torch.fft.rfft2(pred_f32))
+        tf = torch.view_as_real(torch.fft.rfft2(target_f32))
+        return self.loss_weight * self.criterion(pf, tf)

diff --git a/basicsr/models/lr_scheduler.py b/basicsr/models/lr_scheduler.py
--- a/basicsr/models/lr_scheduler.py
+++ b/basicsr/models/lr_scheduler.py
@@ -131,6 +131,7 @@ def get_position_from_periods(iteration, cumulative_period):
     for i, period in enumerate(cumulative_period):
         if iteration <= period:
             return i
+    return len(cumulative_period) - 1 

diff --git a/basicsr/train.py b/basicsr/train.py
--- a/basicsr/train.py
+++ b/basicsr/train.py
@@ -150,10 +150,15 @@ def main():
 
     # load resume states if necessary
     if opt['path'].get('resume_state'):
-        device_id = torch.cuda.current_device()
-        resume_state = torch.load(
-            opt['path']['resume_state'],
-            map_location=lambda storage, loc: storage.cuda(device_id))
+        if torch.cuda.is_available() and opt.get('num_gpu', 1) > 0:
+            device_id = torch.cuda.current_device()
+            resume_state = torch.load(
+                opt['path']['resume_state'],
+                map_location=lambda storage, loc: storage.cuda(device_id))
+        else:
+            resume_state = torch.load(
+                opt['path']['resume_state'],
+                map_location='cpu')
     else:
         resume_state = None
```

---

## 4. OUTPUT NGUYÊN VĂN TỪ CÁC TẬP LỆNH KIỂM CHỨNG

### 4.1. Output nguyên văn `Shadow_Removal/tools/check_progressive.py`
*(Trích đoạn logic từ `basicsr/train.py:L223` và `L247-L254`:)*
```python
groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])
...
j = ((current_iter > groups) != True).nonzero()[0]
if len(j) == 0:
    bs_j = len(groups) - 1
else:
    bs_j = j[0]

mini_gt_size = mini_gt_sizes[bs_j]
mini_batch_size = mini_batch_sizes[bs_j]
```

**Console Output:**
```text
===========================================================================
Checking Progressive Learning Config: Shadow_Removal/Options/1_Restormer_baseline.yml
===========================================================================
Schedule Assertion PASSED: sum(iters) == total_iter == sum(periods) == 60000

--- Progressive Learning Stages ---
Giai đoạn    | Iter bắt đầu   | Iter kết thúc  | Patch (px) | Batch  | Điểm ảnh/iter 
-------------------------------------------------------------------------------------
Stage 1      | 1              | 20000          | 128        | 16     | 262,144       
Stage 2      | 20001          | 34000          | 160        | 10     | 256,000       
Stage 3      | 34001          | 48000          | 192        | 8      | 294,912       
Stage 4      | 48001          | 60000          | 256        | 4      | 262,144       

--- Point-by-Point Stage Verification (train.py logic) ---
  Iter      1 -> Stage 1 (Patch: 128, Batch: 16)
  Iter  10000 -> Stage 1 (Patch: 128, Batch: 16)
  Iter  20000 -> Stage 1 (Patch: 128, Batch: 16)
  Iter  20001 -> Stage 2 (Patch: 160, Batch: 10)
  Iter  34000 -> Stage 2 (Patch: 160, Batch: 10)
  Iter  34001 -> Stage 3 (Patch: 192, Batch: 8)
  Iter  48000 -> Stage 3 (Patch: 192, Batch: 8)
  Iter  48001 -> Stage 4 (Patch: 256, Batch: 4)
  Iter  60000 -> Stage 4 (Patch: 256, Batch: 4)

--- Real Learning Rate at Key Checkpoints ---
  Iter      0 -> Learning Rate: 0.00030000
  Iter  20000 -> Learning Rate: 0.00030000
  Iter  20001 -> Learning Rate: 0.00030000
  Iter  39999 -> Learning Rate: 0.00015051
  Iter  60000 -> Learning Rate: 0.00000100

All progressive checks PASSED for Shadow_Removal/Options/1_Restormer_baseline.yml!
(Đã kiểm tra và PASS tương tự trên cả 5 yml).
```

### 4.2. Output nguyên văn `Shadow_Removal/tests/test_configs_consistency.py`
```text
Testing config consistency across 5 training configurations...

Checking: Shadow_Removal/Options/1_Restormer_baseline.yml
  total_iter       : 60000
  iters            : [20000, 14000, 14000, 12000] -> sum = 60000
  periods          : [20000, 40000] -> sum = 60000
  gt_sizes         : [128, 160, 192, 256]
  mini_batch_sizes : [16, 10, 8, 4]
  -> Schedule & common fields: OK

Checking: Shadow_Removal/Options/2_Restormer_fft.yml
  total_iter       : 60000
  iters            : [20000, 14000, 14000, 12000] -> sum = 60000
  periods          : [20000, 40000] -> sum = 60000
  gt_sizes         : [128, 160, 192, 256]
  mini_batch_sizes : [16, 10, 8, 4]
  -> Schedule & common fields: OK

Checking: Shadow_Removal/Options/3_DocShadowNetV2_fp_freqavg.yml
  total_iter       : 60000
  iters            : [20000, 14000, 14000, 12000] -> sum = 60000
  periods          : [20000, 40000] -> sum = 60000
  gt_sizes         : [128, 160, 192, 256]
  mini_batch_sizes : [16, 10, 8, 4]
  -> Schedule & common fields: OK

Checking: Shadow_Removal/Options/4_DocShadowNetV2_hf.yml
  total_iter       : 60000
  iters            : [20000, 14000, 14000, 12000] -> sum = 60000
  periods          : [20000, 40000] -> sum = 60000
  gt_sizes         : [128, 160, 192, 256]
  mini_batch_sizes : [16, 10, 8, 4]
  -> Schedule & common fields: OK

Checking: Shadow_Removal/Options/5_DocShadowNetV2_full.yml
  total_iter       : 60000
  iters            : [20000, 14000, 14000, 12000] -> sum = 60000
  periods          : [20000, 40000] -> sum = 60000
  gt_sizes         : [128, 160, 192, 256]
  mini_batch_sizes : [16, 10, 8, 4]
  -> Schedule & common fields: OK

--- Pairwise Ablation Consistency Checks ---
  Config 1 -> Config 2 (+FFT): exactly only added FFTLoss: OK
  Config 2 -> Config 3 (+FP & +FreqAvgUp): exactly only changed down_mode='fp' & up_mode='freqavg': OK
  Config 2 -> Config 4 (+HF only): exactly only changed use_hf=True: OK
  Config 3 -> Config 5 (+HF addition): exactly only changed use_hf: OK
  Config 4 -> Config 5 (+FP & +FreqAvgUp addition): exactly only changed down_mode & up_mode: OK

============================================================
ALL 5 CONFIGURATIONS ARE STRICTLY CONSISTENT AND VERIFIED!
============================================================
```

### 4.3. Output nguyên văn `Shadow_Removal/tests/test_docshadownet_v2.py`
```text
=== Test 1: FreqAvgUp spectral upsampling on constant image ===
  Constant value    0.0000 -> Max difference: 0.00000000
  Constant value    1.0000 -> Max difference: 0.00000000
  Constant value    3.1416 -> Max difference: 0.00000000
  Constant value   -2.5000 -> Max difference: 0.00000000
  Constant value  255.0000 -> Max difference: 0.00000000
FreqAvgUp constant preservation test PASSED!

=== Test 2: Shape & Parameter count of FP and FreqAvgUp vs Downsample and Upsample ===
  Downsample original output shape : torch.Size([2, 384, 16, 16]) | Params: 165,888
  FrequencyPreservedPooling output : torch.Size([2, 384, 16, 16]) | Params: 165,889
  Upsample original output shape   : torch.Size([2, 192, 32, 32]) | Params: 2,654,208
  FreqAvgUp output shape           : torch.Size([2, 192, 32, 32]) | Params: 2,654,209
FP and FreqAvgUp shape & +1 param comparison test PASSED!

=== Test 3: DocShadowNetV2 with all flags OFF vs Restormer Original ===
  Restormer original parameters   : 26,126,644
  DocShadowNetV2 (flags OFF) params: 26,126,644
  Output shape match: torch.Size([1, 3, 64, 64]) == torch.Size([1, 3, 64, 64])
Parameter and shape identity test PASSED!

=== Test 4: DocShadowNetV2 Ablation Table at dim=48 (Real Dimension) ===
  [Baseline (all flags off)      ] Real Params (dim=48): 26,126,644
  [+FP only                      ] Real Params (dim=48): 26,126,645
  [+FreqAvgUp only               ] Real Params (dim=48): 26,126,645
  [+FP & FreqAvgUp               ] Real Params (dim=48): 26,126,646
  [+HF only                      ] Real Params (dim=48): 26,129,812
  [Full (+FP, +FreqAvgUp, +HF)   ] Real Params (dim=48): 26,129,814
  [Full with reduce_top=True     ] Real Params (dim=48): 25,440,390

Checking Forward & Backward for all variants with dim=16:
  [Baseline (all flags off)      ] Params:  738,888 | Out: [1, 3, 64, 64] | Fwd/Bwd: OK
  [+FP only                      ] Params:  738,889 | Out: [1, 3, 64, 64] | Fwd/Bwd: OK
  [+FreqAvgUp only               ] Params:  738,889 | Out: [1, 3, 64, 64] | Fwd/Bwd: OK
  [+FP & FreqAvgUp               ] Params:  738,890 | Out: [1, 3, 64, 64] | Fwd/Bwd: OK
  [+HF only                      ] Params:  742,056 | Out: [1, 3, 64, 64] | Fwd/Bwd: OK
  [Full (+FP, +FreqAvgUp, +HF)   ] Params:  742,058 | Out: [1, 3, 64, 64] | Fwd/Bwd: OK
  [Full with reduce_top=True     ] Params:  724,634 | Out: [1, 3, 64, 64] | Fwd/Bwd: OK
All ablation variants PASSED!

============================================================
ALL DocShadowNetV2 TESTS PASSED SUCCESSFULLY!
============================================================
```

### 4.4. Output nguyên văn toàn bộ các bài test giữ lại
```text
Testing FFTLoss...
  Forward loss value: 10.218247
  Zero loss value (identical inputs): 0.00000000
  Backward pass: OK (gradients computed, no NaNs)
  Odd dimensions test: OK
ALL FFTLoss tests PASSED!

Testing calculate_rmse...
  Identical uint8 RMSE: 0.000000
  Known diff RMSE (expected 10.0): 10.000000
  Torch tensor RMSE: 0.100000
  Crop border test: OK
ALL calculate_rmse tests PASSED!

Testing backward compatibility...
  Original Restormer instantiation and forward: OK
  Loss module exports check: OK
  Original L1Loss behavior: OK
ALL backward compatibility tests PASSED!

Testing Dataset_PairedImage with RDD data...
  Item 0: lq shape=torch.Size([3, 128, 128]), gt shape=torch.Size([3, 128, 128])
  Full RDD train dataset pairs loaded: 4171
ALL dataset pairing tests PASSED!

Testing data leakage between train, val, test...
  Train images: 4171
  Val   images: 200
  Test  images: 545
  Train & Val  overlap: 0
  Train & Test overlap: 0
  Val   & Test overlap: 0
ZERO data leakage detected! All splits are strictly disjoint.
```

### 4.5. Output nguyên văn `Shadow_Removal/test.py` chạy trên Checkpoint Smoke
```text
======================================================================
RUNNING EVALUATION ON [cpu]
Config   : Shadow_Removal/Options/smoke_baseline.yml
Weights  : experiments/smoke_baseline/models/net_g_latest.pth
Split    : val
======================================================================
Network created: Restormer (738,888 parameters)
Total images to evaluate: 10
  [2/10] 0246.JPG                  | PSNR: 12.05 dB | SSIM: 0.5598 | RMSE: 63.69 | Time: 4.885s
  [4/10] 1920.JPG                  | PSNR: 17.17 dB | SSIM: 0.5219 | RMSE: 35.32 | Time: 0.120s
  [6/10] 2114.JPG                  | PSNR: 6.94 dB | SSIM: 0.2150 | RMSE: 114.71 | Time: 0.108s
  [8/10] 5026.jpg                  | PSNR: 25.30 dB | SSIM: 0.8448 | RMSE: 13.86 | Time: 0.104s
  [10/10] 5399.jpg                  | PSNR: 10.61 dB | SSIM: 0.4258 | RMSE: 75.19 | Time: 0.139s

======================================================================
EVALUATION COMPLETED SUCCESSFULLY
======================================================================
Results CSV : results/smoke_baseline_val\metrics.csv
Images Dir  : results/smoke_baseline_val\restored
Total Evaluated : 10 images
Average PSNR    : 14.3975 dB
Average SSIM    : 0.5870
Average RMSE    : 56.2284
Average Speed   : 1.9147 sec/image
======================================================================
```
*(Bảng tổng hợp từ `Shadow_Removal/tools/summarize_results.py`:)*
```text
# BẢNG TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ (ABLATION SUMMARY)

| Thí nghiệm / Mô hình | Số ảnh | PSNR (dB) ↑ | SSIM ↑ | RMSE ↓ | Giây / ảnh |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **smoke_baseline_val** | 10 | 14.40 ± 5.32 | 0.5870 ± 0.1834 | 56.23 ± 29.10 | 1.915s |
```

---

## 5. MỤC "CHƯA KIỂM TRA" & CÁC ĐIỂM CHƯA THỰC HIỆN

Các hạng mục sau được ghi nhận rõ ràng do giới hạn máy local chỉ có CPU:

1. **CHƯA KIỂM TRA TRÊN GPU**:
   - Mức tiêu thụ VRAM đỉnh thực tế trên GPU (T4 / V100 / A100) ở Stage 4 (patch 256x256, batch 4).
   - Tốc độ huấn luyện thật (giây/iter) trên GPU. Đã chuẩn bị sẵn công cụ `Shadow_Removal/tools/probe_gpu.py` để chạy ngay khi lên Colab.
2. **CHƯA KIỂM TRA TRÊN GOOGLE DRIVE**:
   - Thao tác mount Google Drive thật và symlink `experiments -> /content/drive/MyDrive/...` trong môi trường Colab thật.
   - Các bash scripts (`setup_colab.sh`, `run_experiment.sh`) đã được viết chuẩn cú pháp POSIX bash nhưng chỉ chạy thực sự khi kết nối phiên Colab.
3. **CHƯA LÀM (THEO YÊU CẦU)**:
   - **OCR CER / WER**: Đánh giá độ chính xác nhận dạng văn bản (Character/Word Error Rate) sau khi khử bóng chưa được thực hiện vì cần cài đặt mô hình OCR (Tesseract / EasyOCR / PaddleOCR); nhóm sẽ thực hiện ở đợt đánh giá cuối cùng.
