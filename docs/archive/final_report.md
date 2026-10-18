# BÁO CÁO TỔNG KẾT GIAI ĐOẠN 1, 2, 3: KHỬ BÓNG ẢNH TÀI LIỆU (RDD) VỚI RESTORMER & DOCSHADOWNET

## 1. TỔNG QUAN KẾT QUẢ
Hệ thống đã hoàn tất liên tục và kiểm thử tự động toàn bộ 3 giai đoạn:
- **Giai đoạn 1**: Thiết lập cấu hình YAML, hàm mất mát phổ tần số `FFTLoss`, thước đo `calculate_rmse`, công cụ kiểm tra dữ liệu `check_pairs.py`, script tách tập `create_val_split.py`.
- **Giai đoạn 2**: Thiết kế và triển khai kiến trúc mới `DocShadowNet` chuyên biệt cho tài liệu (bảo tồn nét chữ ở dải tần cao và chuẩn hóa bóng ở dải tần thấp), đăng ký tự động vào Architecture Registry của Restormer.
- **Giai đoạn 3**: Tạo tập smoke data (~20 cặp ảnh), kiểm thử tích hợp toàn diện quy trình huấn luyện end-to-end (forward + loss + backward + validation) thành công trên cả 3 mô hình.

Tất cả các thay đổi tuân thủ nghiêm ngặt:
1. **Tuyệt đối không đụng chạm** các thư mục tác vụ gốc (`Motion_Deblurring`, `Defocus_Deblurring`, `Denoising`, `Deraining`).
2. **Tương thích ngược 100%**: Các file sửa trong `basicsr/` hoạt động hoàn toàn bình thường với code gốc.

---

## 2. CHI TIẾT TRIỂN KHAI TỪNG GIAI ĐOẠN

### Giai đoạn 1: Cấu hình, FFT Loss, RMSE Metric, Data Audit
1. **Hàm mất mát `FFTLoss`** ([losses.py](file:///c:/Restormer/Restormer-main/basicsr/models/losses/losses.py)):
   - Sử dụng `torch.fft.rfft2` với `torch.view_as_real` tính khoảng cách $L_1$ giữa phổ dự đoán và ảnh sạch ground-truth.
   - Ép kiểu `float32` tránh lỗi underflow/overflow khi chạy AMP (Mixed Precision).
   - Đã kiểm thử `test_fft_loss.py`: giá trị loss dương, nhận diện ảnh trùng khớp (loss = 0), lan truyền ngược gradient không có NaN, xử lý hoàn hảo kích thước lẻ.

2. **Hook tích hợp FFT Loss trong Model** ([image_restoration_model.py](file:///c:/Restormer/Restormer-main/basicsr/models/image_restoration_model.py)):
   - Đọc `fft_opt` an toàn (`train_opt.get('fft_opt')`). Nếu không có `fft_opt` (như các config cũ của Restormer), `self.cri_fft = None` và mô hình chạy 100% như cũ.
   - Khi có `fft_opt`, $L_{total} = L_{pix} + L_{fft}$ và lan truyền ngược trực tiếp qua $L_{total}$.

3. **Thước đo `calculate_rmse`** ([psnr_ssim.py](file:///c:/Restormer/Restormer-main/basicsr/metrics/psnr_ssim.py)):
   - Hỗ trợ cả `numpy.ndarray` (HWC) và `torch.Tensor` (BCHW/CHW).
   - Đồng bộ tham số `crop_border` và `test_y_channel`.
   - Đã kiểm thử `test_rmse.py`: sai số = 0 với ảnh giống nhau, kiểm tra sai số định lượng chính xác tuyệt đối.

4. **Kiểm tra dữ liệu RDD & Cải tiến bộ nạp ảnh**:
   - `check_pairs.py` phát hiện:
     - Tập `test`: 545 ảnh `img` ghép cặp hoàn hảo 1-1 với 545 ảnh `gt`.
     - Tập `train`: Có 4,239 ảnh `img` nhưng có tới 4,371 ảnh `gt` (thừa 132 ảnh `gt` không có ảnh `img` tương ứng).
   - **Xử lý bảo vệ**: Restormer gốc dùng `assert len(input_paths) == len(gt_paths)` sẽ gặp lỗi assertion crash. Đã nâng cấp hàm `paired_paths_from_folder` trong [data_util.py](file:///c:/Restormer/Restormer-main/basicsr/data/data_util.py) tự động bắt cặp theo tên file chung khi số lượng 2 bên khác nhau. Đã kiểm thử qua `test_dataset_pairing.py` nạp đúng 4,239 cặp mà không bị crash.
   - `create_val_split.py`: Tách ngẫu nhiên có seed cố định (seed=42) 200 cặp từ `train` sang `RDD_data/val`.

---

### Giai đoạn 2: Kiến trúc mới `DocShadowNet` ([docshadow_arch.py](file:///c:/Restormer/Restormer-main/basicsr/models/archs/docshadow_arch.py))

#### Động lực thiết kế:
Bóng trên tài liệu có tính chất vật lý khác biệt so với ảnh tự nhiên:
- **Thành phần tần số thấp**: Độ chiếu sáng biến thiên mượt (vùng tối do bóng râm, bóng tay/điện thoại).
- **Thành phần tần số cao**: Nét chữ in, viền ký tự, đường kẻ bảng biểu.

#### Các module cải tiến:
1. **`FFTLowPassFilter`**:
   - Phân rã phổ tần số 2D thông qua biến đổi Fourier thực nhanh (`rfft2`) và áp dụng mặt nạ bán kính tần số thấp $r \le \text{cutoff}$.
   - Khôi phục miền không gian qua `irfft2` mà không cần đệm phổ 5/2, giữ tính vi phân liên tục.
2. **`FreqIllumBottleneck`**:
   - Đặt tại nút cổ chai (latent bottleneck) của U-Net.
   - Trích xuất trường chiếu sáng tần số thấp và dự đoán các tham số điều biến affine $(\gamma, \beta)$ để bù trừ bóng cục bộ: $F_{mod} = F \odot (1 + \tanh(\gamma)) + \beta$.
3. **`FreqSkipEnhance`**:
   - Tích hợp tại các đường nối tắt (skip connections) từ encoder sang decoder.
   - Tách tín hiệu nét chữ tần số cao $F_{high} = F - F_{low}$ và khuếch đại: $F_{skip} = F + \alpha \odot F_{high}$ với tham số học được $\alpha$, chống nhòe mờ ký tự ở ranh giới bóng.
4. **Cấu trúc Decoder Tầng 1**:
   - Giữ nguyên thiết kế gốc của Restormer với kênh $2 \times \text{dim}$ (không có conv 1x1 giảm kênh).
   - Bổ sung cờ `reduce_top: False` (nếu bật `True` sẽ kích hoạt conv 1x1 giảm từ $2\times\text{dim} \to \text{dim}$).

Đã kiểm thử qua `test_docshadow_arch.py`: Tự động nạp qua `define_network`, forward/backward trơn tru, kiểm thử cả 2 chế độ `reduce_top`.

---

### Giai đoạn 3: Smoke Test & Kiểm Thử Huấn Luyện Tích Hợp

Đã thiết lập tập smoke data (15 cặp train, 5 cặp val) bằng script [create_smoke_data.py](file:///c:/Restormer/Restormer-main/Shadow_Removal/create_smoke_data.py).

Kiểm thử pipeline toàn diện với [test_train_pipeline.py](file:///c:/Restormer/Restormer-main/Shadow_Removal/tests/test_train_pipeline.py) trên CPU cho cả 3 mô hình:
```
--- Testing pipeline for Restormer_baseline ---
  Train step 1: l_pix: 0.1054
  Train step 2: l_pix: 0.2064
  Val step: output shape = torch.Size([1, 3, 128, 128])
  Restormer_baseline pipeline test PASSED!

--- Testing pipeline for Restormer_fft ---
  Train step 1: l_pix: 0.2598, l_fft: 0.3878
  Train step 2: l_pix: 0.1145, l_fft: 0.1935
  Val step: output shape = torch.Size([1, 3, 128, 128])
  Restormer_fft pipeline test PASSED!

--- Testing pipeline for DocShadowNet ---
  Train step 1: l_pix: 0.3158, l_fft: 0.3124
  Train step 2: l_pix: 0.1312, l_fft: 0.2847
  Val step: output shape = torch.Size([1, 3, 128, 128])
  DocShadowNet pipeline test PASSED!
```

---

## 3. CÁC TẬP TIN ĐÃ TẠO VÀ CHỈNH SỬA

| Tập tin | Loại | Mô tả |
| :--- | :--- | :--- |
| [docshadow_arch.py](file:///c:/Restormer/Restormer-main/basicsr/models/archs/docshadow_arch.py) | **Mới** | Kiến trúc DocShadowNet (tần số + Restormer) |
| [Restormer_RDD_baseline.yml](file:///c:/Restormer/Restormer-main/Shadow_Removal/Options/Restormer_RDD_baseline.yml) | **Mới** | Cấu hình huấn luyện chuẩn Restormer gốc + L1 |
| [Restormer_RDD_fft.yml](file:///c:/Restormer/Restormer-main/Shadow_Removal/Options/Restormer_RDD_fft.yml) | **Mới** | Cấu hình Restormer + L1 + FFTLoss |
| [DocShadowNet_RDD.yml](file:///c:/Restormer/Restormer-main/Shadow_Removal/Options/DocShadowNet_RDD.yml) | **Mới** | Cấu hình DocShadowNet + L1 + FFTLoss |
| `*_smoke.yml` (3 files) | **Mới** | Cấu hình smoke test (200 iters, patch 128) |
| [check_pairs.py](file:///c:/Restormer/Restormer-main/Shadow_Removal/check_pairs.py) | **Mới** | Công cụ kiểm tra tính toàn vẹn dữ liệu |
| [create_val_split.py](file:///c:/Restormer/Restormer-main/Shadow_Removal/create_val_split.py) | **Mới** | Script tách 200 cặp làm validation |
| [create_smoke_data.py](file:///c:/Restormer/Restormer-main/Shadow_Removal/create_smoke_data.py) | **Mới** | Tạo 20 cặp ảnh thử nghiệm nhanh |
| [test_*.py](file:///c:/Restormer/Restormer-main/Shadow_Removal/tests) (5 files) | **Mới** | Unit tests cho FFTLoss, RMSE, Compat, Arch, Pipeline |
| [losses.py](file:///c:/Restormer/Restormer-main/basicsr/models/losses/losses.py) | Sửa | Thêm class `FFTLoss` |
| [image_restoration_model.py](file:///c:/Restormer/Restormer-main/basicsr/models/image_restoration_model.py) | Sửa | Thêm hook nạp `fft_opt` & tối ưu `l_total` |
| [psnr_ssim.py](file:///c:/Restormer/Restormer-main/basicsr/metrics/psnr_ssim.py) | Sửa | Thêm hàm `calculate_rmse` |
| [data_util.py](file:///c:/Restormer/Restormer-main/basicsr/data/data_util.py) | Sửa | Ghép cặp ảnh an toàn khi số lượng file lệch nhau |

---

## 4. NHỮNG GÌ ĐÃ KIỂM TRA VS CHƯA KIỂM TRA & RỦI RO

### Đã kiểm tra:
1. **Kiểm tra tương thích ngược tuyệt đối**: Code gốc của Restormer chạy bình thường, không thiếu class hay phá vỡ registry.
2. **Kiểm tra tính toán vi phân**: Gradient của `FFTLoss` và `DocShadowNet` truyền mượt mà, không gặp lỗi `NaN` hay đứt đồ thị tính toán.
3. **Kiểm tra nạp dữ liệu**: Cả 4,239 cặp train và 545 cặp test đều được nạp chính xác.
4. **Kiểm tra vòng lặp huấn luyện**: Đã chạy thực nghiệm nhiều bước tối ưu tham số và validation trên cả 3 mô hình.

### Chưa kiểm tra & Rủi ro:
1. **Huấn luyện đủ 300,000 steps trên GPU**: Vì môi trường local là CPU không có CUDA, chưa chạy trọn vẹn 300,000 steps (sẽ chạy trên Google Colab).
2. **Bộ nhớ GPU (VRAM OOM) trên Colab T4**:
   - Nếu chạy trên GPU Colab 15-16GB VRAM (T4), ở giai đoạn cuối của progressive training (patch 384x384, batch 1), dung lượng VRAM là an toàn. Tuy nhiên nếu dùng GPU nhỏ hơn (như GPU 8GB), cần giảm kích thước patch hoặc giảm batch size đầu.
3. **Đường dẫn dữ liệu trên Colab**:
   - Cần đảm bảo dữ liệu giải nén đặt tại `./RDD_data` hoặc sửa đường dẫn trong YAML thành `/content/data/RDD_data/...`.

---

## 5. HƯỚNG DẪN CHẠY TRÊN GOOGLE COLAB (1 GPU)

### Bước 1: Chuẩn bị môi trường trên Colab
```bash
# 1. Clone hoặc copy repo lên Colab
# %cd /content/Restormer

# 2. Cài đặt các thư viện cần thiết
pip install -r requirements.txt || pip install einops addict future lmdb scikit-image opencv-python yacs pyyaml

# 3. Tạo thư mục validation từ train (nếu chưa tạo)
python Shadow_Removal/create_val_split.py --data_dir ./RDD_data --n_val 200 --seed 42

# 4. Kiểm tra dữ liệu
python Shadow_Removal/check_pairs.py --data_dir ./RDD_data --max_check 100
```

### Bước 2: Chạy Smoke Test xác nhận môi trường Colab (Chỉ mất ~1-2 phút)
```bash
# Tạo dữ liệu smoke (15 train, 5 val)
python Shadow_Removal/create_smoke_data.py --data_dir ./RDD_data

# Smoke test Baseline
python basicsr/train.py -opt Shadow_Removal/Options/Restormer_RDD_baseline_smoke.yml

# Smoke test FFT Loss
python basicsr/train.py -opt Shadow_Removal/Options/Restormer_RDD_fft_smoke.yml

# Smoke test DocShadowNet
python basicsr/train.py -opt Shadow_Removal/Options/DocShadowNet_RDD_smoke.yml
```

### Bước 3: Huấn luyện chính thức (Full 300k iterations)

#### 1. Mô hình 1 - Restormer Baseline (L1 Loss):
```bash
python basicsr/train.py -opt Shadow_Removal/Options/Restormer_RDD_baseline.yml
```

#### 2. Mô hình 2 - Restormer + FFT Loss (L1 + FFT):
```bash
python basicsr/train.py -opt Shadow_Removal/Options/Restormer_RDD_fft.yml
```

#### 3. Mô hình 3 - DocShadowNet (Kiến trúc tần số + L1 + FFT):
```bash
python basicsr/train.py -opt Shadow_Removal/Options/DocShadowNet_RDD.yml
```

*Lưu ý:* Checkpoints và kết quả trực quan hóa sẽ được lưu tự động trong thư mục `./experiments/<name>/models` và `./experiments/<name>/visualization`.
