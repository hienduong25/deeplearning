# HƯỚNG DẪN THỰC THI TRÊN GOOGLE COLAB CHO NHÓM

Tài liệu này tóm tắt quy trình chuẩn hoá từng bước cho từng thành viên khi chạy huấn luyện trên Google Colab GPU.

---

## BƯỚC 1: KẾT NỐI GPU & MOUNT GOOGLE DRIVE
Trong notebook Colab (chọn Runtime -> Change runtime type -> **T4 / V100 / A100 GPU**):

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## BƯỚC 2: CLONE REPOSITORY
Nếu repo là Private, lưu GitHub Token vào mục **Colab Secrets** (biểu tượng chìa khoá bên trái: Name = `GH_TOKEN`), sau đó clone an toàn mà không để lộ token:

```python
from google.colab import userdata
token = userdata.get('GH_TOKEN')
!git clone https://{token}@github.com/username/Restormer-main.git /content/Restormer
%cd /content/Restormer
```
*(Nếu repo là Public: `!git clone <URL> /content/Restormer` và `%cd /content/Restormer`)*

---

## BƯỚC 3: THIẾT LẬP MÔI TRƯỜNG & DỮ LIỆU TỰ ĐỘNG
Chạy một lệnh duy nhất: script sẽ tự động copy zip từ Drive về `/content`, giải nén, cài đặt thư viện, biên dịch `setup.py develop --no_cuda_ext` và kiểm tra chống rò rỉ dữ liệu:

```bash
!bash Shadow_Removal/colab/setup_colab.sh /content/drive/MyDrive/shadow/RDD_data_final.zip
```

---

## BƯỚC 4: THĂM DÒ PHẦN CỨNG & DỰ ĐOÁN THỜI GIAN (PROBE GPU)
Chạy script probe để đo VRAM đỉnh và thời gian thực thi:

```bash
!python Shadow_Removal/tools/probe_gpu.py --opt Shadow_Removal/Options/5_DocShadowNetV2_full.yml --iters 10
```
- Kết quả và gợi ý kích thước batch sẽ được ghi vào tệp `Shadow_Removal/colab/probe_<GPU>.md`.

---

## BƯỚC 5: KHỞI CHẠY THÍ NGHIỆM ĐƯỢC PHÂN CÔNG
Mỗi người chạy đúng số thứ tự của mình (xem chi tiết tại `Shadow_Removal/colab/ASSIGNMENT.md`):

```bash
# Người A (Đợt 1: Exp 1 - Baseline):
!bash Shadow_Removal/colab/run_experiment.sh 1

# Người B (Đợt 1: Exp 2 - Restormer + FFT):
!bash Shadow_Removal/colab/run_experiment.sh 2

# Người C (Đợt 1: Exp 5 - DocShadowNetV2 Full):
!bash Shadow_Removal/colab/run_experiment.sh 5
```

---

## BƯỚC 6: XỬ LÝ KHI BỊ NGẮT KẾT NỐI (AUTO-RESUME)
Nếu phiên Colab bị ngắt giữa chừng, toàn bộ checkpoint và optimizer states đã được lưu an toàn trong Drive (`/content/drive/MyDrive/shadow/experiments/`).
Chỉ cần:
1. Mở lại notebook Colab.
2. Chạy lại Bước 1 (Mount Drive), Bước 2 (cd vào repo), và Bước 3 (setup).
3. Chạy lại đúng lệnh ở Bước 5 (ví dụ: `!bash Shadow_Removal/colab/run_experiment.sh 5`).
   *Hệ thống sẽ tự động tìm checkpoint `.state` mới nhất và tiếp tục huấn luyện ngay lập tức.*

---

## BƯỚC 7: GIẢI PHÓNG DUNG LƯỢNG DRIVE (PRUNE CHECKPOINTS)
Khi cần giải phóng bộ nhớ Drive, giữ lại 2 checkpoint mới nhất và xoá các checkpoint cũ:

```bash
!python Shadow_Removal/tools/prune_checkpoints.py --exp 5_DocShadowNetV2_full --keep 2
```
