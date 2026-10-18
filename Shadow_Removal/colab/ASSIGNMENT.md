# BẢNG PHÂN CÔNG HUẤN LUYỆN COLAB (NHÓM 3 NGƯỜI)

## 1. NGUYÊN TẮC CÔNG BẰNG & NHẤT QUÁN TOÀN NHÓM
1. **Cùng bộ dữ liệu**: Toàn bộ nhóm sử dụng chung tệp `RDD_data_final.zip` chuẩn hoá (Train: 4,171, Val: 200, Test: 545; KHÔNG chứa `back_gt` hay `smoke`).
2. **Cùng thiết lập huấn luyện**: Cả 5 mô hình đều có `manual_seed: 100`, `total_iter: 60000`, `lr: 3e-4`, và lịch trình Progressive Learning giống hệt nhau.
3. **Huấn luyện từ đầu (Scratch)**: Không nạp trọng số pretrained để bảng ablation phản ánh đúng 100% hiệu quả kiến trúc.
4. **Lưu trữ tập trung**: Tất cả checkpoint được lưu trực tiếp vào thư mục Google Drive chung: `/content/drive/MyDrive/shadow/experiments/`.
5. **Đồng nhất số bước (Budget)**: Nếu chạy `probe_gpu.py` thấy 60k quá dài so với thời lượng Colab, nhóm sẽ thống nhất cùng chạy chung một cấu hình `budget_<N>` (ví dụ 30k) cho TẤT CẢ 5 bài toán.

---

## 2. KẾ HOẠCH PHÂN CHIA HUẤN LUYỆN

```
               [ĐỢT 1 - Kiểm tra lỗi & Thiết lập mốc]
   Người A: Exp 1 (Baseline Restormer)
   Người B: Exp 2 (Restormer + FFTLoss)
   Người C: Exp 5 (DocShadowNetV2 Full: FP + FreqAvgUp + HF + FFT)  <-- Rủi ro cao nhất, chạy sớm!

               [ĐỢT 2 - Hoàn thiện Ablation & Đánh giá]
   Người A: Exp 3 (DocShadowNetV2: FP + FreqAvgUp + FFT)
   Người B: Exp 4 (DocShadowNetV2: HF + FFT)
   Người C: Chạy test.py đánh giá 5 mô hình & Tổng hợp bảng kết quả Ablation
```

### Chi tiết phân công từng thành viên

| Thành viên | Đợt 1 (Mô hình) | Đợt 2 (Mô hình / Tác vụ) | Lệnh thực thi chính |
| :---: | :--- | :--- | :--- |
| **Người A** | **Cấu hình 1**: `1_Restormer_baseline`<br>*(Baseline chuẩn Restormer CVPR 2022)* | **Cấu hình 3**: `3_DocShadowNetV2_fp_freqavg`<br>*(Kiểm chứng nhánh tần số thấp)* | Đợt 1: `bash Shadow_Removal/colab/run_experiment.sh 1`<br>Đợt 2: `bash Shadow_Removal/colab/run_experiment.sh 3` |
| **Người B** | **Cấu hình 2**: `2_Restormer_fft`<br>*(Đo lường tác động của hàm mất mát Fourier)* | **Cấu hình 4**: `4_DocShadowNetV2_hf`<br>*(Kiểm chứng nhánh cao tần HighFreqBranch)* | Đợt 1: `bash Shadow_Removal/colab/run_experiment.sh 2`<br>Đợt 2: `bash Shadow_Removal/colab/run_experiment.sh 4` |
| **Người C** | **Cấu hình 5**: `5_DocShadowNetV2_full`<br>*(Mô hình đề xuất hoàn chỉnh)* | **Đánh giá & Tổng hợp**:<br>- Chạy `Shadow_Removal/test.py` cho cả 5 mô hình<br>- Chạy `summarize_results.py` tạo bảng Ablation | Đợt 1: `bash Shadow_Removal/colab/run_experiment.sh 5`<br>Đợt 2: Chạy kiểm thử trên tập Test & sinh bảng Markdown |

---

## 3. CHECKLIST KHI GẶP SỰ CỐ
- **Colab bị ngắt kết nối (Disconnect/Timeout)**:
  Mở lại notebook -> Mount Drive -> Chạy lại đúng lệnh `bash Shadow_Removal/colab/run_experiment.sh <X>`. Script sẽ tự động phát hiện tệp `.state` mới nhất và tiếp tục huấn luyện ngay lập tức.
- **Báo lỗi CUDA Out of Memory (OOM)**:
  Chạy `python Shadow_Removal/tools/probe_gpu.py --opt <file.yml>` để kiểm tra VRAM tối đa. Nếu VRAM < 15GB, chỉ cần hạ nhẹ `mini_batch_sizes: [16, 10, 8, 2]`.
- **Ổ Google Drive gần đầy**:
  Chạy `python Shadow_Removal/tools/prune_checkpoints.py --exp <tên_exp> --keep 2` để giải phóng dung lượng các checkpoint cũ.
