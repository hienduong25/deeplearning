#!/usr/bin/env bash
# ==============================================================================
# Script setup môi trường tự động trên Google Colab cho nhóm 3 người
# Sử dụng:
#   bash Shadow_Removal/colab/setup_colab.sh /content/drive/MyDrive/shadow/RDD_data_final.zip
# ==============================================================================

set -e

ZIP_PATH="$1"

if [ -z "$ZIP_PATH" ]; then
    echo "[LỖI] Vui lòng truyền đường dẫn đến tệp zip dữ liệu trong Google Drive!"
    echo "Ví dụ: bash Shadow_Removal/colab/setup_colab.sh /content/drive/MyDrive/shadow/RDD_data_final.zip"
    exit 1
fi

if [ ! -f "$ZIP_PATH" ]; then
    echo "[LỖI] Tệp zip không tồn tại tại: $ZIP_PATH"
    exit 1
fi

echo "=================================================================="
echo "1. KIỂM TRA THÔNG TIN PHẦN CỨNG GPU"
echo "=================================================================="
nvidia-smi

echo ""
echo "=================================================================="
echo "2. SAO CHÉP & GIẢI NÉN DỮ LIỆU"
echo "=================================================================="
mkdir -p /content/data
echo "Đang sao chép từ $ZIP_PATH về /content/..."
cp "$ZIP_PATH" /content/rdd_temp.zip

echo "Đang giải nén vào /content/data/tmp_rdd..."
mkdir -p /content/data/tmp_rdd
unzip -q /content/rdd_temp.zip -d /content/data/tmp_rdd
rm /content/rdd_temp.zip

# Xử lý trường hợp zip chứa lớp thư mục ngoài RDD_data/
mkdir -p /content/data/RDD_data
if [ -d "/content/data/tmp_rdd/RDD_data/train" ]; then
    cp -r /content/data/tmp_rdd/RDD_data/* /content/data/RDD_data/
elif [ -d "/content/data/tmp_rdd/train" ]; then
    cp -r /content/data/tmp_rdd/* /content/data/RDD_data/
else
    echo "[CẢNH BÁO] Cấu trúc thư mục zip khác dự kiến, đang di chuyển trực tiếp..."
    cp -r /content/data/tmp_rdd/* /content/data/RDD_data/
fi
rm -rf /content/data/tmp_rdd

echo "Cấu trúc thư mục dữ liệu tại /content/data/RDD_data:"
ls -lh /content/data/RDD_data

echo ""
echo "=================================================================="
echo "3. CÀI ĐẶT CÁC THƯ VIỆN PHỤ THUỘC"
echo "=================================================================="
pip install -q einops scikit-image opencv-python pyyaml lmdb addict requests tqdm tensorboard

echo ""
echo "=================================================================="
echo "4. BIÊN DỊCH BASICSR (NO CUDA EXTENSIONS)"
echo "=================================================================="
python setup.py develop --no_cuda_ext

echo ""
echo "=================================================================="
echo "5. KIỂM CHỨNG KHỚP CẶP & CHỐNG RÒ RỈ DỮ LIỆU"
echo "=================================================================="
python Shadow_Removal/check_pairs.py --data_dir /content/data/RDD_data
python Shadow_Removal/tests/test_splits_no_leak.py --data_dir /content/data/RDD_data

echo ""
echo "=================================================================="
echo "SETUP HOÀN TẤT VÀ MÔI TRƯỜNG ĐÃ SẴN SÀNG HUẤN LUYỆN!"
echo "=================================================================="
