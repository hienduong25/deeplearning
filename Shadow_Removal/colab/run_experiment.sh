#!/usr/bin/env bash
# ==============================================================================
# Script chạy huấn luyện và tự động Resume khi Colab bị ngắt kết nối
# Sử dụng:
#   bash Shadow_Removal/colab/run_experiment.sh 1    # Chạy cấu hình 1 (Baseline)
#   bash Shadow_Removal/colab/run_experiment.sh 5    # Chạy cấu hình 5 (DocShadowNetV2 Full)
# ==============================================================================

set -e

EXP_NUM="$1"

if [ -z "$EXP_NUM" ]; then
    echo "[LỖI] Vui lòng nhập số thứ tự thí nghiệm (1 đến 5)!"
    echo "  1: 1_Restormer_baseline"
    echo "  2: 2_Restormer_fft"
    echo "  3: 3_DocShadowNetV2_fp_freqavg"
    echo "  4: 4_DocShadowNetV2_hf"
    echo "  5: 5_DocShadowNetV2_full"
    exit 1
fi

case "$EXP_NUM" in
    1)
        OPT_FILE="Shadow_Removal/Options/1_Restormer_baseline.yml"
        EXP_NAME="1_Restormer_baseline"
        ;;
    2)
        OPT_FILE="Shadow_Removal/Options/2_Restormer_fft.yml"
        EXP_NAME="2_Restormer_fft"
        ;;
    3)
        OPT_FILE="Shadow_Removal/Options/3_DocShadowNetV2_fp_freqavg.yml"
        EXP_NAME="3_DocShadowNetV2_fp_freqavg"
        ;;
    4)
        OPT_FILE="Shadow_Removal/Options/4_DocShadowNetV2_hf.yml"
        EXP_NAME="4_DocShadowNetV2_hf"
        ;;
    5)
        OPT_FILE="Shadow_Removal/Options/5_DocShadowNetV2_full.yml"
        EXP_NAME="5_DocShadowNetV2_full"
        ;;
    *)
        echo "[LỖI] Số thí nghiệm không hợp lệ: $EXP_NUM (chỉ chọn từ 1 đến 5)"
        exit 1
        ;;
esac

echo "=================================================================="
echo "THỰC THI THÍ NGHIỆM $EXP_NUM: $EXP_NAME"
echo "=================================================================="

# 1. Tạo liên kết lưu trữ experiments vào Google Drive (đảm bảo không mất checkpoint khi ngắt kết nối)
DRIVE_EXP_DIR="/content/drive/MyDrive/shadow/experiments"
mkdir -p "$DRIVE_EXP_DIR"

if [ ! -L "experiments" ]; then
    if [ -d "experiments" ]; then
        echo "Di chuyển thư mục experiments hiện tại sang Google Drive..."
        cp -r experiments/* "$DRIVE_EXP_DIR/" 2>/dev/null || true
        rm -rf experiments
    fi
    echo "Tạo symlink: experiments -> $DRIVE_EXP_DIR"
    ln -s "$DRIVE_EXP_DIR" experiments
fi

# 2. Kiểm tra checkpoint state đã có để tự động Resume
STATES_DIR="experiments/$EXP_NAME/training_states"
LATEST_STATE=""

if [ -d "$STATES_DIR" ]; then
    # Tìm state có số iteration cao nhất
    LATEST_STATE=$(ls -v "$STATES_DIR"/*.state 2>/dev/null | tail -n 1 || true)
fi

RUN_OPT="$OPT_FILE"

if [ -n "$LATEST_STATE" ] && [ -f "$LATEST_STATE" ]; then
    echo ">> PHÁT HIỆN CHECKPOINT CŨ: $LATEST_STATE"
    echo ">> Đang sinh cấu hình Resume tự động..."
    RESUME_YML="Shadow_Removal/Options/temp_resume_${EXP_NAME}.yml"
    python Shadow_Removal/tools/make_resume_yml.py --opt "$OPT_FILE" --state "$LATEST_STATE" --out "$RESUME_YML"
    RUN_OPT="$RESUME_YML"
else
    echo ">> Chưa có checkpoint cũ. Khởi động huấn luyện MỚI TỪ ĐẦU (scratch)."
fi

# 3. Chạy huấn luyện
echo "Khởi chạy lệnh: python basicsr/train.py -opt $RUN_OPT"
python basicsr/train.py -opt "$RUN_OPT"

# 4. Tự động tỉa bớt checkpoint cũ sau khi kết thúc để tránh tràn bộ nhớ Drive
echo "Tỉa bớt checkpoint cũ (giữ lại 2 bản mới nhất)..."
python Shadow_Removal/tools/prune_checkpoints.py --exp "$EXP_NAME" --keep 2 || true

echo "THÍ NGHIỆM $EXP_NAME HOÀN THÀNH THÀNH CÔNG!"
