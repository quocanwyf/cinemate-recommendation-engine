#!/bin/bash

# Dừng lại nếu có lỗi
set -e

echo ">>> Creating models directory..."
mkdir -p models

echo ">>> Downloading SVD model..."
# Lấy link chia sẻ của Google Drive và trích xuất ID file
# Ví dụ link: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
# Thay FILE_ID_SVD bằng ID thật của bạn
gdown --id 1cw4trFOJcKElY_VVBuvmQAvKT38vx9UA -O models/svd_production_model

echo ">>> Downloading Cosine Similarity matrix..."
# Thay FILE_ID_COSINE bằng ID thật của bạn
gdown --id 1pegfmlxEaAWft1EKUP8eqXzbRKbJ0I9G -O models/cosine_similarity_matrix.npy

echo ">>> Downloading Movies DataFrame..."
# Thay FILE_ID_DATAFRAME bằng ID thật của bạn
gdown --id 1wwrl_48EJ39HCFtjjMf7Qy3pHl5nKhH5 -O models/movies_df_for_similarity.pkl

echo ">>> Model download complete."