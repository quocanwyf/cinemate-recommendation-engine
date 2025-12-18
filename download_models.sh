#!/bin/bash
set -e

echo ">>> Creating models directory..."
mkdir -p models

echo ">>> Downloading SVD model..."
gdown 1hRs0q3X1lIGhuSedHI04V2lLXNa4bfal -O models/svd_model_v1.pkl

echo ">>> Downloading TF-IDF matrix..."
gdown 1iAotmE9Qi6yTcAhoeGI03_BknCbE6BrL -O models/tfidf_matrix.npz

echo ">>> Downloading TF-IDF vectorizer..."
gdown 1YAWCkBomR0MQW9p2pxETSkZ1hfCN_Irf -O models/tfidf_vectorizer.pkl

echo ">>> Downloading Movies DataFrame for TF-IDF..."
gdown 1raAjz3LVu5M6Z2yuO6QBUH3mSWU6IXgs -O models/movie_map.pkl

echo ">>> Model download complete."
