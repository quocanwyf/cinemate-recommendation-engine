#!/bin/bash
set -e

echo ">>> Creating models directory..."
mkdir -p models

echo ">>> Downloading SVD model..."
gdown 1LBJTRTjHKTuMHJLEgI7KCsb6Q0sgK7ni -O models/svd_production_model.pkl

echo ">>> Downloading TF-IDF matrix..."
gdown 18JThN9cGmqUS_HBRZXAEwYx6CxJOj5R9 -O models/tfidf_matrix.npz

echo ">>> Downloading TF-IDF vectorizer..."
gdown 1uawGB5xbvcPBOdaYn1BuuOpWtdrLf90y -O models/tfidf_vectorizer.pkl

echo ">>> Downloading Movies DataFrame for TF-IDF..."
gdown 1rTLhx8XJBY-8kcDen4UwawHed3dWWhL2 -O models/movies_df_for_tfidf.pkl

echo ">>> Model download complete."