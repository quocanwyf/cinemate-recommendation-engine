import os
import pickle
import json
import pandas as pd
import numpy as np
import scipy.sparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from surprise import dump
from sklearn.metrics.pairwise import cosine_similarity

# --- KHỞI TẠO CÁC BIẾN MODEL ---
svd_model = None
tfidf_matrix = None
movies_df = None
indices_map = None
model_info = {}

# --- TẢI CÁC MODEL VÀ ASSETS KHI KHỞI ĐỘNG ---
try:
    print("--- Loading All Models on Startup ---")
    
    # 1. Tải SVD model từ file .pkl
    print("Loading SVD model from .pkl file...")
    with open('models/svd_production_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    print("SVD model loaded successfully!")
    
    # 2. Tải các "nguyên liệu" cho Content-Based
    print("Loading Content-Based assets...")
    tfidf_matrix = scipy.sparse.load_npz('models/tfidf_matrix.npz')
    movies_df = pd.read_pickle('models/movies_df_for_tfidf.pkl')
    # Tạo bản đồ ánh xạ: key là movieId (dạng int), value là index của nó trong DataFrame
    indices_map = pd.Series(movies_df.index, index=movies_df['id'])
    print("Content-Based assets loaded successfully!")

    # 3. Tải file thông tin model (nếu có)
    model_info_path = 'models/model_info.json'
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        print("Model info loaded.")

except Exception as e:
    print(f"CRITICAL: Failed to load models on startup: {e}")

# --- KHỞI TẠO ỨNG DỤNG FASTAPI ---
app = FastAPI(title="CineMate Recommendation API", version="1.0.0")

# --- ĐỊNH NGHĨA CÁC LỚP REQUEST BODY ---
class SvdRecommendationRequest(BaseModel):
    user_id: str
    movie_ids: list[int]

# --- ĐỊNH NGHĨA CÁC API ENDPOINTS ---

@app.get("/")
async def root():
    return {
        "message": "🎬 CineMate Recommendation API is running!",
        "models_status": {
            "svd": "loaded" if svd_model else "failed_to_load",
            "content_based": "loaded" if tfidf_matrix is not None else "failed_to_load",
        }
    }

@app.post("/recommend/svd")
async def get_svd_recommendations(request: SvdRecommendationRequest):
    if not svd_model:
        raise HTTPException(status_code=503, detail="SVD Model is not available.")
    
    try:
        predictions = []
        for movie_id in request.movie_ids:
            # uid và iid phải là chuỗi để khớp với dữ liệu train
            pred = svd_model.predict(uid=request.user_id, iid=str(movie_id))
            predictions.append({'movieId': int(pred.iid), 'score': round(pred.est, 4)})
        
        predictions.sort(key=lambda x: x['score'], reverse=True)
        return {"data": predictions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SVD prediction error: {str(e)}")

@app.get("/recommend/content-based/{movie_id}")
async def get_content_based_recommendations(movie_id: int, top_n: int = 10):
    if tfidf_matrix is None or movies_df is None:
        raise HTTPException(status_code=503, detail="Content-Based model is not available.")
    
    try:
        # Lấy index của phim từ movieId
        idx = indices_map[movie_id]
        
        # Chỉ tính cosine similarity cho 1 phim này với tất cả các phim khác
        cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        # Lấy index và điểm của các phim tương tự
        sim_scores = list(enumerate(cosine_similarities))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        
        # Lấy ID của các phim đó
        movie_indices = [i[0] for i in sim_scores]
        recommended_movie_ids = movies_df['id'].iloc[movie_indices].tolist()
        
        return {"data": recommended_movie_ids}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found in model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def get_model_info_endpoint():
    if not model_info:
        raise HTTPException(status_code=404, detail="Model info file not found.")
    return model_info