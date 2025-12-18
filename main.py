import os
import pickle
import json
import pandas as pd
import numpy as np
import scipy.sparse
import time
import re
from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional

# --- CẤU HÌNH & BIẾN TOÀN CỤC ---
MODELS_PATH = 'models'
svd_model = None
tfidf_matrix = None
tfidf_vectorizer = None  # File thứ 4: Dùng để transform text mới nếu cần
movies_df = None
indices_map = None

app = FastAPI(
    title="CineMate AI Recommendation Engine",
    description="API hệ thống gợi ý phim Hybrid (SVD + Content-Based)",
    version="2.1.0"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. STARTUP: LOAD ĐỦ 4 TÀI NGUYÊN ---
@app.on_event("startup")
def startup_event():
    global svd_model, tfidf_matrix, tfidf_vectorizer, movies_df, indices_map
    print("="*60)
    print("🚀 CINEMATE AI SERVER STARTING...")
    try:
        # Load SVD
        with open(f'{MODELS_PATH}/svd_model_v1.pkl', 'rb') as f:
            svd_model = pickle.load(f)
        
        # Load Matrix & Vectorizer
        tfidf_matrix = scipy.sparse.load_npz(f'{MODELS_PATH}/tfidf_matrix.npz')
        with open(f'{MODELS_PATH}/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
            
        # Load Movie Map & Build Index
        movies_df = pd.read_pickle(f'{MODELS_PATH}/movie_map.pkl')
        # Đảm bảo index của Series là movieId, value là số thứ tự dòng trong matrix
        indices_map = pd.Series(movies_df.index, index=movies_df['id'])
        
        print(f"✅ Loaded 4/4 assets. Matrix shape: {tfidf_matrix.shape}")
        print("="*60)
    except FileNotFoundError as e:
        print(f"❌ CRITICAL ERROR: Missing file in '{MODELS_PATH}' folder: {e}")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR DURING STARTUP: {e}")

# --- 2. VALIDATION SCHEMA ---
class SvdBatchRequest(BaseModel):
    user_id: str = Field(..., description="UUID của người dùng")
    movie_ids: List[int] = Field(..., min_items=1, max_items=500, description="List 1-500 movieIds")

    @validator('user_id')
    def check_uuid_format(cls, v):
        uuid_regex = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
        if not uuid_regex.match(v):
            raise ValueError('user_id must be a valid UUID string (36 chars)')
        return v

# --- 3. API ENDPOINTS ---

@app.get("/", tags=["Health"])
async def health_check():
    # Kiểm tra xem các biến đã được gán giá trị và không phải None
    files = [
        svd_model is not None,
        tfidf_matrix is not None,
        tfidf_vectorizer is not None,
        movies_df is not None
    ]
    
    count_loaded = sum(files)
    
    return {
        "status": "online" if count_loaded == 4 else "error",
        "assets_loaded": f"{count_loaded}/4",
        "details": {
            "svd": "OK" if files[0] else "Missing",
            "matrix": "OK" if files[1] else "Missing",
            "vectorizer": "OK" if files[2] else "Missing",
            "movie_map": "OK" if files[3] else "Missing"
        }
    }



@app.post("/recommend/svd/batch", tags=["Collaborative"])
async def predict_batch(request: SvdBatchRequest):
    """Dự đoán điểm cho một nhóm phim cụ thể (Dùng để Ranking ở Backend)"""
    if not svd_model:
        raise HTTPException(status_code=503, detail="SVD engine is offline")
    
    try:
        results = []
        for m_id in request.movie_ids:
            # uid: str (UUID), iid: int (movieId)
            pred = svd_model.predict(uid=request.user_id, iid=int(m_id))
            results.append({
                "movieId": m_id,
                "score": round(pred.est, 4)
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return {"data": results}
    except Exception as e:
        print(f"Error in batch: {e}")
        raise HTTPException(status_code=500, detail="Error during batch prediction")

@app.get("/recommend/content-based/{movie_id}", tags=["Content"])
async def get_similar_movies(
    movie_id: int = Path(..., ge=1),
    top_n: int = Query(10, ge=1, le=50)
):
    """Tìm phim tương tự dựa trên nội dung (Cosine Similarity)"""
    if tfidf_matrix is None or indices_map is None:
        raise HTTPException(status_code=503, detail="Content engine offline")
    
    try:
        if movie_id not in indices_map:
            raise HTTPException(status_code=404, detail=f"Movie ID {movie_id} not found in AI assets")
            
        idx = indices_map[movie_id]
        # Tính toán vector similarity
        cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        # Lấy top N (bỏ qua phần tử đầu tiên vì là chính nó)
        related_indices = cosine_sim.argsort()[-(top_n+1):-1][::-1]
        recommended_ids = movies_df['id'].iloc[related_indices].astype(int).tolist()
        
        return {"data": recommended_ids}
    except Exception as e:
        print(f"Error in Content-Based: {e}")
        raise HTTPException(status_code=500, detail="AI Similarity calculation failed")