from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from surprise import dump
import json
import numpy as np  # ✅ THÊM IMPORT
import pandas as pd  # ✅ THÊM IMPORT
import os

# ✅ Initialize all variables với default values
svd_model = None
model_info = {}
cosine_sim_matrix = None
movies_df = None
indices = None

# Load SVD model
try:
    # Tải SVD model
    _, svd_model = dump.load('models/svd_production_model')
    print("SVD model loaded.")
    
    # Tải các "nguyên liệu" cho Content-Based
    tfidf_matrix = scipy.sparse.load_npz('models/tfidf_matrix.npz')
    movies_df = pd.read_pickle('models/movies_df_for_tfidf.pkl')
    indices_map = pd.Series(movies_df.index, index=movies_df['id']).drop_duplicates()
    print("Content-Based assets loaded.")

except Exception as e:
    print(f"CRITICAL: Failed to load models on startup: {e}")

app = FastAPI(title="CineMate Recommendation API")

class RecommendationRequest(BaseModel):
    user_id: str
    movie_ids: list[int]

@app.post("/recommend/svd")
async def get_svd_recommendations(request: RecommendationRequest):
    # ✅ Fix variable name: svd_model thay vì model
    if not svd_model:
        raise HTTPException(status_code=503, detail="SVD Model not available")
    
    try:
        predictions = []
        
        for movie_id in request.movie_ids:
            # ✅ Ensure string types for Surprise
            pred = svd_model.predict(
                uid=str(request.user_id),  # ✅ Convert to string
                iid=str(movie_id)          # ✅ Convert to string
            )
            
            predictions.append({
                'movieId': movie_id,
                'score': round(pred.est, 4)
            })
        
        predictions.sort(key=lambda x: x['score'], reverse=True)
        return {"data": predictions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SVD prediction error: {str(e)}")

@app.get("/recommend/content-based/{movie_id}")
async def get_content_based_recommendations(movie_id: int, top_n: int = 10):
    if tfidf_matrix is None or movies_df is None:
        raise HTTPException(status_code=503, detail="Content-Based model is not available.")
    
    try:
        idx = indices_map[movie_id]
        
        # TÍNH TOÁN THEO YÊU CẦU:
        # Chỉ tính cosine similarity cho 1 phim này với tất cả các phim khác
        cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        # Lấy index của các phim tương tự
        sim_scores = list(enumerate(cosine_similarities))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        
        recommended_movie_ids = movies_df['id'].iloc[movie_indices].tolist()
        return {"data": recommended_movie_ids}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found in model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
async def root():
    return {
        "message": "🎬 CineMate Recommendation API",
        "version": "1.0.0",
        "available_models": {
            "svd": svd_model is not None,
            "content_based": cosine_sim_matrix is not None
        },
        "endpoints": [
            "POST /recommend/svd",
            "GET /recommend/content-based/{movie_id}",
            "GET /model/info",
            "GET /health"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models": {
            "svd": "loaded" if svd_model else "not_loaded",
            "content_based": "loaded" if cosine_sim_matrix else "not_loaded"
        },
        "files_check": {
            "svd_model": os.path.exists('models/svd_production_model'),
            "model_info": os.path.exists('models/model_info.json'),
            "cosine_matrix": os.path.exists('models/cosine_similarity_matrix.npy'),
            "movies_df": os.path.exists('models/movies_df_for_similarity.pkl')
        }
    }

@app.get("/model/info")
async def get_model_info():
    info = {}
    
    if model_info:
        info["svd"] = model_info
    
    if cosine_sim_matrix is not None and movies_df is not None:
        info["content_based"] = {
            "total_movies": len(movies_df),
            "matrix_shape": list(cosine_sim_matrix.shape),
            "sample_movie_ids": list(indices.index[:10]) if indices is not None else []
        }
    
    return info if info else {"error": "No model info available"}