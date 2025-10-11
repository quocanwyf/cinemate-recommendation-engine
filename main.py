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
    print("📦 Loading SVD recommendation model...")
    
    if os.path.exists('models/svd_production_model'):
        _, svd_model = dump.load('models/svd_production_model')
        print("✅ SVD Model loaded successfully!")
    else:
        print("⚠️  SVD model file not found")
    
    if os.path.exists('models/model_info.json'):
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)
        print("✅ Model info loaded successfully!")
    else:
        print("⚠️  Model info file not found")
        
except Exception as e:
    svd_model = None
    model_info = {}
    print(f"❌ SVD Model loading failed: {e}")

# Load Content-Based model
try:
    print("📦 Loading Content-Based model...")
    
    cosine_matrix_path = 'models/cosine_similarity_matrix.npy'
    movies_df_path = 'models/movies_df_for_similarity.pkl'
    
    if os.path.exists(cosine_matrix_path) and os.path.exists(movies_df_path):
        cosine_sim_matrix = np.load(cosine_matrix_path)
        movies_df = pd.read_pickle(movies_df_path)
        
        # ✅ Tạo indices mapping
        indices = pd.Series(movies_df.index, index=movies_df['id']).drop_duplicates()
        
        print("✅ Content-Based model loaded successfully!")
        print(f"   - Movies in model: {len(movies_df)}")
        print(f"   - Similarity matrix shape: {cosine_sim_matrix.shape}")
    else:
        print("⚠️  Content-based model files not found")
        print("💡 Run notebook locally to generate these files")
        
except Exception as e:
    cosine_sim_matrix = None
    movies_df = None
    indices = None  # ✅ QUAN TRỌNG: Set indices = None
    print(f"⚠️  Content-Based model loading failed: {e}")

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
    # ✅ Check all required variables
    if cosine_sim_matrix is None or movies_df is None or indices is None:
        raise HTTPException(
            status_code=503, 
            detail="Content-Based model not available. Run notebook to generate model files."
        )
    
    try:
        # ✅ Check if movie exists in model
        if movie_id not in indices:
            available_sample = list(indices.index[:10])
            raise HTTPException(
                status_code=404, 
                detail=f"Movie ID {movie_id} not found. Available sample: {available_sample}"
            )
        
        # Get movie index
        idx = indices[movie_id]
        
        # Calculate similarity scores
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar movies (skip first one - itself)
        sim_scores = sim_scores[1:top_n+1]
        
        # Get movie IDs
        movie_indices = [i[0] for i in sim_scores]
        recommended_movie_ids = movies_df['id'].iloc[movie_indices].tolist()
        
        return {
            "data": recommended_movie_ids,
            "source_movie_id": movie_id,
            "total_recommendations": len(recommended_movie_ids)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content-based error: {str(e)}")

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