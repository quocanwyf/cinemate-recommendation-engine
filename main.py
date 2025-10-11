from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from surprise import dump
import json

# Load model
try:
    print("Loading recommendation model...")
    _predictions, model = dump.load('models/svd_production_model')
    
    with open('models/model_info.json', 'r') as f:
        model_info = json.load(f)
    
    print("✅ Model loaded successfully!")
except Exception as e:
    model = None
    model_info = {}
    print(f"❌ CRITICAL: {e}")

try:
    cosine_sim_matrix = np.load('models/cosine_similarity_matrix.npy')
    movies_df = pd.read_pickle('models/movies_df_for_similarity.pkl')
    # Tạo một series để tra cứu index từ movieId
    indices = pd.Series(movies_df.index, index=movies_df['id']).drop_duplicates()
except:
    cosine_sim_matrix = None
    movies_df = None

app = FastAPI(title="CineMate Recommendation API")

class RecommendationRequest(BaseModel):
    user_id: str        # UUID from DB
    movie_ids: list[int] # List of integer IDs

@app.post("/recommend/svd")
async def get_svd_recommendations(request: RecommendationRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        predictions = []
        
        for movie_id in request.movie_ids:
            # ✅ userId = string, movieId = int (khớp với training data)
            pred = model.predict(
                uid=request.user_id,  # String UUID
                iid=movie_id          # Integer movie ID
            )
            
            predictions.append({
                'movieId': movie_id,
                'score': round(pred.est, 4)
            })
        
        # Sort by score descending
        predictions.sort(key=lambda x: x['score'], reverse=True)
        
        return {"data": predictions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/content-based/{movie_id}")
async def get_content_based_recommendations(movie_id: int, top_n: int = 10):
    if cosine_sim_matrix is None or movies_df is None:
        raise HTTPException(status_code=503, detail="Content-Based model is not available.")
    
    try:
        # Lấy index của phim từ movieId
        idx = indices[movie_id]
        
        # Lấy điểm tương đồng của phim đó với tất cả các phim khác
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        
        # Sắp xếp các phim dựa trên điểm tương đồng
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Lấy 10 phim giống nhất (bỏ qua phim đầu tiên vì là chính nó)
        sim_scores = sim_scores[1:top_n+1]
        
        # Lấy ID của các phim đó
        movie_indices = [i[0] for i in sim_scores]
        recommended_movie_ids = movies_df['id'].iloc[movie_indices].tolist()
        
        return {"data": recommended_movie_ids}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found in model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "CineMate API is running!"}

@app.get("/model/info")
async def get_model_info():
    return model_info if model_info else {"error": "No info available"}