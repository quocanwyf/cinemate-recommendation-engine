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

@app.get("/")
async def root():
    return {"message": "CineMate API is running!"}

@app.get("/model/info")
async def get_model_info():
    return model_info if model_info else {"error": "No info available"}