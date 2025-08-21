from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model at startup - FIXED: Load from correct file
try:
    model_data = joblib.load("model/model_with_threshold.joblib")  # Changed this line
    
    # Check if it's the new format (dict) or old format (model only)
    if isinstance(model_data, dict):
        model = model_data['model']
        optimal_threshold = model_data.get('optimal_threshold', 0.5)
        feature_names = model_data.get('feature_names', None)
        logger.info(f"Loaded model with optimal threshold: {optimal_threshold}")
    else:
        # Old format - just the model
        model = model_data
        optimal_threshold = 0.5
        feature_names = None
        logger.info("Loaded model with default threshold: 0.5")
        
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    optimal_threshold = 0.5
    feature_names = None

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    probabilities: List[float]
    threshold_used: float
    prediction_label: str  # Added for clarity

@app.post("/predict/", response_model=PredictionResponse)
async def get_prediction(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Validate input
        if len(request.features) != 73:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected 73 features, got {len(request.features)}"
            )
        
        # Reshape features for prediction
        reshaped = np.array(request.features).reshape(1, -1)
        
        # Get prediction probabilities
        probabilities = model.predict_proba(reshaped)[0]
        
        # Use optimal threshold
        prediction = int(probabilities[1] >= optimal_threshold)
        confidence = float(max(probabilities))
        
        # Add human readable label
        prediction_label = "Stress" if prediction == 1 else "No Stress"
        
        logger.info(f"Prediction: {prediction_label}, Confidence: {confidence:.3f}, Threshold: {optimal_threshold}")
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities.tolist(),
            threshold_used=optimal_threshold,
            prediction_label=prediction_label
        )
        
    except ValueError as e:
        logger.error(f"Value error in prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold": optimal_threshold,
        "feature_count_expected": 71
    }

@app.get("/model/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "threshold": optimal_threshold,
        "feature_names_available": feature_names is not None,
        "feature_count": len(feature_names) if feature_names else "unknown"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)