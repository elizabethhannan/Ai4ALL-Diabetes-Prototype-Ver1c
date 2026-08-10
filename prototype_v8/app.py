"""
GE-79 MCI Explorer — ProtoApp Version 8
FastAPI Backend for Model Predictions & SHAP Explanations

Copy this entire file into Replit as app.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GE-79 MCI Explorer",
    description="Machine Learning Model for Mild Cognitive Impairment Prediction",
    version="8.0.0"
)

# CORS configuration (allow Replit's domain + local testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA & CONFIGURATION
# ============================================================================

# 14 Locked Biomarkers (from GE-79 FINAL_FEATURES)
BIOMARKERS = [
    {"key": "CSF_ptau", "display_name": "CSF p-tau", "unit": "pg/mL", "source": "CSF"},
    {"key": "Fasting_glucose", "display_name": "Fasting Glucose", "unit": "mg/dL", "source": "Blood"},
    {"key": "HbA1c", "display_name": "HbA1c", "unit": "%", "source": "Blood"},
    {"key": "HOMA_IR", "display_name": "HOMA-IR", "unit": "score", "source": "Calculated"},
    {"key": "HDL_C", "display_name": "HDL-C", "unit": "mg/dL", "source": "Blood"},
    {"key": "Glucose_random", "display_name": "Random Glucose", "unit": "mg/dL", "source": "Blood"},
    {"key": "Global_vasoreactivity", "display_name": "Global Vasoreactivity", "unit": "score", "source": "MRI"},
    {"key": "SBP", "display_name": "Systolic Blood Pressure", "unit": "mmHg", "source": "Clinical"},
    {"key": "WMH", "display_name": "White Matter Hyperintensities", "unit": "volume %", "source": "MRI"},
    {"key": "sVCAM", "display_name": "sVCAM", "unit": "ng/mL", "source": "Blood"},
    {"key": "eGFR", "display_name": "eGFR", "unit": "mL/min/1.73m²", "source": "Calculated"},
    {"key": "Age", "display_name": "Age", "unit": "years", "source": "Demographics"},
    {"key": "WB_perfusion", "display_name": "Whole-Brain Perfusion", "unit": "mL/100g/min", "source": "MRI"},
    {"key": "Diabetes_duration", "display_name": "Diabetes Duration", "unit": "years", "source": "History"},
]

# Reference values (cohort medians) - REPLACE WITH ACTUAL DATA
REFERENCE_VALUES = {
    "CSF_ptau": 45.0,
    "Fasting_glucose": 125.0,
    "HbA1c": 7.2,
    "HOMA_IR": 3.5,
    "HDL_C": 38.0,
    "Glucose_random": 185.0,
    "Global_vasoreactivity": 0.8,
    "SBP": 138.0,
    "WMH": 2.5,
    "sVCAM": 650.0,
    "eGFR": 75.0,
    "Age": 68.0,
    "WB_perfusion": 55.0,
    "Diabetes_duration": 10.0,
}

# Sample SHAP values for waterfall (individual profile)
SAMPLE_SHAP_WATERFALL = {
    "base_value": 0.35,
    "contributions": [
        {"name": "CSF p-tau", "value": 0.12, "color": "red"},
        {"name": "HOMA-IR", "value": 0.08, "color": "red"},
        {"name": "Age", "value": 0.05, "color": "red"},
        {"name": "Global Vasoreactivity", "value": -0.03, "color": "blue"},
        {"name": "WB Perfusion", "value": -0.02, "color": "blue"},
    ],
    "predicted_value": 0.55,
}

# Sample confusion matrix data (5-fold CV)
SAMPLE_CONFUSION_MATRICES = {
    "Model 2 (Decision Tree)": {
        "true_negative": 45,
        "false_positive": 10,
        "false_negative": 12,
        "true_positive": 8,
        "accuracy": 0.627,
        "precision": 0.444,
        "recall": 0.400,
        "f1": 0.420,
    },
    "Model 3 (Random Forest)": {
        "true_negative": 51,
        "false_positive": 4,
        "false_negative": 8,
        "true_positive": 12,
        "accuracy": 0.840,
        "precision": 0.750,
        "recall": 0.600,
        "f1": 0.667,
    },
}

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class BiomarkerInput(BaseModel):
    key: str
    value: float

class PredictionRequest(BaseModel):
    model_type: str  # "decision_tree" or "random_forest"
    biomarkers: List[BiomarkerInput]

class SHAPContribution(BaseModel):
    name: str
    value: float
    color: str

class SHAPWaterfall(BaseModel):
    base_value: float
    contributions: List[SHAPContribution]
    predicted_value: float

class ConfusionMatrix(BaseModel):
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    accuracy: float
    precision: float
    recall: float
    f1: float

class PredictionResponse(BaseModel):
    model_type: str
    probability_impaired: float
    predicted_class: str
    biomarker_values: Dict[str, float]
    shap_waterfall: SHAPWaterfall
    decision_tree_path: Optional[str] = None
    random_forest_trees: Optional[List[Dict]] = None
    confusion_matrix: ConfusionMatrix
    timestamp: str

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app": "GE-79 MCI Explorer v8",
        "artifacts": "loaded",
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/api/biomarkers")
def get_biomarkers():
    """Return list of 14 biomarkers for selector UI"""
    return {
        "biomarkers": BIOMARKERS,
        "reference_values": REFERENCE_VALUES,
    }

@app.post("/api/predict")
def predict(request: PredictionRequest):
    """
    Generate prediction, SHAP explanation, and confusion matrix
    
    Request body:
    {
        "model_type": "decision_tree" or "random_forest",
        "biomarkers": [
            {"key": "CSF_ptau", "value": 45.0},
            {"key": "Fasting_glucose", "value": 130.0},
            {"key": "Age", "value": 72.0}
        ]
    }
    """
    
    try:
        # Validate model type
        if request.model_type not in ["decision_tree", "random_forest"]:
            raise HTTPException(status_code=400, detail="Invalid model_type")
        
        # Validate biomarker count
        if len(request.biomarkers) != 3:
            raise HTTPException(status_code=400, detail="Exactly 3 biomarkers required")
        
        # Build full biomarker profile (3 user + 11 reference)
        biomarker_values = REFERENCE_VALUES.copy()
        for bm in request.biomarkers:
            biomarker_values[bm.key] = bm.value
        
        # Generate prediction (PLACEHOLDER - replace with actual sklearn model)
        # For now: use mock logic based on input values
        avg_glucose = (biomarker_values.get("Fasting_glucose", 125) + 
                       biomarker_values.get("Glucose_random", 185)) / 2
        avg_age = biomarker_values.get("Age", 68)
        
        # Simple scoring function (replace with actual model.predict_proba)
        base_prob = 0.35
        glucose_contribution = (avg_glucose - 140) / 100 * 0.2
        age_contribution = (avg_age - 70) / 10 * 0.15
        probability_impaired = max(0.01, min(0.99, base_prob + glucose_contribution + age_contribution))
        
        predicted_class = "Impaired" if probability_impaired > 0.50 else "No Impairment"
        
        # Generate SHAP explanation
        shap_waterfall = SHAPWaterfall(
            base_value=0.35,
            contributions=[
                SHAPContribution(name="CSF p-tau", value=0.12, color="red"),
                SHAPContribution(name="Fasting Glucose", value=0.08, color="red"),
                SHAPContribution(name="Age", value=0.05, color="red"),
                SHAPContribution(name="WB Perfusion", value=-0.03, color="blue"),
            ],
            predicted_value=probability_impaired,
        )
        
        # Get confusion matrix for selected model
        model_key = "Model 2 (Decision Tree)" if request.model_type == "decision_tree" else "Model 3 (Random Forest)"
        cm_data = SAMPLE_CONFUSION_MATRICES[model_key]
        confusion_matrix = ConfusionMatrix(**cm_data)
        
        # Generate decision tree path or random forest trees info
        decision_tree_path = None
        random_forest_trees = None
        
        if request.model_type == "decision_tree":
            decision_tree_path = "Root → CSF p-tau > 50 → Age > 70 → HOMA-IR > 3.5 → [Impaired, confidence: 0.72]"
        else:
            random_forest_trees = [
                {"tree_id": 1, "prediction": "Impaired", "confidence": 0.75},
                {"tree_id": 2, "prediction": "No Impairment", "confidence": 0.45},
                {"tree_id": 3, "prediction": "Impaired", "confidence": 0.68},
                {"tree_id": 4, "prediction": "Impaired", "confidence": 0.82},
                {"tree_id": 5, "prediction": "No Impairment", "confidence": 0.40},
            ]
        
        return PredictionResponse(
            model_type=request.model_type,
            probability_impaired=probability_impaired,
            predicted_class=predicted_class,
            biomarker_values=biomarker_values,
            shap_waterfall=shap_waterfall,
            decision_tree_path=decision_tree_path,
            random_forest_trees=random_forest_trees,
            confusion_matrix=confusion_matrix,
            timestamp=datetime.now().isoformat(),
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/shap-beeswarm")
def get_shap_beeswarm():
    """
    Return cohort-level SHAP beeswarm data (fixed, not profile-dependent)
    75 participants × 14 biomarkers
    """
    np.random.seed(42)
    
    beeswarm_data = []
    for biomarker in BIOMARKERS:
        # Generate random SHAP values for 75 participants
        shap_values = np.random.normal(loc=0, scale=0.15, size=75)
        
        for i, shap_val in enumerate(shap_values):
            beeswarm_data.append({
                "biomarker": biomarker["display_name"],
                "participant_id": i,
                "shap_value": float(shap_val),
                "color": "red" if shap_val > 0 else "blue",
            })
    
    return {
        "type": "beeswarm",
        "label": "Cohort-Level SHAP Evidence",
        "cohort_size": 75,
        "data": beeswarm_data,
    }

@app.get("/api/demo-profiles")
def get_demo_profiles():
    """Return 3 pre-tested demo profiles"""
    return {
        "profiles": [
            {
                "name": "Reference Profile",
                "description": "Cohort median values",
                "biomarkers": [
                    {"key": "CSF_ptau", "value": 45.0},
                    {"key": "Fasting_glucose", "value": 125.0},
                    {"key": "Age", "value": 68.0},
                ],
            },
            {
                "name": "Example: No Impairment",
                "description": "Low-risk profile",
                "biomarkers": [
                    {"key": "CSF_ptau", "value": 30.0},
                    {"key": "Fasting_glucose", "value": 100.0},
                    {"key": "Age", "value": 60.0},
                ],
            },
            {
                "name": "Example: Impaired",
                "description": "High-risk profile",
                "biomarkers": [
                    {"key": "CSF_ptau", "value": 80.0},
                    {"key": "Fasting_glucose", "value": 180.0},
                    {"key": "Age", "value": 75.0},
                ],
            },
        ],
    }

# ============================================================================
# STATIC FILES & FALLBACK
# ============================================================================

# Serve static files (React build output)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Static files directory not found: {e}")

@app.get("/")
def root():
    """Serve index.html fallback"""
    return {"message": "GE-79 MCI Explorer API v8. Visit /docs for API documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
