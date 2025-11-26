# fastapi_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import joblib
import numpy as np

app = FastAPI(title="Diabetes Risk Local API")

# Path where you may keep your .pkl model (optional)
LOCAL_MODEL_PATH = "./model/diabetes-risk-pred-kaisar-v1.1.pkl"  # adjust if needed
USE_LOCAL_MODEL = os.path.exists(LOCAL_MODEL_PATH)

model = None
if USE_LOCAL_MODEL:
    try:
        model = joblib.load(LOCAL_MODEL_PATH)
        print("Loaded local model:", LOCAL_MODEL_PATH)
    except Exception as e:
        print("Failed to load local model:", e)
        model = None
        USE_LOCAL_MODEL = False

class PredictRequest(BaseModel):
    a1c: List[float]
    glucose_pp: List[float]
    glucose_fast: List[float]
    family_history: List[str]  # "Yes"/"No" or 1/0
    age: List[int]
    activity: List[int]
    bmi: List[float]
    systolic_bp: List[int]

@app.post("/predict")
def predict(req: PredictRequest):
    # convert to DataFrame-like structure or numpy array
    n = len(req.a1c)
    if not (len(req.glucose_pp) == n == len(req.glucose_fast) == len(req.family_history) == len(req.age) == len(req.activity) == len(req.bmi) == len(req.systolic_bp)):
        raise HTTPException(status_code=400, detail="All input lists must have same length")

    # Build feature matrix in order expected by model
    X = []
    for i in range(n):
        fh = req.family_history[i]
        fhv = 1 if str(fh).lower() in ("yes","y","1","true","t") else 0
        X.append([
            req.a1c[i],
            req.glucose_pp[i],
            req.glucose_fast[i],
            fhv,
            req.age[i],
            req.activity[i],
            req.bmi[i],
            req.systolic_bp[i]
        ])
    X = np.array(X)

    # If local model available, use it
    if USE_LOCAL_MODEL and model is not None:
        try:
            probs = model.predict_proba(X)[:,1].tolist()
            preds = ["Yes" if p >= 0.5 else "No" for p in probs]
            return {"preds": preds, "probs": probs}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Local model error: {e}")

    # Otherwise, instruct user to use HF (or you can integrate HF call here)
    raise HTTPException(status_code=501, detail="Local model not found. Please deploy a model or use Hugging Face option in the Streamlit app.")
