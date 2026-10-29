from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "combined_fraud_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. "
        "Run src/save_combined_model.py first."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Fraudulent Product Return Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReturnClaim(BaseModel):
    order_value_inr: float = Field(ge=0)
    days_since_delivery: float = Field(ge=0)
    num_images: int = Field(ge=0)

    image_blur_score: float = Field(ge=0, le=1)
    exif_timestamp_gap_days: float = Field(ge=0)
    image_reuse_flag: int = Field(ge=0, le=1)

    clip_image_text_sim: float = Field(ge=0, le=1)
    clip_image_listing_sim: float = Field(ge=0, le=1)
    vqa_defect_visible: float = Field(ge=0, le=1)
    claim_consistency_score: float = Field(ge=0, le=1)

    cust_total_orders_12m: int = Field(ge=0)
    cust_return_rate_12m: float = Field(ge=0, le=1)
    cust_account_age_days: int = Field(ge=0)
    cust_prev_fraud_flags: int = Field(ge=0)

    refund_requested_amount: float = Field(ge=0)
    delivery_pincode_risk: float = Field(ge=0, le=1)

    category: str
    return_reason_code: str
    complaint_language: str
    payment_mode: str
    complaint_text: str


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(claim: ReturnClaim):
    try:
        input_data = pd.DataFrame([claim.model_dump()])

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]

        fraud_index = list(model.classes_).index(1)
        fraud_probability = float(probabilities[fraud_index])

        return {
            "prediction": prediction,
            "result": (
                "Potentially fraudulent"
                if prediction == 1
                else "Likely genuine"
            ),
            "fraud_probability_percent": round(
                fraud_probability * 100, 2
            ),
            "disclaimer": (
                "Experimental prediction only; "
                "this result is not proof of fraud."
            )
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}"
        )
