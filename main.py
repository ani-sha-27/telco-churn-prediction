from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

BASE_DIR = Path(__file__).resolve().parent

# Exact column order the scaler and model were trained on
FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "InternetService_Fiber optic", "InternetService_No",
    "Contract_One year", "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]

YesNo = Literal["Yes", "No"]
YesNoPhone = Literal["Yes", "No", "No phone service"]
YesNoInternet = Literal["Yes", "No", "No internet service"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the saved artifacts once, when the server starts
    app.state.model = joblib.load(BASE_DIR / "churn_model.pkl")
    app.state.scaler = joblib.load(BASE_DIR / "scaler.pkl")
    app.state.threshold = float(joblib.load(BASE_DIR / "threshold.pkl"))
    yield


app = FastAPI(title="Customer Churn Prediction API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local development; restrict this when you deploy
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerData(BaseModel):
    """One customer, using the same values as the original Telco CSV."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: YesNoPhone
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoInternet
    OnlineBackup: YesNoInternet
    DeviceProtection: YesNoInternet
    TechSupport: YesNoInternet
    StreamingTV: YesNoInternet
    StreamingMovies: YesNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0, description="Use 0 for a brand-new customer")

    @model_validator(mode="after")
    def check_services_make_sense(self):
        addons = [self.OnlineSecurity, self.OnlineBackup, self.DeviceProtection,
                  self.TechSupport, self.StreamingTV, self.StreamingMovies]
        if self.InternetService == "No" and "Yes" in addons:
            raise ValueError("InternetService is 'No', so add-on services cannot be 'Yes'")
        if self.PhoneService == "No" and self.MultipleLines == "Yes":
            raise ValueError("PhoneService is 'No', so MultipleLines cannot be 'Yes'")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
                "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
                "StreamingMovies": "No", "Contract": "Month-to-month",
                "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.7, "TotalCharges": 151.65,
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    threshold: float
    result: str


def to_features(c: CustomerData) -> pd.DataFrame:
    """Turn the raw customer into the 23 encoded columns the model was trained on."""
    yes = lambda v: int(v == "Yes")  # 'No', 'No phone service', 'No internet service' -> 0
    row = {
        "gender": int(c.gender == "Male"),
        "SeniorCitizen": c.SeniorCitizen,
        "Partner": yes(c.Partner),
        "Dependents": yes(c.Dependents),
        "tenure": c.tenure,
        "PhoneService": yes(c.PhoneService),
        "MultipleLines": yes(c.MultipleLines),
        "OnlineSecurity": yes(c.OnlineSecurity),
        "OnlineBackup": yes(c.OnlineBackup),
        "DeviceProtection": yes(c.DeviceProtection),
        "TechSupport": yes(c.TechSupport),
        "StreamingTV": yes(c.StreamingTV),
        "StreamingMovies": yes(c.StreamingMovies),
        "PaperlessBilling": yes(c.PaperlessBilling),
        "MonthlyCharges": c.MonthlyCharges,
        "TotalCharges": c.TotalCharges,
        "InternetService_Fiber optic": int(c.InternetService == "Fiber optic"),
        "InternetService_No": int(c.InternetService == "No"),
        "Contract_One year": int(c.Contract == "One year"),
        "Contract_Two year": int(c.Contract == "Two year"),
        "PaymentMethod_Credit card (automatic)": int(c.PaymentMethod == "Credit card (automatic)"),
        "PaymentMethod_Electronic check": int(c.PaymentMethod == "Electronic check"),
        "PaymentMethod_Mailed check": int(c.PaymentMethod == "Mailed check"),
    }
    return pd.DataFrame([row], columns=FEATURES)


@app.get("/")
def health():
    return {"status": "ok", "model": "Logistic Regression", "threshold": app.state.threshold}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    X = to_features(customer)
    X_scaled = pd.DataFrame(app.state.scaler.transform(X), columns=FEATURES)  # scale once
    proba = float(app.state.model.predict_proba(X_scaled)[0, 1])
    threshold = app.state.threshold
    pred = int(proba >= threshold)
    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=pred,
        threshold=threshold,
        result="Likely to churn" if pred == 1 else "Likely to stay",
    )
