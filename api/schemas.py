"""Schémas Pydantic : contrats d'entrée/sortie de l'API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Profil client attendu par l'endpoint /predict."""

    gender: str
    age: int = Field(ge=0, le=120)
    country: str
    city: str
    customer_segment: str
    tenure_months: int = Field(ge=0)
    signup_channel: str
    contract_type: str
    monthly_logins: int = Field(ge=0)
    weekly_active_days: int = Field(ge=0, le=7)
    avg_session_time: float = Field(ge=0)
    features_used: int = Field(ge=0)
    usage_growth_rate: float
    last_login_days_ago: int = Field(ge=0)
    monthly_fee: float = Field(ge=0)
    total_revenue: float = Field(ge=0)
    payment_method: str
    payment_failures: int = Field(ge=0)
    discount_applied: str
    price_increase_last_3m: str
    support_tickets: int = Field(ge=0)
    avg_resolution_time: float = Field(ge=0)
    complaint_type: Optional[str] = None
    csat_score: float = Field(ge=1, le=5)
    escalations: int = Field(ge=0)
    email_open_rate: float = Field(ge=0, le=1)
    marketing_click_rate: float = Field(ge=0, le=1)
    nps_score: int = Field(ge=-100, le=100)
    survey_response: str
    referral_count: int = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "age": 57, "country": "Canada", "city": "Sydney",
                "customer_segment": "Individual", "tenure_months": 9,
                "signup_channel": "Mobile", "contract_type": "Monthly",
                "monthly_logins": 7, "weekly_active_days": 5, "avg_session_time": 26.8,
                "features_used": 1, "usage_growth_rate": -0.28, "last_login_days_ago": 2,
                "monthly_fee": 30, "total_revenue": 270, "payment_method": "Card",
                "payment_failures": 1, "discount_applied": "No",
                "price_increase_last_3m": "Yes", "support_tickets": 1,
                "avg_resolution_time": 25.1, "complaint_type": "Billing",
                "csat_score": 2.0, "escalations": 0, "email_open_rate": 0.78,
                "marketing_click_rate": 0.33, "nps_score": -19,
                "survey_response": "Neutral", "referral_count": 2,
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float = Field(description="Probabilité de churn (0-1)")
    churn_prediction: int = Field(description="Classe prédite (1 = churn) au seuil retenu")
    threshold: float = Field(description="Seuil de décision appliqué")
    risk_level: str = Field(description="Niveau de risque : faible / modéré / élevé")
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None
