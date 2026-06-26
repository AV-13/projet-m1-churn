"""Tests de l'API : /health, /predict, validation des entrées.

Ces tests nécessitent que le modèle ait été entraîné (`make train`).
Ils sont ignorés automatiquement si les artefacts sont absents.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from api.service import service  # noqa: E402

EXAMPLE = {
    "gender": "Female", "age": 57, "country": "Canada", "city": "Sydney",
    "customer_segment": "Individual", "tenure_months": 9, "signup_channel": "Mobile",
    "contract_type": "Monthly", "monthly_logins": 7, "weekly_active_days": 5,
    "avg_session_time": 26.8, "features_used": 1, "usage_growth_rate": -0.28,
    "last_login_days_ago": 2, "monthly_fee": 30, "total_revenue": 270,
    "payment_method": "Card", "payment_failures": 1, "discount_applied": "No",
    "price_increase_last_3m": "Yes", "support_tickets": 1, "avg_resolution_time": 25.1,
    "complaint_type": "Billing", "csat_score": 2.0, "escalations": 0,
    "email_open_rate": 0.78, "marketing_click_rate": 0.33, "nps_score": -19,
    "survey_response": "Neutral", "referral_count": 2,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "model_loaded" in r.json()


def test_predict_valid(client):
    if not service.is_loaded:
        pytest.skip("Modèle non entraîné — lancez `make train`.")
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert body["risk_level"] in ("faible", "modéré", "élevé")


def test_predict_invalid_payload(client):
    bad = dict(EXAMPLE)
    bad["csat_score"] = 99  # hors plage (1-5)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422  # validation Pydantic
