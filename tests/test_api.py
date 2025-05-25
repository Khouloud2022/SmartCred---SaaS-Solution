# ml_service/tests/test_api.py

import pytest
import json
from src.api.app import app as flask_app

@pytest.fixture
def app():
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'ok'
    assert data['model_loaded'] is True

def test_successful_prediction(client):
    """Test the predict endpoint with valid data."""
    test_data = {
        "annual_income": 80000,
        "debt_to_income": 12.0,
        "emp_length": 6,
        "homeownership": "MORTGAGE",
        "verified_income": "Source Verified",
        "loan_amount": 15000,
        "interest_rate": 11.5,
        "term": 36,
        "grade": "B",
        "late_payments": 1
    }
    response = client.post('/predict', data=json.dumps(test_data), content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction_label' in data
    assert 'prediction_value' in data
    assert 'probability_of_risk_percent' in data
    assert data['prediction_value'] in [0, 1]

def test_missing_field_prediction(client):
    """Test the predict endpoint with missing data."""
    test_data = {
        "annual_income": 80000,
        # "debt_to_income" is missing
        "emp_length": 6,
        "homeownership": "MORTGAGE",
        "verified_income": "Source Verified",
        "loan_amount": 15000,
        "interest_rate": 11.5,
        "term": 36,
        "grade": "B",
        "late_payments": 1
    }
    response = client.post('/predict', data=json.dumps(test_data), content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert "Input data is missing required feature" in data['error']