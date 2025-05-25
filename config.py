# ml_service/config.py

from pathlib import Path

# Define the base directory of the ml_service.
BASE_DIR = Path(__file__).resolve().parent

# --- PATHS ---
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "accepted_2007_to_2018Q4.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SRC_DIR = BASE_DIR / "src"

# --- ADD THESE TWO LINES ---
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "lending_club_cleaned.parquet"

# Ensure the necessary directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True) # Also ensure this one exists

# --- ARTIFACT FILE NAMES ---
MODEL_NAME = "credit_risk_model.pkl"


# --- Artifact file names ---
MODEL_NAME = "credit_risk_model.pkl"
SCALER_NAME = "scaler.pkl"
HOMEOWNERSHIP_ENCODER_NAME = "le_homeownership.pkl"
VERIFIED_INCOME_ENCODER_NAME = "le_verified_income.pkl"
GRADE_ENCODER_NAME = "le_grade.pkl"
FEATURES_NAME = "selected_features.pkl"
METRICS_NAME = "model_metrics.json"

# --- Model training parameters ---
# List of features to be used in the model
FEATURES_TO_USE = [
    'annual_income', 'debt_to_income', 'emp_length', 'homeownership',
    'verified_income', 'loan_amount', 'interest_rate', 'term', 'grade',
    'late_payments', 'loan_to_income', 'grade_interest_rate', 'income_debt_ratio'
]

TARGET_VARIABLE = 'loan_status'