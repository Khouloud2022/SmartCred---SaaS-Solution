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
# ml_service/config.py
FEATURES_TO_USE = [
    # --- Key Numeric Features (ensure these exact names are in your Parquet columns) ---
    'loan_amnt', 
    'term',        # This should be numeric now (e.g., 36.0, 60.0)
    'int_rate', 
    'installment', 
    'emp_length',  # Numeric
    'annual_inc', 
    'dti', 
    'delinq_2yrs', 
    'fico_range_low', 
    # 'fico_range_high', # Often, just one of the FICO ranges is used, or an average
    'inq_last_6mths', 
    'mths_since_last_delinq', # Might have many NaNs handled by median
    'open_acc', 
    'pub_rec', 
    'revol_bal', 
    'revol_util', 
    'total_acc',
    'credit_history_years', # Your engineered feature

    # --- One-Hot Encoded 'grade' columns (example, if 'A' was dropped) ---
    'grade_B', 
    'grade_C', 
    'grade_D', 
    'grade_E', 
    'grade_F', 
    'grade_G',

    # --- One-Hot Encoded 'home_ownership' columns (example, if one was dropped, e.g. NONE or OTHER if it's rare) ---
    'home_ownership_MORTGAGE', 
    'home_ownership_OWN', 
    'home_ownership_RENT',
    # If 'home_ownership_OTHER' or 'home_ownership_NONE' exists and is relevant, add it.
    # If you only have a few values like 'NONE', 'ANY', they might be dropped by get_dummies(drop_first=True)
    # or grouped into 'Other' if your cardinality reduction did that.
    # Based on your output: 'home_ownership_NONE', 'home_ownership_OTHER' are present, so you'd add them if desired.

    # --- One-Hot Encoded 'verification_status' (example, if 'Not Verified' was dropped) ---
    'verification_status_Source Verified', 
    'verification_status_Verified',

    # --- One-Hot Encoded 'purpose' (example, select the most relevant ones or all if not too many) ---
    'purpose_credit_card', 
    'purpose_debt_consolidation', 
    'purpose_home_improvement', 
    'purpose_major_purchase',
    'purpose_other', # If 'Other' was created due to cardinality reduction
    # ... add other purpose_... columns you deem important

    # --- One-Hot Encoded 'addr_state' (example, if 'AL' was dropped by drop_first=True) ---
    # This will create many columns. Consider if you want to use all of them,
    # or group less frequent states into an 'addr_state_Other' in build_dataset.py.
    # Your output shows all states are dummified. Select the ones you want.
    'addr_state_AR', 'addr_state_AZ', 'addr_state_CA', 'addr_state_CO', # ... and so on

    # --- Other One-Hot Encoded Columns ---
    'initial_list_status_w', # Assuming 'f' was dropped
    'application_type_Joint App', # Assuming 'Individual' was dropped
    'disbursement_method_DirectPay' # Assuming 'Cash' was dropped
    
    # Add ALL other numeric and one-hot encoded columns you want to use as features
    # from the full list of 153 (154 minus 'target') columns in your Parquet file.
]

TARGET_VARIABLE = 'target'