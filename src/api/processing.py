# ml_service/src/api/processing.py
import pandas as pd
import numpy as np
import re # For string cleaning in emp_length, term
from datetime import datetime # For default application_date
import logging # Optional: for logging within this module

# Configure logging if you want specific logs from this module
# logger = logging.getLogger(__name__) 
# logging.basicConfig(level=logging.INFO) # Or inherit from app.py's config

def preprocess_input_for_prediction(data: dict, expected_model_cols: list) -> pd.DataFrame:
    """
    Preprocesses raw JSON input data to match the model's training format.
    Handles basic cleaning, feature engineering, one-hot encoding, and column alignment.
    """
    df = pd.DataFrame([data]) # Convert single JSON object to a DataFrame row

    # === Basic Cleaning and Feature Engineering (Mirror from build_dataset.py) ===
    
    # Employment Length (e.g., "5 years", "10+ years", "< 1 year" to numeric)
    if 'emp_length' in df.columns:
        emp_val = df['emp_length'].iloc[0]
        if pd.isna(emp_val) or str(emp_val).lower() in ['nan', 'n/a', '0 years', '< 1 year']:
            df.loc[0, 'emp_length'] = 0.0
        elif isinstance(emp_val, str) and '10+' in emp_val.lower():
            df.loc[0, 'emp_length'] = 10.0
        elif isinstance(emp_val, str):
            match = re.search(r'(\d+)', emp_val)
            df.loc[0, 'emp_length'] = float(match.group(1)) if match else 0.0
        elif isinstance(emp_val, (int, float)):
             df.loc[0, 'emp_length'] = float(emp_val)
        else:
            df.loc[0, 'emp_length'] = 0.0 # Fallback
        df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce').fillna(0.0)

    # Term (e.g., "36 months" to 36.0)
    if 'term' in df.columns:
        term_val = df['term'].iloc[0]
        if isinstance(term_val, str):
            match = re.search(r'(\d+)', term_val)
            df.loc[0, 'term'] = float(match.group(1)) if match else np.nan
        elif isinstance(term_val, (int, float)):
            df.loc[0, 'term'] = float(term_val)
        else:
            df.loc[0, 'term'] = np.nan # Fallback
        df['term'] = pd.to_numeric(df['term'], errors='coerce').fillna(36.0) # Default to 36 if unparseable

    # Credit History Years
    # Requires 'earliest_cr_line' in input (e.g., "Jan-2005" or "2005-01-15")
    # 'application_date' can also be passed in JSON or defaults to today
    if 'earliest_cr_line' in data: # Check original data dict
        try:
            earliest_cr_line_dt = pd.to_datetime(data['earliest_cr_line'], errors='coerce')
            app_date_str = data.get('application_date', datetime.now().strftime('%Y-%m-%d'))
            application_date_dt = pd.to_datetime(app_date_str, errors='coerce')

            if pd.notna(earliest_cr_line_dt) and pd.notna(application_date_dt):
                df['credit_history_years'] = (application_date_dt - earliest_cr_line_dt).days / 365.25
            else:
                df['credit_history_years'] = np.nan
        except Exception:
            df['credit_history_years'] = np.nan
    elif 'credit_history_years' in expected_model_cols: # If expected by model but not calculable
        df['credit_history_years'] = np.nan # Will be imputed next

    # Impute key numeric features (expected by the model) if they are missing or became NaN
    # This list should ideally come from a config or be based on features that received imputation during training
    numeric_cols_to_impute = ['annual_inc', 'dti', 'loan_amnt', 'int_rate', 'fico_range_low', 'credit_history_years', 'emp_length', 'term']
    for col in numeric_cols_to_impute:
        if col in df.columns: # If column exists in current df (some engineered ones might not yet)
            df[col] = pd.to_numeric(df[col], errors='coerce') # Ensure numeric
            if df[col].isnull().any():
                # IMPORTANT: For API, use medians SAVED from the TRAINING SET.
                # For simplicity now, filling with 0 or a placeholder.
                # In production, load saved medians.
                logging.warning(f"API input for '{col}' is NaN. Filling with 0. Load trained medians for production.")
                df.loc[0, col] = 0.0
        elif col in expected_model_cols: # If model expects it but not in input or engineered yet
            logging.warning(f"API input missing expected feature '{col}', filling with 0.")
            df[col] = 0.0
            
    # --- One-Hot Encoding for Categorical Features ---
    # Get list of object columns present in the input df that need encoding
    # These are the original categorical feature names BEFORE they are dummified
    # e.g., 'home_ownership', 'grade', 'purpose', etc.
    input_categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    # Fill NaNs in categorical columns with a placeholder string before dummification
    for col in input_categorical_cols:
        df[col] = df[col].astype(str).fillna('Missing')

    if input_categorical_cols:
        # drop_first=False initially to create all dummies, then reindex will handle it.
        # Or, if you saved the categories for each OHE column from training, use that.
        df = pd.get_dummies(df, columns=input_categorical_cols, drop_first=False, dtype=float)

    # --- Align columns with the model's training columns ---
    df_aligned = df.reindex(columns=expected_model_cols, fill_value=0.0)
    
    return df_aligned