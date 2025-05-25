# ml_service/src/api/processing.py

import pandas as pd

def preprocess_for_prediction(data: dict, le_homeownership, le_verified_income, le_grade, features_list: list) -> pd.DataFrame:
    """
    Preprocesses raw JSON input data for a single prediction.
    
    Args:
        data (dict): A dictionary containing the raw input features.
        le_homeownership, le_verified_income, le_grade: Loaded LabelEncoder objects.
        features_list (list): The list of features the model was trained on.
        
    Returns:
        pd.DataFrame: A preprocessed DataFrame ready for scaling and prediction.
    """
    # Convert the dictionary to a single-row DataFrame
    df = pd.DataFrame([data])
    
    # --- Data Cleaning and Type Conversion ---
    # Apply the same logic as in training for fields that might be missing or in wrong format
    df['emp_length'] = pd.to_numeric(df.get('emp_length', 0), errors='coerce').fillna(0)
    
    # --- Categorical Encoding ---
    # Use the loaded encoders to transform the raw string values
    df['homeownership'] = le_homeownership.transform(df['homeownership'].str.upper())
    df['verified_income'] = le_verified_income.transform(df['verified_income'].str.title())
    df['grade'] = le_grade.transform(df['grade'].str.upper())
    
    # --- Feature Engineering ---
    # Re-create the same engineered features
    df['annual_income'] = df['annual_income'].clip(lower=1)
    df['loan_to_income'] = df['loan_amount'] / df['annual_income']
    df['grade_interest_rate'] = df['grade'] * df['interest_rate']
    df['income_debt_ratio'] = df['annual_income'] / (df.get('debt_to_income', 0) + 0.01)
    
    # --- Column Alignment ---
    # Ensure columns are in the exact same order as during training
    try:
        df_aligned = df[features_list]
    except KeyError as e:
        raise ValueError(f"Input data is missing required feature: {e}") from e
        
    return df_aligned