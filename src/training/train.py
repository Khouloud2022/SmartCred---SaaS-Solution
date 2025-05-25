# ml_service/src/training/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import ADASYN
from lightgbm import LGBMClassifier
import joblib
import logging
import json

# Import the configuration variables
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training():
    """Main function to run the model training pipeline."""
    
    # --- Load Data ---
    logger.info("Loading data...")
    try:
        df = pd.read_parquet(config.PROCESSED_DATA_PATH)
        #df = pd.read_csv(config.RAW_DATA_PATH)
        logger.info(f"Dataset loaded successfully with shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Dataset not found at {config.RAW_DATA_PATH}")
        raise

    # --- Data Cleaning ---
    logger.info("Cleaning data...")
    df['annual_income'] = pd.to_numeric(df['annual_income'], errors='coerce').fillna(df['annual_income'].median())
    df['debt_to_income'] = pd.to_numeric(df['debt_to_income'], errors='coerce').fillna(df['debt_to_income'].median())
    df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce').fillna(0)
    df['loan_amount'] = pd.to_numeric(df['loan_amount'], errors='coerce').fillna(df['loan_amount'].median())
    df['interest_rate'] = pd.to_numeric(df['interest_rate'], errors='coerce').fillna(df['interest_rate'].median())
    df['term'] = pd.to_numeric(df['term'], errors='coerce').fillna(df['term'].mode()[0])
    if 'late_payments' not in df.columns:
        df['late_payments'] = 0
    df['late_payments'] = pd.to_numeric(df['late_payments'], errors='coerce').fillna(0)
    df['homeownership'] = df['homeownership'].fillna('UNKNOWN').str.upper()
    df['verified_income'] = df['verified_income'].fillna('UNKNOWN').str.title()
    df['grade'] = df['grade'].fillna('UNKNOWN').str.upper()

    # --- Handle Categorical Variables ---
    le_homeownership = LabelEncoder()
    le_verified_income = LabelEncoder()
    le_grade = LabelEncoder()
    df['homeownership'] = le_homeownership.fit_transform(df['homeownership'])
    df['verified_income'] = le_verified_income.fit_transform(df['verified_income'])
    df['grade'] = le_grade.fit_transform(df['grade'])

    # --- Handle Target Variable ---
    def map_loan_status(status):
        status = str(status).strip().lower()
        good_status = ['fully paid', 'paid', 'current']
        bad_status = ['default', 'charged off', 'late', 'in grace period']
        if status in good_status or status == '0':
            return 0
        elif status in bad_status or status.startswith('late') or status == '1':
            return 1
        return 0 # Default to good status if unknown
    df[config.TARGET_VARIABLE] = df[config.TARGET_VARIABLE].apply(map_loan_status)

    # --- Feature Engineering ---
    logger.info("Performing feature engineering...")
    df['annual_income'] = df['annual_income'].clip(lower=1000)
    df['loan_to_income'] = df['loan_amount'] / df['annual_income']
    df['grade_interest_rate'] = df['grade'] * df['interest_rate']
    df['income_debt_ratio'] = df['annual_income'] / (df['debt_to_income'] + 0.01)

    # --- Capping Outliers ---
    for col in ['loan_to_income', 'income_debt_ratio', 'interest_rate', 'debt_to_income', 'loan_amount']:
        if col in df.columns:
            cap = df[col].quantile(0.999)
            df[col] = df[col].clip(upper=cap)

    # --- Final Feature Selection ---
    X = df[config.FEATURES_TO_USE]
    y = df[config.TARGET_VARIABLE]

    # --- Data Splitting ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Handling Imbalance with ADASYN ---
    logger.info(f"Class distribution before ADASYN:\n{y_train.value_counts(normalize=True)}")
    adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=5)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_scaled, y_train)
    logger.info(f"Class distribution after ADASYN:\n{pd.Series(y_train_resampled).value_counts(normalize=True)}")

    # --- Model Training with GridSearchCV ---
    logger.info("Training LightGBM model with GridSearchCV...")
    lgbm = LGBMClassifier(random_state=42, objective='binary', metric='f1')
    param_grid = {
        'n_estimators': [150, 250],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'scale_pos_weight': [50, 100] # Weight for the positive class (defaults)
    }
    grid_search = GridSearchCV(lgbm, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_lgbm = grid_search.best_estimator_
    logger.info(f"Best parameters found: {grid_search.best_params_}")

    # --- Probability Calibration ---
    logger.info("Calibrating model probabilities...")
    calibrated_lgbm = CalibratedClassifierCV(best_lgbm, method='isotonic', cv=3)
    calibrated_lgbm.fit(X_train_resampled, y_train_resampled)

    # --- Evaluation ---
    logger.info("Evaluating model on the test set...")
    y_pred = calibrated_lgbm.predict(X_test_scaled)
    y_prob = calibrated_lgbm.predict_proba(X_test_scaled)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    logger.info(f"Model Performance Metrics: {metrics}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # --- Save Artifacts ---
    logger.info("Saving model artifacts...")
    joblib.dump(calibrated_lgbm, config.ARTIFACTS_DIR / config.MODEL_NAME)
    joblib.dump(scaler, config.ARTIFACTS_DIR / config.SCALER_NAME)
    joblib.dump(le_homeownership, config.ARTIFACTS_DIR / config.HOMEOWNERSHIP_ENCODER_NAME)
    joblib.dump(le_verified_income, config.ARTIFACTS_DIR / config.VERIFIED_INCOME_ENCODER_NAME)
    joblib.dump(le_grade, config.ARTIFACTS_DIR / config.GRADE_ENCODER_NAME)
    joblib.dump(config.FEATURES_TO_USE, config.ARTIFACTS_DIR / config.FEATURES_NAME)
    
    with open(config.ARTIFACTS_DIR / config.METRICS_NAME, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    logger.info("Training pipeline completed successfully.")


if __name__ == '__main__':
    run_training()