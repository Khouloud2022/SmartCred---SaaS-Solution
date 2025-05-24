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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load CSV data
csv_path = 'loans_full_schema.csv'
try:
    df = pd.read_csv(csv_path)
    logger.info("Dataset loaded successfully")
except FileNotFoundError:
    logger.error(f"Dataset not found at {csv_path}")
    raise

# Data cleaning
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

# Handle categorical variables
le_homeownership = LabelEncoder()
le_verified_income = LabelEncoder()
le_grade = LabelEncoder()
df['homeownership'] = le_homeownership.fit_transform(df['homeownership'])
df['verified_income'] = le_verified_income.fit_transform(df['verified_income'])
df['grade'] = le_grade.fit_transform(df['grade'])

# Handle loan_status (target)
def map_loan_status(status):
    status = str(status).strip().lower()
    good_status = ['fully paid', 'paid', 'current']
    bad_status = ['default', 'charged off', 'late', 'in grace period']
    if status in good_status or status == '0':
        return 0
    elif status in bad_status or status.startswith('late') or status == '1':
        return 1
    return 0

df['loan_status'] = df['loan_status'].apply(map_loan_status)

# Feature engineering
df['annual_income'] = df['annual_income'].clip(lower=1000)  # Prevent division by near-zero
df['loan_to_income'] = df['loan_amount'] / df['annual_income']
df['grade_interest_rate'] = df['grade'] * df['interest_rate']
df['income_debt_ratio'] = df['annual_income'] / (df['debt_to_income'] + 0.01)  # Robust offset

# Minimal capping at 99.9th percentile to allow high-risk signals
for col in ['loan_to_income', 'income_debt_ratio', 'interest_rate', 'debt_to_income', 'loan_amount']:
    cap = df[col].quantile(0.999)
    df[col] = df[col].clip(upper=cap)
    logger.info(f"{col} capped at 99.9th percentile: {cap}")

# Check for infinities or NaNs
logger.info(f"NaNs in features: {df[['loan_to_income', 'income_debt_ratio']].isna().sum()}")
logger.info(f"Infinities in features: {np.isinf(df[['loan_to_income', 'income_debt_ratio']]).sum()}")

# Features and target
features = ['annual_income', 'debt_to_income', 'emp_length', 'homeownership', 
            'verified_income', 'loan_amount', 'interest_rate', 'term', 'grade', 
            'late_payments', 'loan_to_income', 'grade_interest_rate', 'income_debt_ratio']
X = df[features]
y = df['loan_status']

# Check class balance
logger.info(f"Class distribution:\n{y.value_counts(normalize=True)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply ADASYN
adasyn = ADASYN(sampling_strategy=0.5, random_state=42)
X_train_scaled, y_train = adasyn.fit_resample(X_train_scaled, y_train)
logger.info(f"Post-ADASYN class distribution:\n{pd.Series(y_train).value_counts(normalize=True)}")

# Train LightGBM with cost-sensitive learning
lgbm = LGBMClassifier(
    random_state=42,
    scale_pos_weight=100,
    objective='binary',
    metric='f1'
)

# Grid search for LightGBM
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'num_leaves': [31, 50]
}
grid_search = GridSearchCV(lgbm, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
best_lgbm = grid_search.best_estimator_

# Calibrate probabilities
calibrated_lgbm = CalibratedClassifierCV(best_lgbm, method='sigmoid', cv=5)
calibrated_lgbm.fit(X_train_scaled, y_train)

# Evaluate on test set
y_pred = calibrated_lgbm.predict(X_test_scaled)
y_prob = calibrated_lgbm.predict_proba(X_test_scaled)[:, 1]
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_prob)
}
logger.info(f"LightGBM performance: {metrics}")
logger.info(f"LightGBM classification report:\n{classification_report(y_test, y_pred)}")

# Feature importance
feature_importances = pd.Series(best_lgbm.feature_importances_, index=features)
logger.info(f"Feature importances:\n{feature_importances.sort_values(ascending=False)}")

# Save model, scaler, encoders, and metrics
joblib.dump(calibrated_lgbm, 'credit_risk_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_homeownership, 'le_homeownership.pkl')
joblib.dump(le_verified_income, 'le_verified_income.pkl')
joblib.dump(le_grade, 'le_grade.pkl')
joblib.dump(features, 'selected_features.pkl')
joblib.dump(metrics, 'model_metrics.pkl')
logger.info("Model artifacts and metrics saved")