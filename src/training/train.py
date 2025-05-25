# ml_service/src/training/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import ADASYN
from lightgbm import LGBMClassifier
import joblib
import logging
import json
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training():
    logger.info("Loading data...")
    try:
        df = pd.read_parquet(config.PROCESSED_DATA_PATH)
        logger.info(f"Dataset loaded successfully with shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Processed dataset not found at {config.PROCESSED_DATA_PATH}. Run build_dataset.py first.")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    logger.info("Data is expected to be preprocessed by build_dataset.py. Performing final checks...")

    if config.TARGET_VARIABLE not in df.columns:
        logger.error(f"Target variable '{config.TARGET_VARIABLE}' not found in DataFrame. Check build_dataset.py.")
        raise KeyError(f"Target variable '{config.TARGET_VARIABLE}' is missing.")

    missing_features_in_df = [f for f in config.FEATURES_TO_USE if f not in df.columns]
    if missing_features_in_df:
        logger.error(f"Features from config.FEATURES_TO_USE missing in DataFrame: {missing_features_in_df}")
        raise KeyError(f"One or more features are missing: {missing_features_in_df}")

    X = df[config.FEATURES_TO_USE].copy() # Use .copy() to avoid SettingWithCopyWarning later
    y = df[config.TARGET_VARIABLE]

    non_numeric_cols_in_X = X.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols_in_X:
        logger.error(f"Non-numeric columns found in feature set X: {non_numeric_cols_in_X}. build_dataset.py might not have encoded all categoricals.")
        raise ValueError(f"Feature set X contains non-numeric columns: {non_numeric_cols_in_X}")
        
    if X.isnull().sum().sum() > 0:
        logger.warning(f"NaNs found in feature set X. Imputing with median. Check build_dataset.py for thorough NaN handling.")
        for col in X.columns[X.isnull().any()]:
            X.loc[:, col] = X[col].fillna(X[col].median()) # Use .loc to assign back

    logger.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"Class distribution in y_train before ADASYN:\n{y_train.value_counts(normalize=True)}")
    if y_train.nunique() > 1:
        n_minority_samples_in_ytrain = y_train.value_counts().min()
        # ADASYN's n_neighbors must be less than the number of samples in the smallest class.
        # Default is 5. We adjust it if the minority class is too small.
        adasyn_neighbors = min(5, n_minority_samples_in_ytrain - 1) if n_minority_samples_in_ytrain > 1 else 1
        if adasyn_neighbors < 1: adasyn_neighbors = 1 # Smallest possible value for n_neighbors if minority is 1
        
        if n_minority_samples_in_ytrain <= adasyn_neighbors :
             logger.warning(f"Minority class in y_train has {n_minority_samples_in_ytrain} samples, which is not greater than n_neighbors={adasyn_neighbors}. Skipping ADASYN.")
             X_train_resampled, y_train_resampled = X_train_scaled, y_train
        else:
            logger.info(f"Applying ADASYN for class imbalance with n_neighbors={adasyn_neighbors}...")
            adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=adasyn_neighbors)
            try:
                X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_scaled, y_train)
                logger.info(f"Class distribution in y_train_resampled after ADASYN:\n{pd.Series(y_train_resampled).value_counts(normalize=True)}")
            except Exception as e_adasyn:
                logger.error(f"Error during ADASYN: {e_adasyn}. Using original training data.")
                X_train_resampled, y_train_resampled = X_train_scaled, y_train
    else:
        logger.warning("Only one class present in y_train. Skipping ADASYN.")
        X_train_resampled, y_train_resampled = X_train_scaled, y_train
        
    logger.info("Training LightGBM model with GridSearchCV...")
    lgbm = LGBMClassifier(random_state=42, objective='binary') # metric will be set by scoring in GridSearchCV
    
    # Calculate imbalance ratio from the original y_train (before ADASYN)
    # This helps inform scale_pos_weight if ADASYN doesn't fully balance or is skipped
    original_y_train_counts = y_train.value_counts()
    imbalance_ratio_original = 1.0 # Default if only one class or perfectly balanced
    if len(original_y_train_counts) > 1 and original_y_train_counts.get(1, 0) > 0 : # Check if class 1 exists
        imbalance_ratio_original = original_y_train_counts.get(0,0) / original_y_train_counts.get(1,0)
    logger.info(f"Original y_train imbalance ratio (class 0 / class 1): {imbalance_ratio_original:.2f}")

    scale_pos_weight_options = [round(imbalance_ratio_original), 25, 50, 75, 100]
    # Remove duplicates and ensure values are at least 1
    scale_pos_weight_options = sorted(list(set(max(1, round(val)) for val in scale_pos_weight_options)))


    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 8], # Slightly deeper trees
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 40, 50], # Max number of leaves in one tree
        'scale_pos_weight': scale_pos_weight_options
    }
    
    # Scoring metric that focuses on the minority class recall or a balance
    # 'recall_macro' averages recall for each class without weighting by support
    # 'f1_macro' averages F1 for each class without weighting by support
    # Or create a custom scorer for recall of the positive class (class 1)
    recall_positive_scorer = make_scorer(recall_score, pos_label=1, zero_division=0)

    grid_search = GridSearchCV(lgbm, param_grid, cv=3, scoring=recall_positive_scorer, n_jobs=-1, verbose=2) # Changed scoring
    
    logger.info(f"Starting GridSearchCV with param_grid: {param_grid} and scoring: 'recall_class1'")
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_lgbm = grid_search.best_estimator_
    logger.info(f"Best parameters found by GridSearchCV: {grid_search.best_params_}")
    logger.info(f"Best score (recall for class 1) from GridSearchCV: {grid_search.best_score_:.4f}")


    logger.info("Calibrating model probabilities using the best estimator...")
    # Fit calibration on the same resampled data used for grid search for consistency
    calibrated_lgbm = CalibratedClassifierCV(best_lgbm, method='isotonic', cv=3) 
    calibrated_lgbm.fit(X_train_resampled, y_train_resampled)

    logger.info("Evaluating calibrated model on the test set...")
    y_pred = calibrated_lgbm.predict(X_test_scaled)
    y_prob = calibrated_lgbm.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_class1': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        'recall_class1': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        'f1_score_class1': f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob) if y_test.nunique() > 1 else 0.5
    }
    logger.info(f"Model Performance Metrics on Test Set: {metrics}")
    logger.info(f"Classification Report on Test Set:\n{classification_report(y_test, y_pred, zero_division=0)}")

    logger.info("Saving model artifacts...")
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(calibrated_lgbm, config.ARTIFACTS_DIR / config.MODEL_NAME)
    joblib.dump(scaler, config.ARTIFACTS_DIR / config.SCALER_NAME)
    joblib.dump(config.FEATURES_TO_USE, config.ARTIFACTS_DIR / config.FEATURES_NAME)
    
    with open(config.ARTIFACTS_DIR / config.METRICS_NAME, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    logger.info("Training pipeline completed successfully.")

if __name__ == '__main__':
    run_training()