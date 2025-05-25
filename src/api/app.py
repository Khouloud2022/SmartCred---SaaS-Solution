# ml_service/src/api/app.py

import flask
import joblib
import pandas as pd
import numpy as np
import logging
import json
import re # For string cleaning
from datetime import datetime # For default application_date

# Corrected imports based on file locations within 'src' package
from src import config as app_config # config.py is in src/
from src.api.processing import preprocess_input_for_prediction # processing.py is in src/api/

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Load Model Artifacts On Startup ---
# Paths are now relative to BASE_DIR defined in config.py,
# and config.py itself knows where its BASE_DIR (src/) is.
MODEL_PATH = app_config.ARTIFACTS_DIR / app_config.MODEL_NAME
SCALER_PATH = app_config.ARTIFACTS_DIR / app_config.SCALER_NAME
FEATURES_PATH = app_config.ARTIFACTS_DIR / app_config.FEATURES_NAME

model = None
scaler = None
model_columns = None 

try:
    model = joblib.load(MODEL_PATH)
    logging.info(f"Model loaded successfully from {MODEL_PATH}")
    scaler = joblib.load(SCALER_PATH)
    logging.info(f"Scaler loaded successfully from {SCALER_PATH}")
    
    model_columns_raw = joblib.load(FEATURES_PATH)
    if isinstance(model_columns_raw, np.ndarray):
        model_columns = model_columns_raw.tolist()
    elif isinstance(model_columns_raw, list) and len(model_columns_raw) > 0 and isinstance(model_columns_raw[0], list):
        model_columns = model_columns_raw[0] 
    else:
        model_columns = model_columns_raw

    if not isinstance(model_columns, list) or not all(isinstance(col, str) for col in model_columns):
        raise ValueError(f"Loaded 'model_columns' is not a flat list of strings: {model_columns}")
        
    logging.info(f"Model features (columns) loaded successfully from {FEATURES_PATH}. Expecting {len(model_columns)} features.")
    # logging.debug(f"Model columns: {model_columns[:20]}...")

except FileNotFoundError as e:
    logging.error(f"FATAL: Could not load model artifact: {e}. Ensure 'train.py' has run successfully and artifacts are in {app_config.ARTIFACTS_DIR}.")
except ValueError as e:
    logging.error(f"FATAL: Error with loaded model_columns: {e}")
except Exception as e:
    logging.error(f"FATAL: An unexpected error occurred loading artifacts: {e}")

app = flask.Flask(__name__)

# --- API Endpoints ---
@app.route('/health', methods=['GET'])
def health_check():
    if model and scaler and model_columns:
        return flask.jsonify({"status": "ok", "message": "ML service is healthy."})
    else:
        return flask.jsonify({"status": "error", "message": "ML service is unhealthy. Model artifacts not loaded."}), 503

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler or not model_columns:
        return flask.jsonify({"error": "Model artifacts not loaded. Check server logs."}), 503

    try:
        json_data = flask.request.get_json()
        if not json_data:
            return flask.jsonify({"error": "No input data provided."}), 400
        
        logging.info(f"Received prediction request with data: {json_data}")

        # Preprocess the input data using the function from processing.py
        # Pass the expected model columns (which includes one-hot encoded names)
        processed_df = preprocess_input_for_prediction(json_data, model_columns)
        
        if processed_df.shape[1] != len(model_columns):
            logging.error(f"Preprocessed data has {processed_df.shape[1]} columns, model expects {len(model_columns)}.")
            logging.error(f"Model expects: {model_columns}")
            logging.error(f"Processed has: {processed_df.columns.tolist()}")
            return flask.jsonify({"error": "Feature mismatch after preprocessing. Check server logs."}), 500

        try:
            scaled_features = scaler.transform(processed_df)
        except ValueError as ve:
            logging.error(f"Error during scaling: {ve}. Feature names mismatch or unexpected NaNs/Infs.")
            logging.error(f"Columns sent to scaler: {processed_df.columns.tolist()}")
            if processed_df.isnull().sum().sum() > 0:
                logging.error("NaNs found in data before scaling:")
                logging.error(processed_df[processed_df.isnull().any(axis=1)])
            if np.isinf(processed_df.values.astype(np.float64)).any():
                 logging.error("Infinities found in data before scaling:")
                 logging.error(processed_df[np.isinf(processed_df.values.astype(np.float64)).any(axis=1)])
            return flask.jsonify({"error": f"Error during feature scaling: {ve}"}), 500

        prediction_val = model.predict(scaled_features)[0]
        prediction_prob = model.predict_proba(scaled_features)[0]
        
        risk_probability_class1 = prediction_prob[1]
        
        response = {
            'prediction_label': 'High Risk' if prediction_val == 1 else 'Low Risk',
            'prediction_value': int(prediction_val),
            'probability_of_risk_percent': round(risk_probability_class1 * 100, 2)
        }
        
        logging.info(f"Prediction result: {response}")
        return flask.jsonify(response)

    except KeyError as e:
        logging.error(f"KeyError during prediction: {e}. Input data missing required field for preprocessing.")
        return flask.jsonify({"error": f"Missing expected field in input: {str(e)}"}), 400
    except ValueError as e:
        logging.error(f"ValueError during prediction: {e}")
        return flask.jsonify({"error": f"Invalid data format or value: {str(e)}"}), 400
    except Exception as e:
        logging.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        return flask.jsonify({"error": "An internal server error occurred during prediction."}), 500

if __name__ == '__main__':
    # This block is for local development using 'python -m src.api.app'
    # Gunicorn in Docker will directly import 'app' from 'src.api.app'
    logging.info(f"Starting Flask development server on host 0.0.0.0, port 5000")
    app.run(host='0.0.0.0', port=5000, debug=True)