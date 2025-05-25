# ml_service/src/api/app.py

import flask
import joblib
import logging
import config
from src.api.processing import preprocess_for_prediction

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = flask.Flask(__name__)

# --- Load Artifacts On Startup ---
try:
    model = joblib.load(config.ARTIFACTS_DIR / config.MODEL_NAME)
    scaler = joblib.load(config.ARTIFACTS_DIR / config.SCALER_NAME)
    le_homeownership = joblib.load(config.ARTIFACTS_DIR / config.HOMEOWNERSHIP_ENCODER_NAME)
    le_verified_income = joblib.load(config.ARTIFACTS_DIR / config.VERIFIED_INCOME_ENCODER_NAME)
    le_grade = joblib.load(config.ARTIFACTS_DIR / config.GRADE_ENCODER_NAME)
    features_list = joblib.load(config.ARTIFACTS_DIR / config.FEATURES_NAME)
    logging.info("All model artifacts loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"FATAL: Could not load model artifact: {e}. The application cannot start.")
    model = None # Prevents app from running in a broken state

# --- API Endpoints ---
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify service is running."""
    return flask.jsonify({"status": "ok", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    if not model:
        return flask.jsonify({"error": "Model is not available. Please check server logs."}), 503

    try:
        json_data = flask.request.get_json()
        if not json_data:
            return flask.jsonify({"error": "No input data provided."}), 400

        logging.info(f"Received prediction request with data: {json_data}")

        # --- Preprocessing ---
        processed_df = preprocess_for_prediction(
            data=json_data,
            le_homeownership=le_homeownership,
            le_verified_income=le_verified_income,
            le_grade=le_grade,
            features_list=features_list
        )

        # --- Scaling ---
        scaled_features = scaler.transform(processed_df)

        # --- Prediction ---
        prediction_val = model.predict(scaled_features)[0]
        prediction_prob = model.predict_proba(scaled_features)[0]
        
        risk_probability = prediction_prob[1] # Probability of the '1' class (high risk)
        
        response = {
            'prediction_label': 'High Risk' if prediction_val == 1 else 'Low Risk',
            'prediction_value': int(prediction_val),
            'probability_of_risk_percent': round(risk_probability * 100, 2)
        }
        
        logging.info(f"Prediction result: {response}")
        return flask.jsonify(response)

    except ValueError as e:
        logging.error(f"ValueError during prediction: {e}")
        return flask.jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        return flask.jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    # This block is for local development only.
    # In production, a WSGI server like Gunicorn will run the 'app' object.
    app.run(host='0.0.0.0', port=5000, debug=True)