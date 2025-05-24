from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load model artifacts
try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('credit_risk_model.pkl')
    le_homeownership = joblib.load('le_homeownership.pkl')
    le_verified_income = joblib.load('le_verified_income.pkl')
    le_grade = joblib.load('le_grade.pkl')
    selected_features = joblib.load('selected_features.pkl')
    logger.info("Model artifacts loaded successfully")
    logger.info(f"Valid grades: {le_grade.classes_}")
    logger.info(f"Valid verified_income: {le_verified_income.classes_}")
    logger.info(f"Valid homeownership: {le_homeownership.classes_}")
except Exception as e:
    logger.error(f"Failed to load model artifacts: {str(e)}")
    raise

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Credit Risk Prediction API. Use POST /predict with JSON data.'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data provided")
            return jsonify({'error': 'No JSON data provided'}), 400

        # Required fields
        required_fields = ['annual_income', 'debt_to_income', 'emp_length', 'homeownership',
                          'verified_income', 'loan_amount', 'interest_rate', 'term', 'grade']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return jsonify({'error': f"Missing required fields: {missing_fields}"}), 400

        # Validate categorical inputs
        homeownership = data['homeownership'].upper()
        if homeownership not in le_homeownership.classes_:
            logger.warning(f"Unknown homeownership value: {homeownership}, defaulting to UNKNOWN")
            homeownership = 'UNKNOWN'

        verified_income = data['verified_income'].title()
        if verified_income not in le_verified_income.classes_:
            logger.warning(f"Unknown verified_income value: {verified_income}, defaulting to UNKNOWN")
            verified_income = 'UNKNOWN'

        grade = data['grade'].upper()
        if grade not in le_grade.classes_:
            logger.warning(f"Unknown grade value: {grade}, defaulting to UNKNOWN")
            grade = 'UNKNOWN'

        # Prepare feature array
        try:
            annual_income = float(data['annual_income'])
            debt_to_income = float(data['debt_to_income'])
            loan_amount = float(data['loan_amount'])
            interest_rate = float(data['interest_rate'])
            grade_encoded = le_grade.transform([grade])[0]
            all_features = np.array([[
                annual_income,
                debt_to_income,
                float(data['emp_length']),
                le_homeownership.transform([homeownership])[0],
                le_verified_income.transform([verified_income])[0],
                loan_amount,
                interest_rate,
                float(data['term']),
                grade_encoded,
                float(data.get('late_payments', 0)),
                loan_amount / annual_income,  # loan_to_income
                grade_encoded * interest_rate,  # grade_interest_rate
                annual_income / (debt_to_income + 1e-6)  # income_debt_ratio
            ]])
            logger.info(f"Input features: {dict(zip(selected_features, all_features[0]))}")
            logger.info(f"Encoded categoricals: homeownership={le_homeownership.transform([homeownership])[0]}, verified_income={le_verified_income.transform([verified_income])[0]}, grade={grade_encoded}")
        except ValueError as e:
            logger.error(f"Invalid numeric input: {str(e)}")
            return jsonify({'error': f"Invalid numeric input: {str(e)}"}), 400

        # Scale features
        scaled_features = scaler.transform(all_features)
        logger.info(f"Scaled features: {dict(zip(selected_features, scaled_features[0]))}")

        # Predict
        risk_prob = model.predict_proba(scaled_features)[0][1]
        risk_class = model.predict(scaled_features)[0]
        risk_label = 'High Risk' if risk_class == 1 else 'Low Risk'

        logger.info(f"LightGBM prediction: risk_probability={risk_prob:.4f}, risk_class={risk_label}")
        return jsonify({
            'model': 'LightGBM',
            'risk_probability': round(risk_prob, 4),
            'risk_class': risk_label
        })

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)