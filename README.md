# Credit Risk ML Service

This service provides a machine learning model to predict credit risk based on loan application data.

## Project Structure

-   **/artifacts**: Contains all trained model artifacts (model, scaler, encoders).
-   **/data**: Contains the raw training data.
-   **/notebooks**: Jupyter notebooks for exploratory data analysis.
-   **/src**: All Python source code.
    -   `src/training`: Scripts for training the model.
    -   `src/api`: The Flask prediction API.
-   **/tests**: Unit and integration tests.
-   **config.py**: Central configuration for paths and parameters.
-   **Dockerfile**: For building the production Docker image.

## How to Use

### 1. Setup

Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Model Training

To generate the model artifacts from the raw data, run the training script:
```bash
python -m src.training.train
```
This will populate the `/artifacts` directory.

### 3. Run the API Locally

For local development, you can run the Flask app directly:
```bash
python -m src.api.app
```
The API will be available at `http://localhost:5000`.

### 4. Build and Run with Docker

For a production-like environment, build and run the Docker container:
```bash
# Build the image
docker build -t credit-risk-api .

# Run the container
docker run -p 5000:5000 credit-risk-api
```

### 5. API Endpoints

-   **`GET /health`**: Health check for the service.
-   **`POST /predict`**: Main prediction endpoint.

#### Example `/predict` Request

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "annual_income": 80000,
    "debt_to_income": 12.0,
    "emp_length": 6,
    "homeownership": "MORTGAGE",
    "verified_income": "Source Verified",
    "loan_amount": 15000,
    "interest_rate": 11.5,
    "term": 36,
    "grade": "B",
    "late_payments": 1
}'
```