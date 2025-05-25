# ml_service/Dockerfile

# Use a lean official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code and artifacts
COPY src/ /app/src/
COPY artifacts/ /app/artifacts/

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application using Gunicorn
# It looks for the 'app' object in the 'src.api.app' module
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "src.api.app:app"]