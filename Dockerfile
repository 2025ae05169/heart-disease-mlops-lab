
# Use Python 3.11
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Display Python logs immediately
ENV PYTHONUNBUFFERED=1

# Copy API dependencies
COPY requirements-api.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy FastAPI application
COPY app/ ./app/

# Copy preprocessing code
COPY src/ ./src/

# Copy the already trained model
COPY artifacts/heart_disease_model.joblib ./artifacts/heart_disease_model.joblib

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
