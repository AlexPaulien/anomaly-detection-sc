FROM python:3.11-slim

# libgomp1: required by scikit-learn / LightGBM (OpenMP) — lesson from P1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Spark is not required for SERVING (scoring only): the image could be slimmed
# down by dropping pyspark here. To be decided at deployment time.
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

ENV PORT=8080
CMD exec uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT}
