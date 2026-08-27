FROM python:3.11-slim

# libgomp1 : requis par scikit-learn / LightGBM (OpenMP) — leçon P1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Spark n'est pas requis pour le SERVING (scoring seul) : on pourrait alléger
# l'image en retirant pyspark ici. À arbitrer au moment du déploiement.
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

ENV PORT=8080
CMD exec uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT}
