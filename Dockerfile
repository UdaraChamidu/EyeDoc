# -------- Base --------
FROM python:3.10-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# -------- Python deps --------
# If you prefer not to have a requirements.txt, you can inline pip installs here instead.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code (includes your model + vectorstore)
COPY . /app

# Optional: pre-download the sentence-transformers model to speed first run
# RUN python - <<'PY'
# from sentence_transformers import SentenceTransformer
# SentenceTransformer("all-MiniLM-L12-v2")
# PY

# Expose Streamlit default (HF will set $PORT)
EXPOSE 7860

# Healthcheck (optional)
HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# Run Streamlit
CMD bash -lc "streamlit run app.py --server.address 0.0.0.0 --server.port $PORT"