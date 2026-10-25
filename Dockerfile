# Production-ready Dockerfile for Supply Chain Intelligence Platform
FROM python:3.12-slim

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    DATABASE_PATH=/app/data/curated/supply_chain.duckdb

# Install system dependencies (curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application directory and non-root user
WORKDIR /app
RUN useradd -m -u 1000 appuser

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and data assets
COPY . .

# Run pipeline to ensure database and DQ report exist
RUN python -m src.data.dq_check --input ./data/raw --output ./reports/dq_report.json && \
    python -m src.data.pipeline --raw-dir ./data/raw --db-path ./data/curated/supply_chain.duckdb

# Set correct ownership for non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server via Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
