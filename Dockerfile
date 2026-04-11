FROM python:3.11-slim

# Force unbuffered Python stdout/stderr — critical for validator to see structured output
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# No apt packages needed — python:3.11-slim has everything we need

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy all project files
COPY env.py .
COPY inference.py .
COPY models.py .
COPY tasks.py .
COPY client.py .
COPY openenv.yaml .
COPY README.md .
COPY test_env.py .
COPY audit_reward.py .
COPY server/ server/

# Non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Health check for HuggingFace Spaces
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "7860"]
