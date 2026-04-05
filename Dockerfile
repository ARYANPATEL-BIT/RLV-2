FROM python:3.11-slim

# No apt packages needed — python:3.11-slim has everything we need

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy only the files that matter
COPY env.py .
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "7860"]
