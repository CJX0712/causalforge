# CausalForge — reproducible runtime image
# Build:  docker build -t causalforge:0.1.0 .
# Run:    docker run --rm causalforge:0.1.0 benchmark
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY causalforge ./causalforge
COPY ruff.toml ./

# Make the package importable
ENV PYTHONPATH=/app

# Default: run the Monte-Carlo benchmark
ENTRYPOINT ["python", "-m", "causalforge.cli"]
CMD ["benchmark", "--out", "/app/benchmark.json"]
