FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies:
# - build-essential: compile ta-lib C library and some python wheels
# - cron: run scheduled jobs inside container
# - tzdata: support TZ=America/New_York
# - libgomp1: LightGBM runtime dependency
# - libpq-dev: build dependencies for postgres drivers (psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cron \
      tzdata \
      curl \
      ca-certificates \
      libgomp1 \
      libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib
# Build with network-enabled `pip install ta-lib` during image build (no local tarball dependency).

# Copy code (exclude large data via .dockerignore).
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY vnpy /app/vnpy
COPY flagship /app/flagship

# Install python deps.
# NOTE:
# - vnpy is installed from local source.
# - flagship code is imported by path (scripts insert PROJECT_ROOT into sys.path).
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install ".[alpha]" \
    && pip install polygon-api-client alpaca-py psycopg2-binary fastapi uvicorn python-jose[cryptography] python-multipart boto3

# Runtime scripts
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && chmod +x /app/flagship/paper_trading/run_full_daily_cycle.sh

ENTRYPOINT ["/entrypoint.sh"]


