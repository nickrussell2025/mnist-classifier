# Build stage
FROM python:3.13.7-slim-bookworm AS builder

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first (cached layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Copy application code
COPY . /app

# Install project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Runtime stage
FROM python:3.13.7-slim-bookworm

# Create non-root user (HuggingFace requires UID 1000)
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# Copy virtual environment from builder
COPY --from=builder --chown=user:user /app /home/user/app

USER user

# Activate venv by setting PATH
ENV PATH="/home/user/app/.venv/bin:$PATH"

CMD ["python", "predict.py"]