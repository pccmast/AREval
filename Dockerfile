# AREval API Server Dockerfile
# Multi-stage build using uv for reproducible installs

# Stage 1: Build with dependencies
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app

# Enable bytecode compilation and copy-on-write
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Install dependencies (layered for caching)
COPY pyproject.toml uv.lock ./
COPY areval-engine/ ./areval-engine/
COPY areval-api/ ./areval-api/
COPY areval-sdk/ ./areval-sdk/
COPY areval-cli/ ./areval-cli/
COPY README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra all

# Stage 2: Runtime
FROM python:3.12-slim AS runtime
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/areval-engine /app/areval-engine
COPY --from=builder /app/areval-api /app/areval-api
COPY --from=builder /app/areval-sdk /app/areval-sdk
COPY --from=builder /app/pyproject.toml /app/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8700
CMD ["uvicorn", "areval_api.main:app", "--host", "0.0.0.0", "--port", "8700"]
