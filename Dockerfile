# AREval API Server Dockerfile
# Optimized for build-cache: dependency install is a separate layer
# that only rebuilds when pyproject.toml or uv.lock changes.
#
# Build:
#   docker build -t areval-api .
FROM python:3.12-slim

WORKDIR /app

# Install uv (rarely changes)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- Dependency cache layer ---------------------------------------------
# Copy only files that define dependencies; create minimal package stubs
# so that "uv sync" can resolve the local packages declared in pyproject.toml.
COPY pyproject.toml uv.lock README.md ./
RUN mkdir -p areval-engine/areval areval-api/areval_api \
             areval-sdk/areval_sdk areval-cli/areval_cli && \
    for d in areval-engine/areval areval-api/areval_api \
             areval-sdk/areval_sdk areval-cli/areval_cli; do \
        touch "$d/__init__.py"; \
    done
# This layer is cached unless pyproject.toml or uv.lock change.
RUN uv sync --frozen --no-dev --extra all

# ---- Application code layer ---------------------------------------------
# Only the next four COPY lines rebuild on every source-code change.
COPY areval-engine/ ./areval-engine/
COPY areval-api/     ./areval-api/
COPY areval-sdk/     ./areval-sdk/
COPY areval-cli/     ./areval-cli/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8700
CMD ["uvicorn", "areval_api.main:app", "--host", "0.0.0.0", "--port", "8700"]
