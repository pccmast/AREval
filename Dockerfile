# AREval API Server Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy everything needed for build (must come before install!)
COPY pyproject.toml uv.lock README.md ./
COPY areval-engine/ ./areval-engine/
COPY areval-api/ ./areval-api/
COPY areval-sdk/ ./areval-sdk/
COPY areval-cli/ ./areval-cli/

# Install dependencies via uv (fast, uses lock file)
RUN uv sync --frozen --no-dev --extra all

# Use venv python
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8700
CMD ["uvicorn", "areval_api.main:app", "--host", "0.0.0.0", "--port", "8700"]
