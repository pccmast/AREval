# AREval API Server Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[all]"

# Copy code
COPY areval-engine/ ./areval-engine/
COPY areval-api/ ./areval-api/
COPY areval-sdk/ ./areval-sdk/

# Install in development mode
RUN pip install -e ./areval-engine -e ./areval-api -e ./areval-sdk

EXPOSE 8000

CMD ["uvicorn", "areval_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
