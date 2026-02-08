# Stage 1: install Python deps
FROM python:3.12.8-slim AS builder
WORKDIR /build
COPY agent/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Stage 2: runtime
FROM python:3.12.8-slim

COPY --from=builder /install /usr/local
WORKDIR /app
COPY agent/ /app/agent/
COPY dashboard/ /app/dashboard/
COPY ml/models/ /app/ml/models/
RUN mkdir -p /app/agent/positions /app/agent/decisions /app/logs
CMD ["python", "agent/agent.py"]
