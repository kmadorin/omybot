# Stage 1: install Python deps
FROM python:3.12.8-slim AS builder
WORKDIR /build
COPY agent/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Stage 2: runtime
FROM python:3.12.8-slim

# Install Foundry (pinned nightly for reproducibility)
ENV FOUNDRY_VERSION=nightly-2025-02-01
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* && \
    curl -L https://foundry.paradigm.xyz | bash && \
    /root/.foundry/bin/foundryup --install ${FOUNDRY_VERSION}
ENV PATH="/root/.foundry/bin:${PATH}"

COPY --from=builder /install /usr/local
WORKDIR /app
COPY agent/ /app/agent/
COPY dashboard/ /app/dashboard/
COPY ml/models/ /app/ml/models/
RUN mkdir -p /app/agent/positions /app/agent/decisions /app/logs
CMD ["python", "agent/agent.py"]
