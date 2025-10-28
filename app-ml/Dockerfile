FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        cmake \
        python3-dev \
        libfreetype6-dev \
        pkg-config \
        && rm -rf /var/lib/apt/lists/*

COPY app-ml/requirements.txt .

RUN pip install -r requirements.txt

COPY app-ml/entrypoint/ ./entrypoint/
COPY app-ml/src/ ./src/
COPY common/ ./common/

ENV PYTHONPATH="/app:/app/common:/app/src"

# CMD ["python", "entrypoint/train.py"]