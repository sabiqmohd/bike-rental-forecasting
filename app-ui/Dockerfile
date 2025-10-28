FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app

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

COPY app-ui/requirements.txt .
RUN pip install -r requirements.txt

COPY app-ui/ .
COPY app-ml/src ./src/
COPY common ./common/

EXPOSE 8050

CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050"]