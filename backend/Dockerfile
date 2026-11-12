FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi uvicorn sqlalchemy pydantic pydicom \
    numpy opencv-python-headless Pillow httpx python-multipart pytest

COPY . .
EXPOSE 8000
CMD uvicorn backend.app.main:app --host 0.0.0.0 --port 8000