
FROM python:3.11-slim

WORKDIR /app

# System deps for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9000
# Tip: remove --reload in containers for stability; dev can use it locally.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
