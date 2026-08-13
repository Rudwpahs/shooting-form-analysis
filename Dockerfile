FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HOST=0.0.0.0

WORKDIR /app

COPY packages.txt requirements.txt ./
RUN apt-get update \
    && xargs -a packages.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app ./app
COPY static ./static
COPY models ./models
COPY run.py ./

EXPOSE 10000

# Render injects $PORT. Gunicorn serves the Flask app.
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-10000} -w 1 -t 300 --threads 4 app.server:app"]
