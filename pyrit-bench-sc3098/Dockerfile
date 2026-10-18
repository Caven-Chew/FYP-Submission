FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    DISPLAY=:99

WORKDIR /app

COPY requirements.txt ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cargo \
        rustc \
        xvfb \
        x11vnc \
        novnc \
        websockify \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt \
    && patchright install --with-deps chromium

COPY app ./app
COPY docker-entrypoint.sh /usr/local/bin/sc3098-start
RUN chmod +x /usr/local/bin/sc3098-start

EXPOSE 5000
EXPOSE 6080

ENTRYPOINT ["/usr/local/bin/sc3098-start"]
