# VisionPilot container. Build natively on arm64 (Graviton) for the COOL path.
#
# Baseline build (stock pip OpenCV 5):
#   docker build -t visionpilot:pip .
# COOL build: replace BASE_IMAGE with the COOL container image URI you are entitled to
# from your AWS Marketplace subscription (OpenCV 5 for Graviton4/5), and set USE_COOL=1
# so the pip OpenCV wheel is NOT installed over the optimized build:
#   docker build --build-arg BASE_IMAGE=<cool-image-uri> --build-arg USE_COOL=1 -t visionpilot:cool .
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
ARG USE_COOL=0
WORKDIR /app
COPY requirements.txt .
RUN if [ "$USE_COOL" = "1" ]; then pip install --no-cache-dir numpy==2.5.3; \
    else pip install --no-cache-dir -r requirements.txt; fi
COPY visionpilot ./visionpilot
COPY static ./static
RUN useradd -m app && chown -R app /app
USER app
ENV VP_APPROVALS=/tmp/approvals.jsonl
EXPOSE 8080
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/api/health')"
CMD ["python", "-m", "visionpilot.server", "--port", "8080"]
