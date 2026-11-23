# Reproducible pipeline run (CPU). The collector and the labelling page are meant to run on the host.
#   docker build -t sgtriage .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/docs:/app/docs" --env-file .env sgtriage prepare
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/docs:/app/docs" --env-file .env sgtriage evaluate
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY collect.py ./
COPY sgtriage ./sgtriage
COPY site_config.jso[n] ./
ENTRYPOINT ["python", "-m", "sgtriage"]
CMD ["--help"]
