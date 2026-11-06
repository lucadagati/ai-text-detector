# GPU: docker run --gpus all -p 127.0.0.1:8790:8790 -v aidet-models:/app/aidet/models ai-text-detector
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -m spacy download en_core_web_md
COPY . .
ENV AIDET_HOST=0.0.0.0
EXPOSE 8790
CMD ["python", "server.py"]
