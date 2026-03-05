FROM python:3.11.7-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Hugging Face cache control
ENV HF_HOME=/data/huggingface
ENV HF_HUB_CACHE=/data/huggingface/hub

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY cv ./cv

ENV PYTHONPATH=/app/src

EXPOSE 8100
CMD ["uvicorn", "ai_chat.server:app", "--host", "0.0.0.0", "--port", "8100"]
