FROM python:3.11.7-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Hugging Face cache control
ENV HF_HOME=/data/huggingface
ENV HF_HUB_CACHE=/data/huggingface/hub

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# run once so it i can later avoid downloading and work offline
RUN python - <<EOF
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')

AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
EOF

COPY src ./src
COPY cv ./cv

ENV PYTHONPATH=/app/src

EXPOSE 8100
CMD ["uvicorn", "ai_chat.server:app", "--host", "0.0.0.0", "--port", "8100"]
