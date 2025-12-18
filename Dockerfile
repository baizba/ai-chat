# syntax = docker/dockerfile:1.4

FROM python:3.11.7-slim

WORKDIR /app
RUN pip install --no-cache-dir "fastapi[standard]"
COPY src .
CMD ["fastapi", "dev", "main.py", "--host", "0.0.0.0"]
