FROM python:3.11-slim-buster

WORKDIR /usr/src/app
RUN pip install --upgrade pip

ARG EMBEDDING_MODEL_ID

RUN pip install huggingface_hub
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('$EMBEDDING_MODEL_ID')"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PORT 8000
ENV CURRENT_HOST 0.0.0.0
ENV WORKERS 1

CMD ["sh", "-c", "uvicorn app:app --loop asyncio --reload --workers ${WORKERS} --host $CURRENT_HOST --port $PORT --proxy-headers"]