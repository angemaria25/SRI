FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY modulos ./modulos
COPY interfaz ./interfaz
COPY datos ./datos
COPY local ./local
COPY documentacion ./documentacion
COPY main.py ./main.py
COPY README.md ./README.md

CMD ["python", "main.py"]
