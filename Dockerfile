# Usa la imagen base de Python
FROM python:3.9-slim

# Instala git y dependencias para Whisper, Tesseract y PDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-ita \
    tesseract-ocr-por \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configura el directorio de trabajo
WORKDIR /app

# Copiar el código fuente
COPY . .

# Instala las dependencias, incluyendo whisper
RUN pip install --no-cache-dir -r requirements.txt

# Puerto
EXPOSE 5000

# Define el comando de ejecución
CMD ["python", "app.py"]
