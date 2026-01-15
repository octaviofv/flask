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
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio para modelos estándar (medium) y descargarlos
RUN mkdir -p /usr/share/tesseract-ocr/5/tessdata_standard \
    && wget -q -O /usr/share/tesseract-ocr/5/tessdata_standard/spa.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/spa.traineddata \
    && wget -q -O /usr/share/tesseract-ocr/5/tessdata_standard/eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata \
    && wget -q -O /usr/share/tesseract-ocr/5/tessdata_standard/fra.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata \
    && wget -q -O /usr/share/tesseract-ocr/5/tessdata_standard/deu.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/deu.traineddata \
    && wget -q -O /usr/share/tesseract-ocr/5/tessdata_standard/ita.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/ita.traineddata \
    && wget -q -O /usr/share/tesseract-ocr/5/tessdata_standard/por.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/por.traineddata

# Crear directorio para modelos best (pro) y descargarlos
RUN mkdir -p /usr/share/tesseract-ocr/5/tessdata_best \
    && wget -q -O /usr/share/tesseract-ocr/5/tessdata_best/spa.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/spa.traineddata \
    && wget -q -O /usr/share/tesseract-ocr/5/tessdata_best/eng.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata \
    && wget -q -O /usr/share/tesseract-ocr/5/tessdata_best/fra.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/fra.traineddata \
    && wget -q -O /usr/share/tesseract-ocr/5/tessdata_best/deu.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/deu.traineddata \
    && wget -q -O /usr/share/tesseract-ocr/5/tessdata_best/ita.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/ita.traineddata \
    && wget -q -O /usr/share/tesseract-ocr/5/tessdata_best/por.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/por.traineddata

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
