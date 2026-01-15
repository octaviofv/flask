# Usa la imagen base de Python
FROM python:3.9-slim

# Instala git y dependencias para Whisper y EasyOCR
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
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
