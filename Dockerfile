# Usa la imagen base de Python
FROM python:3.9-slim

# Instala git antes de ejecutar pip install
RUN apt-get update && apt-get install -y git ffmpeg

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
