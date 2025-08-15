# Usa la imagen base de Python
FROM python:3.9-slim

# Instala git, ffmpeg y stunnel
RUN apt-get update && apt-get install -y git ffmpeg stunnel4 && apt-get clean

# Configura el directorio de trabajo
WORKDIR /app

# Copiar el código fuente
COPY . .

# Copia los certificados para stunnel
COPY cert.pem /etc/stunnel/cert.pem
COPY key.pem /etc/stunnel/key.pem
COPY flask_stunnel.conf /etc/stunnel/flask.conf

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Puerto HTTP interno de Flask y HTTPS externo de stunnel
EXPOSE 5000 443

# Comando para ejecutar Flask y stunnel juntos
CMD ["sh", "-c", "stunnel /etc/stunnel/flask.conf & python app.py"]
