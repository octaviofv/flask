FROM python:3.6
MAINTAINER Octavio Flores
COPY . /app
WORKDIR /app
RUN pip install "git+https://github.com/openai/whisper.git" 
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]
