from flask import Flask, request, jsonify
import whisper
import os
import time

app = Flask(__name__)

# Carga del modelo Whisper
model = whisper.load_model("base")

@app.route("/")
def hello():
    return "Servidor Flask corriendo vía HTTP interno (TLSv1 externo vía stunnel)"

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    start = time.time()
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Guardar el archivo temporalmente
        filepath = os.path.join("/tmp", file.filename)
        file.save(filepath)

        # Obtener tamaño del archivo
        file_size = os.path.getsize(filepath)

        transcribe_start = time.time()
        # Procesar archivo con Whisper
        result = model.transcribe(filepath)
        transcribe_end = time.time()
        os.remove(filepath)

        result = {
            "transcription": result["text"],
            "stats": {
                "total_processing_time": round(transcribe_end - transcribe_start, 2),
                "words_per_second": round(len(result["text"]) / (transcribe_end - transcribe_start), 2),
                "file_size_in_bytes": file_size
            },
            "filename": file.filename,
        }

        return jsonify(result)

    return jsonify({"error": "Invalid request"}), 400

if __name__ == "__main__":
    # Escucha en todas las interfaces internas del contenedor
    port = 5000
    app.run(debug=True, host='0.0.0.0', port=port)
