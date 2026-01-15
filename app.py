from flask import Flask, request, jsonify
import whisper
import easyocr
import os
import time
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load Whisper model
model = whisper.load_model("base")

# Load EasyOCR reader (español e inglés por defecto)
ocr_reader = easyocr.Reader(['es', 'en'], gpu=False)

@app.route("/")
def hello():
    return "asdddddd"


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    start = time.time()
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save the file temporarily
        filepath = os.path.join("/tmp", file.filename)
        file.save(filepath)

        # Get the size of the file
        file_size = os.path.getsize(filepath)

        transcribe_start = time.time()
        # Process the file with Whisper
        result = model.transcribe(filepath)
        
        transcribe_end = time.time()
        os.remove(filepath)  # Remove the file after processing

        result = {
            "transcription": result["text"],
            "stats": {
                "total_processing_time": transcribe_end - transcribe_start,
                "words_per_second": round(len(result["text"]) / (transcribe_end - transcribe_start), 2),
                "file_size_in_bytes": file_size
            },
            "filename": file.filename,
        }

        return jsonify(result)

    return jsonify({"error": "Invalid request"}), 400


@app.route('/ocr', methods=['POST'])
def ocr_image():
    """
    Endpoint para realizar OCR en una imagen en base64.
    
    Espera un JSON con el campo 'image' conteniendo la imagen en base64.
    Opcionalmente puede incluir 'languages' como lista de idiomas (default: ['es', 'en'])
    
    Ejemplo de request:
    {
        "image": "base64_encoded_image_string",
        "languages": ["es", "en"]  // opcional
    }
    """
    start = time.time()
    
    # Verificar que se envió JSON
    if not request.is_json:
        return jsonify({"error": "Content-Type debe ser application/json"}), 400
    
    data = request.get_json()
    
    # Verificar que se envió la imagen
    if 'image' not in data:
        return jsonify({"error": "Falta el campo 'image' con la imagen en base64"}), 400
    
    try:
        image_base64 = data['image']
        
        # Remover el prefijo data:image/...;base64, si existe
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decodificar la imagen base64
        image_bytes = base64.b64decode(image_base64)
        
        # Convertir bytes a imagen PIL
        image = Image.open(BytesIO(image_bytes))
        
        # Convertir a RGB si es necesario (por si viene en RGBA o escala de grises)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convertir a numpy array para EasyOCR
        image_np = np.array(image)
        
        # Obtener idiomas personalizados si se proporcionaron
        languages = data.get('languages', ['es', 'en'])
        
        # Si se usan idiomas diferentes a los del reader global, crear uno nuevo
        # Para optimización, usamos el reader global si los idiomas coinciden
        if set(languages) == {'es', 'en'}:
            reader = ocr_reader
        else:
            reader = easyocr.Reader(languages, gpu=False)
        
        ocr_start = time.time()
        
        # Realizar OCR
        results = reader.readtext(image_np)
        
        ocr_end = time.time()
        
        # Extraer solo el texto de los resultados
        # EasyOCR devuelve: [(bbox, text, confidence), ...]
        texts = []
        detailed_results = []
        
        for (bbox, text, confidence) in results:
            texts.append(text)
            detailed_results.append({
                "text": text,
                "confidence": round(confidence, 4),
                "bounding_box": bbox
            })
        
        # Unir todo el texto en un solo string
        full_text = ' '.join(texts)
        
        response = {
            "text": full_text,
            "detailed_results": detailed_results,
            "stats": {
                "total_processing_time": round(ocr_end - start, 4),
                "ocr_time": round(ocr_end - ocr_start, 4),
                "image_size": {
                    "width": image.width,
                    "height": image.height
                },
                "detections_count": len(results),
                "languages_used": languages
            }
        }
        
        return jsonify(response)
        
    except base64.binascii.Error:
        return jsonify({"error": "La imagen no está correctamente codificada en base64"}), 400
    except Exception as e:
        return jsonify({"error": f"Error procesando la imagen: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True,host='0.0.0.0',port=port)
