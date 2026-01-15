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
from pdf2image import convert_from_bytes

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
    Endpoint para realizar OCR en una imagen o PDF en base64.
    
    Campos requeridos:
    - 'base64': string en base64 con el contenido del archivo
    - 'type': "pdf" o "image"
    
    Campos opcionales:
    - 'language': un idioma (default: 'es'). Ejemplos: 'es', 'en', 'fr', 'de'
    
    Formatos soportados: PNG, JPEG, WebP, BMP, GIF, TIFF, PDF
    
    Ejemplo de request:
    {
        "base64": "JVBERi0xLjQKJ...",
        "type": "pdf",
        "language": "es"
    }
    
    Retorna: texto plano con el contenido extraído
    """
    # Verificar que se envió JSON
    if not request.is_json:
        return "Error: Content-Type debe ser application/json", 400
    
    data = request.get_json()
    
    # Verificar campos requeridos
    if 'base64' not in data:
        return "Error: Falta el campo 'base64' con el archivo en base64", 400
    
    if 'type' not in data:
        return "Error: Falta el campo 'type'. Debe ser 'pdf' o 'image'", 400
    
    file_type = data['type'].lower()
    if file_type not in ['pdf', 'image']:
        return "Error: El campo 'type' debe ser 'pdf' o 'image'", 400
    
    try:
        file_base64 = data['base64']
        
        # Remover el prefijo data:...;base64, si existe
        if ',' in file_base64:
            file_base64 = file_base64.split(',')[1]
        
        # Decodificar el base64
        file_bytes = base64.b64decode(file_base64)
        
        # Obtener idioma (solo uno permitido)
        language = data.get('language', 'es')
        
        # Validar que sea un string
        if not isinstance(language, str):
            return "Error: El campo 'language' debe ser un string con un solo idioma", 400
        
        # Si es español, usar el reader global, sino crear uno nuevo
        if language == 'es':
            reader = ocr_reader
        else:
            reader = easyocr.Reader([language], gpu=False)
        
        # Determinar si es PDF según el type indicado
        is_pdf = file_type == 'pdf'
        
        images_to_process = []
        
        if is_pdf:
            # Convertir PDF a imágenes (una por página)
            pdf_images = convert_from_bytes(file_bytes, dpi=200)
            
            for pdf_image in pdf_images:
                if pdf_image.mode != 'RGB':
                    pdf_image = pdf_image.convert('RGB')
                images_to_process.append(pdf_image)
        else:
            # Es una imagen normal
            image = Image.open(BytesIO(file_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            images_to_process.append(image)
        
        # Procesar todas las imágenes (una si es imagen, múltiples si es PDF)
        all_texts = []
        
        for img in images_to_process:
            image_np = np.array(img)
            
            # Realizar OCR
            results = reader.readtext(image_np)
            
            for (bbox, text, confidence) in results:
                all_texts.append(text)
        
        # Unir todo el texto en un solo string
        full_text = ' '.join(all_texts)
        
        # Devolver solo el texto plano
        return full_text, 200, {'Content-Type': 'text/plain; charset=utf-8'}
        
    except base64.binascii.Error:
        return "Error: El archivo no está correctamente codificado en base64", 400
    except Exception as e:
        return f"Error procesando el archivo: {str(e)}", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True,host='0.0.0.0',port=port)
