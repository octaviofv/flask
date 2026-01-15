from flask import Flask, request, jsonify
import whisper
import pytesseract
import os
import time
import json
import base64
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_bytes

app = Flask(__name__)

# Load Whisper model
model = whisper.load_model("base")

# Mapeo de códigos de idioma a códigos de Tesseract
LANGUAGE_MAP = {
    'es': 'spa',
    'en': 'eng',
    'fr': 'fra',
    'de': 'deu',
    'it': 'ita',
    'pt': 'por',
    'nl': 'nld',
    'pl': 'pol',
    'ru': 'rus',
    'ja': 'jpn',
    'zh': 'chi_sim',
    'ko': 'kor',
    'ar': 'ara'
}

# Rutas de los modelos de Tesseract
TESSDATA_PATHS = {
    'fast': '/usr/share/tesseract-ocr/5/tessdata',
    'medium': '/usr/share/tesseract-ocr/5/tessdata_standard',
    'pro': '/usr/share/tesseract-ocr/5/tessdata_best'
}

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
    - 'model': modelo de OCR a usar (default: 'medium'). Opciones: 'fast', 'medium', 'pro'
    
    Formatos soportados: PNG, JPEG, WebP, BMP, GIF, TIFF, PDF
    
    Ejemplo de request:
    {
        "base64": "JVBERi0xLjQKJ...",
        "type": "pdf",
        "language": "es",
        "model": "medium"
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
        
        # Convertir código de idioma a código de Tesseract
        tesseract_lang = LANGUAGE_MAP.get(language, language)
        
        # Obtener modelo (fast, medium o pro, default: medium)
        model_type = data.get('model', 'medium').lower()
        if model_type not in ['fast', 'medium', 'pro']:
            return "Error: El campo 'model' debe ser 'fast', 'medium' o 'pro'", 400
        
        # Configurar la ruta de tessdata según el modelo
        tessdata_path = TESSDATA_PATHS.get(model_type)
        
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
            # Realizar OCR con Tesseract usando el modelo seleccionado
            config = f'--tessdata-dir {tessdata_path}'
            text = pytesseract.image_to_string(img, lang=tesseract_lang, config=config)
            if text.strip():
                all_texts.append(text.strip())
        
        # Unir todo el texto en un solo string
        full_text = '\n'.join(all_texts)
        
        # Devolver solo el texto plano
        return full_text, 200, {'Content-Type': 'text/plain; charset=utf-8'}
        
    except base64.binascii.Error:
        return "Error: El archivo no está correctamente codificado en base64", 400
    except Exception as e:
        return f"Error procesando el archivo: {str(e)}", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True,host='0.0.0.0',port=port)
