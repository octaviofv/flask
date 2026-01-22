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
    return "Servicio de OCR y Transcripción Activo"


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
    Endpoint para realizar OCR en una imagen o PDF en base64 con pre-procesamiento optimizado.
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
        
        # Obtener idioma 
        language = data.get('language', 'es')
        if not isinstance(language, str):
            return "Error: El campo 'language' debe ser un string", 400
        
        tesseract_lang = LANGUAGE_MAP.get(language, language)
        
        # Obtener modelo
        model_type = data.get('model', 'medium').lower()
        if model_type not in ['fast', 'medium', 'pro']:
            return "Error: El campo 'model' debe ser 'fast', 'medium' o 'pro'", 400
        
        tessdata_path = TESSDATA_PATHS.get(model_type)
        is_pdf = file_type == 'pdf'
        
        # --- PARÁMETROS OPCIONALES DE CONFIGURACIÓN ---
        # DPI: Default 300 (estándar). Útil subirlo para scans, mantener para digitales.
        dpi_val = int(data.get('dpi', 300))
        
        # PSM: Default 3 (Auto). Cambiar a 6 si es un bloque de texto uniforme.
        psm_val = int(data.get('psm', 3))
        
        # Binarización: Default False (mejor para digitales). True para scans sucios.
        do_binarize = data.get('binarize', False)
        
        # Resize: Default 1 (sin cambios). > 1 para ampliar imágenes pequeñas/pixeladas.
        resize_factor = float(data.get('resize_factor', 1))
        
        images_to_process = []
        
        # --- PROCESAMIENTO DE IMÁGENES ---
        if is_pdf:
            pdf_images = convert_from_bytes(file_bytes, dpi=dpi_val)
            for img in pdf_images:
                images_to_process.append(img)
        else:
            img = Image.open(BytesIO(file_bytes))
            images_to_process.append(img)
        
        all_texts = []

        # --- PRE-PROCESAMIENTO Y OCR ---
        for img in images_to_process:
            # 1. Convertir a Escala de Grises (Base para todo)
            img = img.convert('L')
            
            # 2. BINARIZACIÓN (Opcional)
            if do_binarize:
                # Umbral 160 es un buen estándar para limpiar ruido gris
                img = img.point(lambda p: p > 160 and 255)
            
            # 3. RE-ESCALADO (Opcional)
            if resize_factor > 1:
                new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # 4. Configuración Tesseract
            config = f'--tessdata-dir "{tessdata_path}" --oem 3 --psm {psm_val}'
            
            # Whitelist opcional (si el usuario quiere restringir caracteres)
            whitelist = data.get('tessedit_char_whitelist', '')
            if whitelist:
                config += f' -c tessedit_char_whitelist="{whitelist}"'
            
            text = pytesseract.image_to_string(img, lang=tesseract_lang, config=config)
            
            if text.strip():
                all_texts.append(text.strip())
        
        full_text = '\n\n--- Página ---\n\n'.join(all_texts)
        
        return full_text, 200, {'Content-Type': 'text/plain; charset=utf-8'}
        
    except base64.binascii.Error:
        return "Error: El archivo no está correctamente codificado en base64", 400
    except Exception as e:
        return f"Error procesando el archivo: {str(e)}", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
