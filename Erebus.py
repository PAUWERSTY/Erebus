# main.py (Modificaciones para devolver JSON con Texto y Audio Base64)

import os
import re
import time
import asyncio
import whisper
import google.generativeai as genai
from dotenv import load_dotenv
import edge_tts
# Quita playsound
# Quita sounddevice, keyboard, scipy
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
# Cambia FileResponse por JSONResponse
from fastapi.responses import JSONResponse
# Añade CORS Middleware
from fastapi.middleware.cors import CORSMiddleware
import uuid
import shutil
import base64 # Para codificar el audio

# --- Carga de Configuraciones ---
load_dotenv()
# ... (GEMINI_API_KEY, ASSISTANT_NAME, GREETING_RESPONSE, etc.) ...
# ... (SYSTEM_INSTRUCTION) ...
WHISPER_MODEL_NAME = "base"
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
EDGE_VOICE = "es-MX-DaliaNeural"
EDGE_RATE = '+5%'
EDGE_VOLUME = '+0%'
LANGUAGE_CODE = "es"
TEMP_DIR = "temp_audio"

os.makedirs(TEMP_DIR, exist_ok=True)

# --- Configuración API Gemini ---
# ... (try...except genai.configure) ...

# --- Inicializar FastAPI ---
app = FastAPI()

# --- Configurar CORS ---
# Ajusta origins según sea necesario para producción
origins = [
    "http://localhost:5173", # Origen común para Vite dev server
    "http://localhost:3000", # Origen común para create-react-app
    # Añade la URL de tu frontend desplegado aquí
    # "https://tu-frontend.up.railway.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"], # Permite todas las cabeceras
)


# --- Funciones Auxiliares (Sin cambios) ---
def clean_text_for_comparison(text: str) -> str:
    # ... (igual) ...
    if not text: return ""
    cleaned = text.lower()
    cleaned = re.sub(r"^[.,!?;:]+|[.,!?;:]+$", "", cleaned).strip()
    return cleaned

def clean_text_for_speech(text: str) -> str:
    # ... (igual) ...
    if not text: return ""
    cleaned = text.replace('\n', ' ').replace('\t', ' ').replace('*', '')
    cleaned = re.sub(r'--+', ' ', cleaned)
    cleaned = cleaned.replace('**Importante:**', 'Importante:')
    cleaned = re.sub(r'(?<=[A-Z])\.(?=[A-Z])', '', cleaned)
    cleaned = re.sub(r'\.\s+(?=[A-Z])', ' ', cleaned)
    cleaned = ' '.join(cleaned.split())
    return cleaned

# --- Funciones de Lógica Principal (Sin cambios en su lógica interna) ---
def transcribe_audio_sync(file_path: str, model_name: str = WHISPER_MODEL_NAME) -> str | None:
    # ... (igual que antes) ...
    print(f"🎧 Cargando modelo Whisper ('{model_name}')...")
    if not os.path.exists(file_path): return None
    try:
        model = whisper.load_model(model_name)
        print(f"   Transcribiendo '{file_path}'...")
        result = model.transcribe(file_path, fp16=False, language=LANGUAGE_CODE)
        transcribed_text = result["text"].strip()
        print("\n📝 Transcripción:", transcribed_text or "(Vacía)")
        return transcribed_text if transcribed_text else None
    except Exception as e:
        print(f"\n❌ Error en transcripción Whisper: {e}")
        return None

async def generate_speech_edge_tts(text_to_speak: str, voice: str = EDGE_VOICE, rate: str = EDGE_RATE, volume: str = EDGE_VOLUME) -> str | None:
    # ... (igual que antes, devuelve ruta del archivo) ...
    if not text_to_speak or not text_to_speak.strip(): return None
    text_for_speech = clean_text_for_speech(text_to_speak)
    if not text_for_speech: return None
    print(f"🔊 Generando audio con edge-tts (Voz: {voice})...")
    output_filename = os.path.join(TEMP_DIR, f"response_{uuid.uuid4()}.mp3")
    try:
        communicate = edge_tts.Communicate(text_for_speech, voice, rate=rate, volume=volume)
        await communicate.save(output_filename)
        print(f"✅ Audio generado: '{output_filename}'")
        return output_filename
    except Exception as e:
        print(f"❌ Error durante la generación con edge-tts: {e}")
        return None

def ask_gemini_sync(prompt: str, model_name: str = GEMINI_MODEL_NAME) -> str | None:
    # ... (igual que antes, devuelve texto) ...
     if not prompt or not prompt.strip(): return None
     print(f"\n🤖 Enviando prompt a Gemini ({model_name})...")
     try:
        model = genai.GenerativeModel(model_name=model_name, system_instruction=SYSTEM_INSTRUCTION)
        response = model.generate_content(prompt)
        gemini_response_text = response.text
        print("\n✨ Respuesta de Gemini:")
        print(gemini_response_text)
        return gemini_response_text
     except Exception as e:
        print(f"\n❌ Error con Gemini: {e}")
        return None

# --- Tarea de Limpieza (Sin cambios) ---
def remove_file(path: str):
    # ... (igual) ...
    try:
        if os.path.exists(path):
            os.remove(path)
            # print(f"🧹 Archivo temporal eliminado: {path}") # Opcional
    except Exception as e:
        print(f"⚠️ No se pudo eliminar {path}: {e}")

# --- Endpoint de la API (MODIFICADO) ---
# Quita response_class=FileResponse
@app.post("/process_audio/")
async def process_audio_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Recibe audio, procesa y devuelve JSON con texto y audio base64."""

    input_filename = os.path.join(TEMP_DIR, f"input_{uuid.uuid4()}{os.path.splitext(file.filename)[1] or '.wav'}")
    output_audio_path: str | None = None # Para asegurar que se pueda limpiar

    try:
        # 1. Guardar archivo subido
        try:
            with open(input_filename, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print(f"📥 Archivo guardado como: {input_filename}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error guardando archivo: {e}")
        finally:
             await file.close()
        background_tasks.add_task(remove_file, input_filename) # Limpiar siempre

        # 2. Transcribir
        loop = asyncio.get_event_loop()
        transcribed_text = await loop.run_in_executor(None, transcribe_audio_sync, input_filename, WHISPER_MODEL_NAME)
        if not transcribed_text:
            raise HTTPException(status_code=400, detail="Transcripción fallida o vacía.")

        # 3. Comprobar saludo
        cleaned_prompt = clean_text_for_comparison(transcribed_text)
        is_greeting = cleaned_prompt in SIMPLE_GREETINGS # SIMPLE_GREETINGS debe estar definido

        response_text: str | None = None
        if is_greeting:
            response_text = GREETING_RESPONSE # GREETING_RESPONSE debe estar definido
            print("🗣️  Saludo detectado.")
        else:
            # 4. Consultar a Gemini
            response_text = await loop.run_in_executor(None, ask_gemini_sync, transcribed_text, GEMINI_MODEL_NAME)

        if not response_text:
            # Si Gemini no respondió o fue un rechazo sin texto explícito
            print("🤔 No se obtuvo respuesta de texto de Gemini o fue un saludo.")
            # Puedes decidir qué hacer: devolver error o un mensaje estándar
            # Aquí generaremos audio de un mensaje genérico si no hay texto
            response_text = "No pude procesar esa solicitud o no tengo una respuesta específica."

        # 5. Generar audio de respuesta
        output_audio_path = await generate_speech_edge_tts(response_text)

        if not output_audio_path:
            raise HTTPException(status_code=500, detail="No se pudo generar el audio de respuesta.")

        # 6. Leer el audio generado y codificar en Base64
        try:
            with open(output_audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            print(f"🔊 Audio codificado en Base64 (tamaño: {len(audio_base64)} caracteres)")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error leyendo/codificando audio generado: {e}")
        finally:
             # Limpiar el archivo de audio generado DESPUÉS de leerlo
             if output_audio_path:
                 background_tasks.add_task(remove_file, output_audio_path)


        # 7. Devolver JSON con texto y audio base64
        return JSONResponse(content={
            "responseText": response_text,
            "audioBase64": audio_base64
        })

    except HTTPException as http_exc:
        # Re-lanzar excepciones HTTP para que FastAPI las maneje
        raise http_exc
    except Exception as e:
        # Capturar otros errores inesperados
        print(f"💥 Error inesperado en el endpoint: {e}")
        # Limpiar archivos si es posible en caso de error general
        if 'input_filename' in locals() and os.path.exists(input_filename):
             background_tasks.add_task(remove_file, input_filename)
        if output_audio_path and os.path.exists(output_audio_path):
             background_tasks.add_task(remove_file, output_audio_path)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")


# --- Punto de entrada (Sin cambios) ---
if __name__ == "__main__":
    import uvicorn
    # ... (igual) ...
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False) # Escucha en todas las interfaces