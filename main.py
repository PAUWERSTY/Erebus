
import os
import re
import time
import asyncio
import whisper # Librería de transcripción
import google.generativeai as genai
from dotenv import load_dotenv
import edge_tts # Librería TTS
import numpy as np # Dependencia de Whisper
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import shutil
import base64

# --- Carga de Configuraciones ---
load_dotenv() # Carga .env para desarrollo local

# --- Constantes y Configuraciones Optimizadas ---
# !! OPTIMIZACIÓN DE MEMORIA: Usar 'tiny' !!
# Opciones: "tiny", "base", "small". 'tiny' usa <1GB RAM, 'base' ~1GB, 'small' ~2GB.
# Considera la precisión vs memoria disponible. 'base' es un compromiso razonable si 'tiny' no basta.
WHISPER_MODEL_NAME = "tiny"
# ----------------------------------------------
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # Modelo rápido de Gemini
EDGE_VOICE = "es-MX-DaliaNeural" # Voz TTS (puedes cambiarla)
EDGE_RATE = '+5%'
EDGE_VOLUME = '+0%'
LANGUAGE_CODE = "es"
TEMP_DIR = "temp_audio"
ASSISTANT_VERSION = "0.2"
POWERED_BY = "Cynosure"

# --- Nombre y Definición del Asistente ---
ASSISTANT_NAME = "E.R.E.B.U.S."
ASSISTANT_DEFINITION = "Entorno de Respuesta Estructurada Basado en Unificación Semántica"
GREETING_RESPONSE = f"Hola, soy {ASSISTANT_NAME}, {ASSISTANT_DEFINITION}. Mi especialización es sobre fármacos y padecimientos comunes. ¿En qué puedo ayudarte?"

# --- Lista de Saludos Simples ---
SIMPLE_GREETINGS = [
    'hola', 'buenos días', 'buenas tardes', 'buenas noches', 'que tal',
    'hola erebus', 'hola erébus'
]

# --- Instrucción de Sistema para Gemini ---
SYSTEM_INSTRUCTION = """
Eres un asistente virtual llamado E.R.E.B.U.S. (...) con un enfoque MUY específico: (...) fármacos y padecimientos comunes. (...) Tu rol NO es ser un chatbot de conocimiento general.

Instrucciones Estrictas:
1.  **Evaluación del Tema:** (...) Determina si trata PRINCIPALMENTE sobre fármacos, (...) condiciones médicas comunes.
2.  **Respuesta Médica (Si Aplica):** (...) proporciona una respuesta informativa y general. (...) **AL FINAL (...) DEBES incluir SIEMPRE la siguiente advertencia (...):**
    ---
    **Importante:** Soy un modelo de lenguaje (...) Consulta siempre a tu médico (...)
    ---
3.  **Rechazo de Otros Temas (Si Aplica):** Si el tema NO es claramente médico (...) DEBES RECHAZAR responder (...). Tu ÚNICA respuesta (...) debe ser (...) similar a esta (...):
    'Mi especialización es únicamente sobre fármacos y padecimientos comunes. No puedo procesar solicitudes sobre otros temas.'
4.  **No Mezclar:** (...) Limítate estrictamente a las opciones 2 o 3 (...). NO respondas a saludos simples (...).
""" # Nota: Instrucción abreviada aquí por brevedad, usa la completa.

# --- Crear directorio temporal ---
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Configuración API Gemini ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# (El resto de la configuración de Gemini es igual)
if not GEMINI_API_KEY:
    print("CRITICAL ERROR: No se encontró la variable de entorno GEMINI_API_KEY.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ API de Gemini configurada.")
    except Exception as e:
        print(f"❌ Error config GenAI: {e}")


# --- Inicializar FastAPI ---
app = FastAPI(
    title=f"{ASSISTANT_NAME} API",
    version=ASSISTANT_VERSION,
    description=f"API para {ASSISTANT_NAME} - v{ASSISTANT_VERSION} powered by {POWERED_BY}"
)

# --- Configurar CORS (Igual) ---
origins = [
    "http://localhost:5173", "http://localhost:3000",
    # Añade URL de tu frontend desplegado
]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- !! OPTIMIZACIÓN: Variable Global para Modelo Whisper !! ---
whisper_model = None

# --- !! OPTIMIZACIÓN: Evento de Inicio de FastAPI para cargar modelo !! ---
@app.on_event("startup")
async def load_whisper_model_on_startup():
    """Carga el modelo Whisper una vez al iniciar la aplicación."""
    global whisper_model
    print(f"--- Iniciando Carga del Modelo Whisper ('{WHISPER_MODEL_NAME}') ---")
    # Asegúrate de tener suficiente memoria para el modelo elegido en tu plan de Railway
    if WHISPER_MODEL_NAME not in ["tiny", "base", "small", "medium", "large"]:
         print(f"❌ Nombre de modelo Whisper inválido: {WHISPER_MODEL_NAME}. Usando 'tiny' por defecto.")
         effective_model_name = "tiny"
    else:
         effective_model_name = WHISPER_MODEL_NAME

    try:
        whisper_model = whisper.load_model(effective_model_name)
        print(f"✅ Modelo Whisper ('{effective_model_name}') cargado en memoria.")
    except Exception as e:
        print(f"❌❌ CRITICAL ERROR: Fallo al cargar modelo Whisper '{effective_model_name}': {e}")
        print("   La funcionalidad de transcripción estará DESACTIVADA.")
        print("   Causas posibles: Memoria insuficiente en el plan de Railway,")
        print("   error de descarga del modelo, dependencia 'torch' faltante (improbable con pip).")
        whisper_model = None # Asegurar que es None si falla la carga

# --- Funciones Auxiliares (Igual) ---
def clean_text_for_comparison(text: str) -> str:
    if not text: return ""
    cleaned = text.lower()
    cleaned = re.sub(r"^[.,!?;:]+|[.,!?;:]+$", "", cleaned).strip()
    return cleaned

def clean_text_for_speech(text: str) -> str:
    if not text: return ""
    cleaned = text.replace('\n', ' ').replace('\t', ' ').replace('*', '')
    cleaned = re.sub(r'--+', ' ', cleaned)
    cleaned = cleaned.replace('**Importante:**', 'Importante:')
    cleaned = re.sub(r'(?<=[A-Z])\.(?=[A-Z])', '', cleaned)
    cleaned = re.sub(r'\.\s+(?=[A-Z])', ' ', cleaned)
    cleaned = ' '.join(cleaned.split())
    return cleaned

# --- Funciones de Lógica Principal ---
# !! OPTIMIZACIÓN: Usa el modelo global, no carga cada vez !!
def transcribe_audio_sync(file_path: str) -> str | None:
    """Transcribe usando el modelo Whisper cargado globalmente."""
    global whisper_model
    if whisper_model is None:
        print("❌ Error: Modelo Whisper no disponible (falló la carga inicial).")
        return None # No se puede transcribir

    # El modelo ya está cargado, solo transcribir
    print(f"🎧 Transcribiendo '{file_path}' con modelo '{WHISPER_MODEL_NAME}' cargado...")
    if not os.path.exists(file_path):
        print(f"❌ Error: Archivo no encontrado para transcribir: {file_path}")
        return None
    try:
        # fp16=False es más compatible con CPU
        result = whisper_model.transcribe(file_path, fp16=False, language=LANGUAGE_CODE)
        transcribed_text = result["text"].strip()
        print("\n📝 Transcripción:", transcribed_text or "(Vacía)")
        print("-" * 30)
        return transcribed_text if transcribed_text else None
    except Exception as e:
        print(f"\n❌ Error durante la transcripción: {e}")
        print("   Asegúrate que 'ffmpeg' esté instalado en el entorno del servidor.")
        print("-" * 30)
        return None

async def generate_speech_edge_tts(text_to_speak: str, voice: str = EDGE_VOICE, rate: str = EDGE_RATE, volume: str = EDGE_VOLUME) -> str | None:
    # (Lógica interna igual que antes)
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
        print(f"❌ Error generando audio con edge-tts: {e}")
        return None

def ask_gemini_sync(prompt: str, model_name: str = GEMINI_MODEL_NAME) -> str | None:
    # (Lógica interna igual que antes)
    if not prompt or not prompt.strip(): return None
    if not GEMINI_API_KEY: return "Error: Configuración IA incompleta."
    print(f"\n🤖 Enviando prompt a Gemini ({model_name})...")
    try:
        model = genai.GenerativeModel(model_name=model_name, system_instruction=SYSTEM_INSTRUCTION)
        response = model.generate_content(prompt)
        gemini_response_text = getattr(response, 'text', None)
        if gemini_response_text:
             print("\n✨ Respuesta Gemini:", gemini_response_text)
        else:
             print("\n🤔 Gemini no devolvió texto.")
             gemini_response_text = "No se obtuvo respuesta de texto."
        print("-" * 30)
        return gemini_response_text
    except Exception as e:
        print(f"\n❌ Error con Gemini: {e}")
        print("-" * 30)
        return "Error al contactar la IA."

# --- Tarea de Limpieza (Igual) ---
def remove_file(path: str):
    try:
        if os.path.exists(path): os.remove(path)
    except Exception as e: print(f"⚠️ No se pudo eliminar {path}: {e}")

# --- Endpoint Raíz (Igual) ---
@app.get("/", tags=["Info"])
async def read_root():
    return {
        "message": f"API de {ASSISTANT_NAME}", "version": ASSISTANT_VERSION,
        "powered_by": POWERED_BY, "docs_url": "/docs", "process_endpoint": "/process_audio/"
    }

# --- Endpoint Principal ---
@app.post("/process_audio/", tags=["Audio Processing"])
async def process_audio_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    input_filename = os.path.join(TEMP_DIR, f"input_{uuid.uuid4()}{file_extension}")
    output_audio_path: str | None = None

    try:
        # 1. Guardar archivo
        try:
            with open(input_filename, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
            print(f"📥 Archivo guardado: {input_filename}")
        except Exception as e: raise HTTPException(status_code=500, detail=f"Error guardando: {e}")
        finally: await file.close()
        background_tasks.add_task(remove_file, input_filename) # Limpiar entrada

        # 2. Transcribir (Usa modelo global)
        loop = asyncio.get_event_loop()
        transcribed_text = await loop.run_in_executor(None, transcribe_audio_sync, input_filename)
        if transcribed_text is None: # Verificar explícitamente None por si falla o está vacío
             # Decidir si un audio vacío es un error 400 o simplemente no necesita respuesta
             print("⚠️ Transcripción fallida o vacía.")
             raise HTTPException(status_code=400, detail="No se pudo transcribir el audio o estaba vacío.")

        # 3. Comprobar Saludo
        cleaned_prompt = clean_text_for_comparison(transcribed_text)
        is_greeting = cleaned_prompt in SIMPLE_GREETINGS

        response_text: str | None = None
        if is_greeting:
            response_text = GREETING_RESPONSE
            print("🗣️  Saludo detectado.")
        else:
            # 4. Consultar a Gemini
            response_text = await loop.run_in_executor(None, ask_gemini_sync, transcribed_text, GEMINI_MODEL_NAME)

        if not response_text:
            response_text = "No se generó respuesta de texto." # Mensaje por defecto

        # 5. Generar Audio
        output_audio_path = await generate_speech_edge_tts(response_text)

        audio_base64: str | None = None
        if output_audio_path:
            # 6. Codificar Audio
            try:
                with open(output_audio_path, "rb") as audio_file: audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                print(f"🔊 Audio codificado (Base64: {len(audio_base64)} chars)")
            except Exception as e: print(f"❌ Error leyendo/codificando: {e}")
            finally: background_tasks.add_task(remove_file, output_audio_path) # Limpiar salida
        else:
            print("⚠️ No se generó audio de respuesta.")

        # 7. Devolver JSON
        return JSONResponse(content={
            "responseText": response_text,
            "audioBase64": audio_base64
        })

    except HTTPException as http_exc:
        print(f"HTTP Exc: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        print(f"💥 Error Inesperado: {e}")
        # Limpieza de emergencia (evita dejar archivos huérfanos)
        if 'input_filename' in locals() and os.path.exists(input_filename): background_tasks.add_task(remove_file, input_filename)
        if output_audio_path and os.path.exists(output_audio_path): background_tasks.add_task(remove_file, output_audio_path)
        raise HTTPException(status_code=500, detail="Error interno del servidor.")

# --- Punto de entrada (para desarrollo local) ---
if __name__ == "__main__":
    import uvicorn
    print("-" * 50)
    print(f"🚀 Iniciando E.R.E.B.U.S. API v{ASSISTANT_VERSION} (Optimizado RAM) - Powered by {POWERED_BY}")
    print(f"   Whisper Model Pre-loading: '{WHISPER_MODEL_NAME}'")
    print("   Ejecutando servidor FastAPI localmente...")
    print(f"   Docs (Swagger): http://127.0.0.1:8000/docs")
    print(f"   Docs (ReDoc):   http://127.0.0.1:8000/redoc")
    print(f"   Endpoint POST:  http://127.0.0.1:8000/process_audio/")
    print("-" * 50)
    # Uvicorn iniciará FastAPI y disparará el evento @app.on_event("startup")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
