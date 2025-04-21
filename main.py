
import os
import re
import time
import asyncio
import whisper
import google.generativeai as genai
from dotenv import load_dotenv
import edge_tts
# numpy es necesario para Whisper
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import shutil
import base64

# --- Carga de Configuraciones ---
load_dotenv() # Carga .env para desarrollo local

# --- Constantes y Configuraciones ---
WHISPER_MODEL_NAME = "base" # Opciones: "tiny", "base", "small", "medium", "large"
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # Optimizado para velocidad
EDGE_VOICE = "es-MX-DaliaNeural" # Voz TTS de Edge (elige tu favorita)
EDGE_RATE = '+5%'
EDGE_VOLUME = '+0%'
LANGUAGE_CODE = "es" # Idioma para Whisper y TTS
TEMP_DIR = "temp_audio" # Directorio para archivos temporales
ASSISTANT_VERSION = "0.2" # Versión para branding
POWERED_BY = "Cynosure" # Branding

# --- Nombre y Definición del Asistente ---
ASSISTANT_NAME = "E.R.E.B.U.S."
ASSISTANT_DEFINITION = "Entorno de Respuesta Estructurada Basado en Unificación Semántica"
GREETING_RESPONSE = f"Hola, soy {ASSISTANT_NAME}, {ASSISTANT_DEFINITION}. Mi especialización es sobre fármacos y padecimientos comunes. ¿En qué puedo ayudarte dentro de este ámbito?"

# --- Lista de Saludos Simples (en minúsculas) ---
SIMPLE_GREETINGS = [
    'hola', 'buenos días', 'buenas tardes', 'buenas noches', 'que tal',
    'hola erebus', 'hola erébus'
]

# --- Instrucción de Sistema para Gemini (Estricta) ---
SYSTEM_INSTRUCTION = """
Eres un asistente virtual llamado E.R.E.B.U.S. (Entorno de Respuesta Estructurada Basado en Unificación Semántica) con un enfoque MUY específico: proporcionar información general sobre fármacos y padecimientos comunes, basándote en conocimiento público. Tu rol NO es ser un chatbot de conocimiento general.

Instrucciones Estrictas:
1.  **Evaluación del Tema:** Analiza CUIDADOSAMENTE el texto del usuario (el prompt). Determina si trata PRINCIPALMENTE sobre fármacos, medicamentos, enfermedades, síntomas o condiciones médicas comunes.
2.  **Respuesta Médica (Si Aplica):** Si el tema es claramente médico según el punto 1, proporciona una respuesta informativa y general. **AL FINAL de ESTA respuesta médica, DEBES incluir SIEMPRE la siguiente advertencia textual sin modificarla:**
    ---
    **Importante:** Soy un modelo de lenguaje de inteligencia artificial y no un profesional médico. La información proporcionada es de carácter general y no debe considerarse un sustituto del consejo, diagnóstico o tratamiento médico profesional. Consulta siempre a tu médico o a un farmacéutico cualificado si tienes preguntas sobre una condición médica o un medicamento. No ignores el consejo médico profesional ni demores en buscarlo por algo que hayas leído aquí.
    ---
3.  **Rechazo de Otros Temas (Si Aplica):** Si el tema NO es claramente médico (ej: historia, política, tecnología, geografía, opiniones, conversaciones casuales, temas personales NO médicos, etc.), DEBES RECHAZAR responder la pregunta. Tu ÚNICA respuesta en este caso debe ser una frase MUY similar a esta, sin añadir nada más (ni siquiera la advertencia médica):
    'Mi especialización es únicamente sobre fármacos y padecimientos comunes. No puedo procesar solicitudes sobre otros temas.'
4.  **No Mezclar:** No intentes responder parcialmente a temas no médicos ni entablar conversaciones fuera del ámbito médico especificado. Limítate estrictamente a las opciones 2 o 3 según el tema detectado. NO respondas a saludos simples, eso se maneja externamente.
"""

# --- Crear directorio temporal ---
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Configuración API Gemini ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("CRITICAL ERROR: No se encontró la variable de entorno GEMINI_API_KEY.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ API de Gemini configurada correctamente.")
    except Exception as e:
        print(f"❌ Error al configurar la API de Google GenAI: {e}")

# --- Inicializar FastAPI ---
app = FastAPI(
    title=f"{ASSISTANT_NAME} API",
    version=ASSISTANT_VERSION,
    description=f"API para procesar audio con {ASSISTANT_NAME} ({ASSISTANT_DEFINITION}) - v{ASSISTANT_VERSION} powered by {POWERED_BY}"
)

# --- Configurar CORS ---
origins = [
    "http://localhost:5173", # Vite dev
    "http://localhost:3000", # CRA dev
    # Añade aquí la URL de tu frontend cuando lo despliegues
    # "https://<tu-frontend>.vercel.app",
    # "https://<tu-frontend>.netlify.app",
    # "https://<tu-frontend>.up.railway.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Funciones Auxiliares ---
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
def transcribe_audio_sync(file_path: str, model_name: str = WHISPER_MODEL_NAME) -> str | None:
    print(f"🎧 Cargando modelo Whisper ('{model_name}')...")
    if not os.path.exists(file_path):
        print(f"❌ Error: Archivo no encontrado: {file_path}")
        return None
    try:
        model = whisper.load_model(model_name)
        print(f"   Transcribiendo '{file_path}' (Idioma: {LANGUAGE_CODE})...")
        result = model.transcribe(file_path, fp16=False, language=LANGUAGE_CODE)
        transcribed_text = result["text"].strip()
        print("\n📝 Transcripción:", transcribed_text or "(Vacía)")
        print("-" * 30)
        return transcribed_text if transcribed_text else None
    except Exception as e:
        print(f"\n❌ Error en transcripción Whisper: {e}")
        print("-" * 30)
        return None

async def generate_speech_edge_tts(text_to_speak: str, voice: str = EDGE_VOICE, rate: str = EDGE_RATE, volume: str = EDGE_VOLUME) -> str | None:
    if not text_to_speak or not text_to_speak.strip():
        print("🔊 No hay texto válido para generar audio.")
        return None
    text_for_speech = clean_text_for_speech(text_to_speak)
    if not text_for_speech:
         print("🔊 El texto quedó vacío después de la limpieza.")
         return None
    print(f"🔊 Generando audio con edge-tts (Voz: {voice})...")
    output_filename = os.path.join(TEMP_DIR, f"response_{uuid.uuid4()}.mp3")
    try:
        communicate = edge_tts.Communicate(text_for_speech, voice, rate=rate, volume=volume)
        await communicate.save(output_filename)
        print(f"✅ Audio generado: '{output_filename}'")
        return output_filename
    except edge_tts.NoAudioGenerated:
        print(f"❌ Error: edge-tts no generó audio.")
        return None
    except Exception as e:
        print(f"❌ Error durante la generación con edge-tts: {e}")
        return None

def ask_gemini_sync(prompt: str, model_name: str = GEMINI_MODEL_NAME) -> str | None:
     if not prompt or not prompt.strip():
         print("⚠️ Prompt vacío para Gemini.")
         return None
     if not GEMINI_API_KEY:
         print("❌ No se puede llamar a Gemini, la API Key no está configurada.")
         return "Error: La configuración de la IA no está completa."

     print(f"\n🤖 Enviando prompt a Gemini ({model_name})...")
     try:
        model = genai.GenerativeModel(model_name=model_name, system_instruction=SYSTEM_INSTRUCTION)
        response = model.generate_content(prompt)
        gemini_response_text = getattr(response, 'text', None)
        if gemini_response_text:
             print("\n✨ Respuesta de Gemini:")
             print(gemini_response_text)
        else:
             print("\n🤔 Gemini no devolvió texto en la respuesta.")
             if hasattr(response, 'prompt_feedback'): print(f"   Feedback: {response.prompt_feedback}")
             gemini_response_text = "La inteligencia artificial no proporcionó una respuesta de texto."
        print("-" * 30)
        return gemini_response_text
     except Exception as e:
        print(f"\n❌ Error con Gemini: {e}")
        print("-" * 30)
        return f"Error al procesar la solicitud con la IA." # Mensaje genérico

# --- Tarea de Limpieza ---
def remove_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"⚠️ No se pudo eliminar {path}: {e}")

# --- Endpoint Raíz ---
@app.get("/", tags=["Info"])
async def read_root():
    """Devuelve información básica sobre la API."""
    return {
        "message": f"Bienvenido a la API de {ASSISTANT_NAME}",
        "version": ASSISTANT_VERSION,
        "powered_by": POWERED_BY,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "process_endpoint": "/process_audio/"
    }

# --- Endpoint Principal ---
@app.post("/process_audio/", tags=["Audio Processing"])
async def process_audio_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Recibe audio, transcribe, consulta a Gemini (si aplica), genera voz y devuelve JSON.
    """
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    input_filename = os.path.join(TEMP_DIR, f"input_{uuid.uuid4()}{file_extension}")
    output_audio_path: str | None = None

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
        background_tasks.add_task(remove_file, input_filename)

        # 2. Transcribir
        loop = asyncio.get_event_loop()
        transcribed_text = await loop.run_in_executor(None, transcribe_audio_sync, input_filename, WHISPER_MODEL_NAME)
        if not transcribed_text:
            raise HTTPException(status_code=400, detail="Transcripción fallida o vacía.")

        # 3. Comprobar saludo
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
            print("🤔 No se obtuvo respuesta de texto válida.")
            response_text = "No pude procesar esa solicitud o no tengo una respuesta específica."

        # 5. Generar audio de respuesta
        output_audio_path = await generate_speech_edge_tts(response_text, voice=EDGE_VOICE, rate=EDGE_RATE, volume=EDGE_VOLUME)

        audio_base64: str | None = None
        if output_audio_path:
            # 6. Leer y codificar audio
            try:
                with open(output_audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                print(f"🔊 Audio codificado en Base64 (tamaño: {len(audio_base64)} caracteres)")
            except Exception as e:
                print(f"❌ Error leyendo/codificando audio generado: {e}")
                audio_base64 = None # Indicar que falló
            finally:
                 # Limpiar el archivo de audio generado DESPUÉS de intentar leerlo
                 background_tasks.add_task(remove_file, output_audio_path)
        else:
            print("⚠️ No se generó archivo de audio de respuesta.")
            # Podrías decidir devolver un error aquí si el audio es crítico

        # 7. Devolver JSON (¡Aquí estaba la parte faltante!)
        return JSONResponse(content={
            "responseText": response_text,
            "audioBase64": audio_base64 # Será None si falló la generación/codificación
        })

    except HTTPException as http_exc:
        # Re-lanzar excepciones HTTP para que FastAPI las maneje correctamente
        print(f"HTTP Exception: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        # Capturar cualquier otro error inesperado
        print(f"💥 Error inesperado en el endpoint: {e}")
        # Limpieza de emergencia
        if 'input_filename' in locals() and os.path.exists(input_filename):
             background_tasks.add_task(remove_file, input_filename)
        if output_audio_path and os.path.exists(output_audio_path):
             background_tasks.add_task(remove_file, output_audio_path)
        # Devolver un error genérico 500
        raise HTTPException(status_code=500, detail=f"Error interno del servidor.")

# --- Punto de entrada (para desarrollo local) ---
if __name__ == "__main__":
    import uvicorn
    print("-" * 50)
    print(f"🚀 Iniciando E.R.E.B.U.S. API v{ASSISTANT_VERSION} - Powered by {POWERED_BY}")
    print("   Ejecutando servidor FastAPI localmente...")
    print(f"   Accede a la documentación (Swagger UI) en: http://127.0.0.1:8000/docs")
    print(f"   Accede a la documentación alternativa (ReDoc) en: http://127.0.0.1:8000/redoc")
    print("   Usa Postman, curl o tu frontend React para enviar un archivo de audio")
    print(f"   a POST http://127.0.0.1:8000/process_audio/")
    print("-" * 50)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
