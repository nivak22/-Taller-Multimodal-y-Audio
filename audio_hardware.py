"""Transcribe consultas de hardware y genera respuestas con Gemini."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL = "gemini-3.8-flash"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300
AUDIO_STEMS = ("aud1", "aud2", "aud3", "aud4")
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm"}
TRANSCRIPTION_PROMPT = (
    "Transcribe literalmente el habla de este audio en español. "
    "Devuelve solo la transcripción, sin resumen ni comentarios."
)
SYSTEM_PROMPT = (
    "Eres un técnico experto en hardware de PC. "
    "Responde de forma técnica al problema transcrito."
)


def find_audio(stem: str) -> Path:
    """Encuentra un audio por nombre base en los formatos admitidos."""
    matches = [
        path
        for path in SCRIPT_DIR.iterdir()
        if path.is_file()
        and path.stem.lower() == stem.lower()
        and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]
    if not matches:
        extensions = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise FileNotFoundError(
            f"No se encontró '{stem}' con alguna de estas extensiones: {extensions}"
        )
    if len(matches) > 1:
        filenames = ", ".join(path.name for path in matches)
        raise RuntimeError(
            f"Hay varios audios para '{stem}': {filenames}. "
            "Deja solo uno para evitar ambigüedad."
        )
    return matches[0]


def create_client() -> genai.Client:
    """Carga la clave de Gemini desde .env o desde el entorno."""
    load_dotenv(SCRIPT_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Falta GEMINI_API_KEY. Copia .env.example a .env y agrega tu clave."
        )
    timeout_milliseconds = get_request_timeout_seconds() * 1000
    return genai.Client(
        api_key=api_key,
        http_options={"timeout": timeout_milliseconds},
    )


def get_request_timeout_seconds() -> int:
    """Obtiene el timeout configurable, en segundos, con validación."""
    value = os.getenv(
        "GEMINI_TIMEOUT_SECONDS", str(DEFAULT_REQUEST_TIMEOUT_SECONDS)
    )
    try:
        timeout_seconds = int(value)
    except ValueError as error:
        raise ValueError("GEMINI_TIMEOUT_SECONDS debe ser un entero.") from error
    if timeout_seconds <= 0:
        raise ValueError("GEMINI_TIMEOUT_SECONDS debe ser mayor que cero.")
    return timeout_seconds


def transcribe_audio(client: genai.Client, audio_path: Path) -> str:
    """Sube un audio y solicita a Gemini una transcripción literal en español."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"No se encontró el audio: {audio_path}")
    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(f"Formato de audio no admitido: {audio_path.suffix}")

    print(f"Subiendo {audio_path.name} a Gemini...", flush=True)
    uploaded_file = client.files.upload(file=str(audio_path))
    print("Solicitando transcripción a Gemini...", flush=True)
    interaction = client.interactions.create(
        model=MODEL,
        store=False,
        input=[
            {"type": "text", "text": TRANSCRIPTION_PROMPT},
            {
                "type": "audio",
                "uri": uploaded_file.uri,
                "mime_type": uploaded_file.mime_type,
            },
        ],
    )
    transcription = interaction.output_text
    if not transcription:
        raise RuntimeError("Gemini devolvió una transcripción vacía.")
    return transcription


def answer_transcription(client: genai.Client, transcription: str) -> str:
    """Genera una respuesta técnica a partir de la transcripción."""
    print("Solicitando respuesta técnica a Gemini...", flush=True)
    interaction = client.interactions.create(
        model=MODEL,
        system_instruction=SYSTEM_PROMPT,
        input=f"Problema transcrito:\n{transcription}",
        store=False,
    )
    response = interaction.output_text
    if not response:
        raise RuntimeError("Gemini devolvió una respuesta vacía.")
    return response


def main() -> None:
    """Transcribe los cuatro audios y obtiene una respuesta por cada uno."""
    audio_paths = [find_audio(stem) for stem in AUDIO_STEMS]
    client = create_client()

    print("Iniciando transcripción de audios con Gemini.", flush=True)
    for index, audio_path in enumerate(audio_paths, start=1):
        print(f"\n{'=' * 60}\nAudio {index}: {audio_path.name}", flush=True)
        try:
            transcription = transcribe_audio(client, audio_path)
            print("\nTranscripción:", flush=True)
            print(transcription, flush=True)

            answer = answer_transcription(client, transcription)
            print("\nRespuesta técnica:", flush=True)
            print(answer, flush=True)
        except Exception as error:
            print(
                f"\nNo se pudo procesar {audio_path.name}: {error}",
                flush=True,
            )


if __name__ == "__main__":
    main()
