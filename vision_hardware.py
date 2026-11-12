"""Analiza componentes de PC con la API multimodal de Gemini."""

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL = "gemini-3.8-flash"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
REQUIRED_FIELDS = {
    "nombre_componente": str,
    "fabricante": str,
    "estado_visible": str,
    "pines_doblados_detectados": bool,
    "categoria": str,
}


def find_image(stem: str) -> Path:
    """Encuentra una imagen por nombre base aceptando PNG, JPG o JPEG."""
    matches = [
        path
        for path in SCRIPT_DIR.iterdir()
        if path.is_file()
        and path.stem.lower() == stem.lower()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if not matches:
        extensions = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise FileNotFoundError(
            f"No se encontró '{stem}' con alguna de estas extensiones: {extensions}"
        )
    if len(matches) > 1:
        filenames = ", ".join(path.name for path in matches)
        raise RuntimeError(
            f"Hay varias imágenes para '{stem}': {filenames}. "
            "Deja solo una para evitar ambigüedad."
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


def make_image_inputs(
    prompt: str, image_paths: list[Path], mime_type_override: str | None = None
) -> list[dict[str, str]]:
    """Codifica imágenes locales en base64 para la API de Gemini."""
    inputs = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
        mime_type, _ = mimetypes.guess_type(image_path.name)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(f"No se pudo determinar el formato de {image_path.name}")
        if mime_type_override is not None:
            mime_type = mime_type_override
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        inputs.append(
            {"type": "image", "data": image_data, "mime_type": mime_type}
        )
    return inputs


def get_text_response(interaction: Any) -> str:
    """Devuelve el texto de la respuesta o informa si Gemini no generó texto."""
    text = interaction.output_text
    if not text:
        raise RuntimeError("Gemini devolvió una respuesta vacía.")
    return text


def ask_about_image(client: genai.Client, image_path: Path) -> None:
    prompt = (
        "¿Qué componente de hardware es este y cuáles son sus "
        "especificaciones principales visibles?"
    )
    print(
        f"\n1. Enviando {image_path.name} a Gemini "
        f"(timeout: {get_request_timeout_seconds()} s)...",
        flush=True,
    )
    interaction = client.interactions.create(
        model=MODEL,
        input=make_image_inputs(prompt, [image_path]),
        store=False,
    )
    print(f"Pregunta: {prompt}")
    print(f"Imagen (archivo local): {image_path.resolve()}")
    print("Respuesta:")
    print(get_text_response(interaction))


def extract_component_json(client: genai.Client, image_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {
            "nombre_componente": {"type": "string"},
            "fabricante": {"type": "string"},
            "estado_visible": {"type": "string"},
            "pines_doblados_detectados": {"type": "boolean"},
            "categoria": {"type": "string"},
        },
        "required": list(REQUIRED_FIELDS),
    }
    prompt = (
        "Analiza la imagen y extrae los datos observables. No inventes detalles "
        "que no sean visibles."
    )
    print("\n2. Solicitando extracción estructurada a Gemini...", flush=True)
    interaction = client.interactions.create(
        model=MODEL,
        input=make_image_inputs(prompt, [image_path]),
        store=False,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": schema,
        },
    )

    try:
        component = json.loads(get_text_response(interaction))
    except json.JSONDecodeError as error:
        print(
            "\n2. Extracción estructurada\n"
            "La respuesta no cumplió el esquema JSON: "
            f"el contenido no es JSON válido ({error})."
        )
        return

    has_expected_shape = (
        isinstance(component, dict)
        and set(component) == set(REQUIRED_FIELDS)
        and all(
            type(component[key]) is value_type
            for key, value_type in REQUIRED_FIELDS.items()
        )
    )
    if not has_expected_shape:
        print("\n2. Extracción estructurada\nLa respuesta no cumplió el esquema JSON.")
        return

    print("\n2. Extracción estructurada")
    print(json.dumps(component, ensure_ascii=False, indent=2))


def compare_images(client: genai.Client, image_paths: list[Path]) -> None:
    prompt = (
        "Compara detalladamente las dos CPU, en el orden presentado. Describe "
        "diferencias visibles, daños posibles y precauciones; no afirmes causas "
        "que no puedan confirmarse solo con las imágenes."
    )
    print("\n3. Enviando las dos imágenes para comparación...", flush=True)
    interaction = client.interactions.create(
        model=MODEL,
        input=make_image_inputs(prompt, image_paths),
        store=False,
    )
    print("\n3. Comparación de CPU nueva y dañada")
    print(get_text_response(interaction))


def provoke_invalid_mime_error(
    client: genai.Client, image_path: Path
) -> None:
    """Envía una imagen con un MIME intencionalmente no válido para provocar error."""
    print("\n4. Enviando solicitud con MIME inválido...", flush=True)
    try:
        client.interactions.create(
            model=MODEL,
            input=make_image_inputs(
                "Identifica el componente de hardware de esta imagen.",
                [image_path],
                mime_type_override="image/formato-invalido",
            ),
            store=False,
        )
    except Exception as error:
        print(f"\n4. Error provocado intencionalmente: {error}")
        print(
            "Causa: se declaró deliberadamente un tipo MIME de imagen no válido "
            "para provocar el rechazo de la API."
        )
        return

    print(
        "\n4. Gemini no rechazó el MIME no válido en esta solicitud; "
        "verifica la causa en el comportamiento de la API y su respuesta."
    )


def main() -> None:
    """Ejecuta secuencialmente los cuatro ejercicios de visión."""
    print("Iniciando pruebas de visión con Gemini.", flush=True)
    motherboard = find_image("motherboard_1")
    cpu_new = find_image("cpu_nueva")
    cpu_damaged = find_image("cpu_quemada")
    fourth_image = find_image("placa_base")
    client = create_client()

    try:
        ask_about_image(client, motherboard)
        extract_component_json(client, motherboard)
        compare_images(client, [cpu_new, cpu_damaged])
        provoke_invalid_mime_error(client, fourth_image)
    except Exception as error:
        error_message = str(error)
        if "billing account" in error_message.lower() or "403" in error_message:
            print(
                "\nGemini rechazó la solicitud por un problema de facturación "
                "del proyecto asociado a la clave. Resuelve la facturación de "
                "ese proyecto o configura una clave de un proyecto habilitado."
            )
            print(f"Detalle de Google: {error_message}")
            return

        print(f"\nLa ejecución se detuvo por un error de Gemini o de red: {error}")


if __name__ == "__main__":
    main()
