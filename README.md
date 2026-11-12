# Taller multimodal: diagnóstico de hardware

Ejemplos de análisis de imágenes, transcripción de audio y diagnóstico técnico
con la API de Gemini.

## Preparación

Desde esta carpeta instala las dependencias y crea la configuración local:

```powershell
py -3 -m pip install google-genai python-dotenv
Copy-Item .env.example .env
notepad .env
```

En `.env`, configura `GEMINI_API_KEY` con una clave nueva creada en
[Google AI Studio](https://aistudio.google.com/apikey). No compartas ni publiques
`.env`; `.gitignore` ya lo excluye.

## Ejecución

```powershell
py -3 vision_hardware.py
py -3 audio_hardware.py
```

Ambos scripts usan `gemini-3.8-flash`. El flujo de visión muestra la pregunta,
el nombre de la imagen y la respuesta; valida la extracción JSON, compara dos
imágenes y provoca un error declarando un MIME inválido. El límite de tres
imágenes de la consigna corresponde a Groq y no se debe atribuir a Gemini.
Gemini limita a 20 MB el tamaño total de las solicitudes con imágenes inline;
las solicitudes mayores deben usar Files API.
Los scripts imprimen mensajes de progreso antes de cada llamada. El timeout
HTTP predeterminado es de 300 segundos; puedes cambiarlo con
`GEMINI_TIMEOUT_SECONDS` en `.env`. Un timeout mayor permite esperar más, pero
no soluciona problemas de conexión, disponibilidad o facturación.

El flujo de audio procesa `aud1` a `aud4`: sube cada archivo para obtener su
transcripción y luego solicita una respuesta técnica. Usa cuatro grabaciones
cortas (10 a 30 segundos), con preguntas de hardware, al menos dos voces
distintas y ruido de fondo en una de ellas.

## Archivos de prueba

Coloca en esta carpeta imágenes en PNG, JPG o JPEG, una por cada nombre base:

- `motherboard_1`: consulta y extracción estructurada.
- `cpu_nueva` y `cpu_quemada`: comparación de CPU.
- `placa_base`: imagen para provocar el error de MIME.

Coloca también los cuatro audios con nombres base `aud1`, `aud2`, `aud3` y
`aud4`, en WAV, MP3, M4A, OGG o WEBM. El script detecta la extensión
automáticamente. Si hay más de un archivo con el mismo nombre base, avisa de
la ambigüedad.

Los ejemplos no incluyen imágenes ni grabaciones. Los límites y cuotas de
Gemini pueden diferir de los de Groq; consulta la documentación vigente de
Google AI Studio para tu cuenta.
