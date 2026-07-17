# Transcriptor Local (Whisper) — App de escritorio para Windows

Aplicación de escritorio con interfaz gráfica para transcribir audio y video
de forma 100% local, usando `faster-whisper`. Soporta español e inglés, y usa
GPU NVIDIA (CUDA) automáticamente si está disponible, con fallback a CPU.

## Contenido

- `app.py` — código de la aplicación (Tkinter + faster-whisper)
- `requirements.txt` — dependencias de Python
- `build_exe.bat` — script para generar el `.exe` con PyInstaller

## Requisitos previos (en tu PC Windows)

1. **Python 3.10 o superior** instalado (marca la opción "Add to PATH" al instalar).
   Descárgalo desde https://www.python.org/downloads/
2. **Para usar GPU (opcional pero recomendado):**
   - Tarjeta NVIDIA con drivers actualizados.
   - CUDA Toolkit y cuDNN instalados (o usa la versión que trae CUDA embebido,
     ver nota abajo). Si no los tienes, la app detecta que no hay GPU y usa CPU
     automáticamente — no falla.
3. **ffmpeg** instalado y en el PATH (necesario para leer los archivos de audio/video).
   - Más fácil: `winget install ffmpeg` en PowerShell, o descargar de
     https://www.gyan.dev/ffmpeg/builds/ y agregarlo al PATH.

## Pasos para probar la app (sin empaquetar, modo rápido)

Abre una terminal (PowerShell o CMD) en la carpeta del proyecto y ejecuta:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Esto abrirá la ventana de la aplicación directamente.

## Empaquetar como .exe (para usar sin terminal, doble clic)

Ejecuta:

```bat
build_exe.bat
```

Al terminar, el ejecutable estará en `dist\TranscriptorLocal.exe`. Puedes
copiar ese único archivo a cualquier carpeta o al escritorio.

> Nota: la primera vez que ejecutes una transcripción, el modelo de Whisper
> se descargará automáticamente (unos cientos de MB a ~3GB según el modelo
> elegido) y quedará en caché para las siguientes veces.

## Uso de la aplicación

1. **Examinar...** → selecciona tu archivo de audio o video (mp3, wav, mp4, etc.)
2. **Elegir carpeta de salida** → dónde se guardarán los archivos generados.
3. **Modelo**: `tiny`/`base` (rápidos, menos precisos) hasta `large-v3`
   (más lento, más preciso). Para tu uso recomiendo `medium` o `large-v3`
   si tienes GPU.
4. **Idioma**: Español, Inglés, o Detectar automáticamente.
5. Marca **Usar GPU** si tienes NVIDIA (si no la detecta, usa CPU sin fallar).
6. Click en **Transcribir**. Verás el progreso en el registro inferior.

Al terminar se generan 3 archivos en la carpeta de salida:
- `nombre_transcripcion.txt` — con marcas de tiempo por segmento
- `nombre_transcripcion.srt` — subtítulos listos para editar video
- `nombre_transcripcion.md` — texto plano en formato Markdown

## Notas sobre GPU

`faster-whisper` usa `ctranslate2`, que necesita las librerías CUDA de NVIDIA
en el sistema para acelerar por GPU. Si al ejecutar ves en el registro
"No se detectó GPU compatible. Usando CPU...", significa que CUDA/cuDNN no
están instalados o no son compatibles — la app seguirá funcionando en CPU
sin problema, solo más lento.

## Modelos recomendados según tu equipo

| Modelo    | VRAM/RAM aprox. | Velocidad | Precisión |
|-----------|------------------|-----------|-----------|
| tiny      | ~1 GB            | Muy rápido| Baja      |
| base      | ~1 GB            | Rápido    | Media-baja|
| small     | ~2 GB            | Media     | Media     |
| medium    | ~5 GB            | Media-lenta| Buena    |
| large-v3  | ~10 GB           | Lenta     | Muy buena |
