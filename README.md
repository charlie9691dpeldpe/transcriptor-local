# Transcriptor Local — App de escritorio para Windows

Aplicación de escritorio con interfaz gráfica para transcribir audio y video
de forma 100% local, usando Whisper (OpenAI) + PyTorch. Soporta español e
inglés, transcripción en vivo, modo claro/oscuro, y usa GPU NVIDIA
automáticamente si está disponible, con fallback a CPU.

## Contenido

- `app.py` — código de la aplicación (Tkinter + Whisper/PyTorch)
- `requirements.txt` — dependencias de Python
- `build_exe.bat` — script para compilar el `.exe` localmente (opcional, no es el flujo principal)
- `.github/workflows/build.yml` — compila el `.exe` automáticamente en GitHub Actions (flujo recomendado)

## Cómo obtener la app (sin instalar Python)

El código vive en este repositorio, y se compila automáticamente a `.exe`
usando GitHub Actions — no necesitas Python, PyInstaller ni nada instalado
en tu PC para obtener el ejecutable.

1. Ve a la pestaña **Actions** de este repositorio.
2. Entra a la ejecución más reciente con ✅ verde de **"Build Windows EXE"**
   (o dispara una nueva con **Run workflow** si hiciste cambios).
3. Baja a la sección **Artifacts** → descarga **TranscriptorLocal-Windows**.
4. Descomprime el `.zip`. Vas a tener:
   ```
   TranscriptorLocal.exe
   models/
   ```
5. Mantén `TranscriptorLocal.exe` y la carpeta `models` **siempre juntos**,
   en la misma carpeta. Los modelos `large-v3` y `medium` ya vienen
   precargados ahí — no hace falta descargar nada la primera vez.

## Requisitos en la PC donde se USA la app

1. **ffmpeg** instalado y en el PATH:
   ```powershell
   winget install ffmpeg
   ```
   Reinicia PowerShell (o la PC) después de instalar.

2. **Driver de NVIDIA actualizado** (opcional, solo para usar GPU). Con
   PyTorch **no hace falta instalar CUDA Toolkit ni cuDNN por separado** —
   el runtime de CUDA viene embebido dentro del propio `.exe`. Si no tienes
   GPU o el driver falla, la app cae a CPU automáticamente sin trabarse.

## Uso de la aplicación

1. **Examinar** → selecciona tu archivo de audio o video (mp3, wav, mp4, etc.)
2. **Elegir** carpeta de salida.
3. **Modelo**: `tiny`/`base` (rápidos, menos precisos) hasta `large-v3`
   (más lento, más preciso). `large-v3` y `medium` ya vienen precargados.
4. **Idioma**: Español, Inglés, o Detectar automáticamente.
5. Marca **Usar GPU** si tienes NVIDIA (si falla, usa CPU sin trabarse).
6. Click en **Transcribir**.

**Panel izquierdo** — transcripción en vivo, con barra de progreso real
(% basado en duración del audio). El texto es editable: corrige lo que
haga falta y usa **Guardar cambios** para sobrescribir los archivos
generados.

**Panel derecho** — todos los controles y el registro de actividad.

**Modo oscuro** — botón arriba a la derecha para alternar entre tema claro
y oscuro.

Al terminar se generan 3 archivos en la carpeta de salida:
- `nombre_transcripcion.txt` — con marcas de tiempo por segmento
- `nombre_transcripcion.srt` — subtítulos listos para editar video
- `nombre_transcripcion.md` — texto plano en formato Markdown

## Notas sobre GPU

La app usa PyTorch, que trae su propio runtime de CUDA embebido — no
depende de que tengas CUDA Toolkit o cuDNN instalados en el sistema, solo
el driver de NVIDIA normal. Esto la hace compatible con versiones de CUDA
más nuevas sin depender de que otras librerías externas ya las soporten.

Si la GPU falla al procesar (driver desactualizado, tarjeta no compatible,
etc.), la app lo detecta automáticamente y reintenta en CPU sin
interrumpir la transcripción.

## Modelos recomendados según tu equipo

| Modelo    | VRAM/RAM aprox. | Velocidad | Precisión | Precargado |
|-----------|------------------|-----------|-----------|------------|
| tiny      | ~1 GB            | Muy rápido| Baja      | No |
| base      | ~1 GB            | Rápido    | Media-baja| No |
| small     | ~2 GB            | Media     | Media     | No |
| medium    | ~5 GB            | Media-lenta| Buena    | **Sí** |
| large-v3  | ~10 GB           | Lenta     | Muy buena | **Sí** |

## Guía completa de referencia

Para el checklist completo de instalación, reinstalación de Windows, y
problemas ya resueltos, consulta `GUIA_MAESTRA.md`.
