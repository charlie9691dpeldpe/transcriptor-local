"""
Transcriptor de Audio Local - GUI estilo Claude (con modo claro/oscuro)
Usa Whisper (OpenAI) + PyTorch para transcribir localmente, con GPU (CUDA) o CPU.
PyTorch trae su propio runtime CUDA embebido, no requiere CUDA Toolkit/cuDNN
instalados por separado en el sistema.
"""

import os
import math
import sys
import threading
import traceback
from datetime import timedelta
from pathlib import Path

# En builds compilados con --windowed (sin consola), Windows deja sys.stdout
# y sys.stderr en None. Cualquier librería que intente escribir ahí (tqdm,
# warnings internos de torch/whisper, etc.) se cae con un AttributeError a
# mitad de una operación -- por ejemplo, a mitad de descargar un modelo,
# dejando el archivo corrupto. Este parche evita ese problema de raíz.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# Sin esto, Windows asume que la app "no sabe" de pantallas de alta
# resolución (DPI) y la escala estirando los píxeles -- por eso se ve
# borrosa/pixelada en monitores 4K o de alta densidad. Hay que avisarle a
# Windows ANTES de crear cualquier ventana.
if sys.platform == "win32":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-Monitor-V2
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "Transcriptor Local"
FONT_FAMILY = "Segoe UI"

# --------------------------------------------------------------------------
# Paletas estilo Claude - claro y oscuro
# --------------------------------------------------------------------------
LIGHT = {
    "bg": "#FAF9F5", "panel": "#FFFFFF", "border": "#E5E4DF",
    "text": "#3D3929", "text_muted": "#87867F", "accent": "#D97757",
    "accent_hover": "#C4633F", "track": "#EDEBE3", "surface": "#FFFFFF",
    "log_bg": "#F5F4EF", "btn_disabled": "#C9C7BE",
}
DARK = {
    "bg": "#262624", "panel": "#30302E", "border": "#44433F",
    "text": "#F1F0EC", "text_muted": "#A9A8A0", "accent": "#D97757",
    "accent_hover": "#E08863", "track": "#3D3B36", "surface": "#3A3935",
    "log_bg": "#211F1C", "btn_disabled": "#5A5852",
}


def format_timestamp_srt(seconds: float) -> str:
    td = timedelta(seconds=max(0, seconds))
    total_ms = int(td.total_seconds() * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_timestamp_txt(seconds: float) -> str:
    td = timedelta(seconds=max(0, seconds))
    total_s = int(td.total_seconds())
    hours, rem = divmod(total_s, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


LANGUAGE_MAP = {
    "Detectar automáticamente": None,
    "Español": "es",
    "Inglés": "en",
}
MODEL_OPTIONS = ["tiny", "base", "small", "medium", "large-v3"]
MODEL_SIZE_LABEL = {
    "tiny": "~75 MB",
    "base": "~145 MB",
    "small": "~480 MB",
    "medium": "~1.5 GB",
    "large-v3": "~3 GB",
}


# --------------------------------------------------------------------------
# Detección y extracción de pistas de audio específicas.
# Whisper por defecto deja que ffmpeg elija automáticamente qué pista de
# audio usar, lo cual puede mezclar/alternar entre pistas en videos con
# doblaje múltiple. Aquí detectamos todas las pistas disponibles y permitimos
# elegir una explícitamente con "-map 0:a:N".
# --------------------------------------------------------------------------

def probe_audio_tracks(filepath):
    """Devuelve una lista de dicts {track_index, language, title} por cada
    pista de audio del archivo, usando ffprobe. Si falla o solo hay una
    pista, devuelve una lista de 0 o 1 elementos sin bloquear la app."""
    import subprocess
    import json

    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index:stream_tags=language,title",
            "-of", "json", filepath,
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        out = subprocess.run(
            cmd, capture_output=True, check=True, creationflags=creationflags
        ).stdout
        data = json.loads(out)
        streams = data.get("streams", [])
    except Exception:
        return []

    tracks = []
    for i, s in enumerate(streams):
        tags = s.get("tags", {}) or {}
        tracks.append({
            "track_index": i,
            "language": tags.get("language"),
            "title": tags.get("title"),
        })
    return tracks


def load_audio_track(filepath, track_index=None, sr=16000):
    """Extrae audio mono 16kHz float32 (formato que espera Whisper), igual
    que whisper.audio.load_audio, pero permitiendo elegir una pista
    específica con -map cuando el archivo tiene varias."""
    import subprocess
    import numpy as np

    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", filepath]
    if track_index is not None:
        cmd += ["-map", f"0:a:{track_index}"]
    cmd += ["-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"]

    creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    proc = subprocess.run(cmd, capture_output=True, creationflags=creationflags)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg falló al extraer audio: {proc.stderr.decode(errors='ignore')[-500:]}"
        )
    return np.frombuffer(proc.stdout, np.int16).flatten().astype(np.float32) / 32768.0


# --------------------------------------------------------------------------
# Puente de progreso global: en vez de adivinar cómo cada librería importó
# `tqdm` internamente (frágil, cambia entre versiones), parcheamos el
# método `update` de la propia CLASE tqdm.tqdm una sola vez. Como todas las
# formas de importar ("import tqdm" o "from tqdm import tqdm") terminan
# apuntando al mismo objeto de clase, este parche funciona sin importar
# cómo lo use la librería por dentro.
# --------------------------------------------------------------------------
_TQDM_PATCHED = False
_progress_callback_holder = {"callback": None}


def _install_tqdm_progress_hook():
    global _TQDM_PATCHED
    if _TQDM_PATCHED:
        return
    try:
        import tqdm as tqdm_pkg
        original_update = tqdm_pkg.tqdm.update

        def patched_update(self, n=1):
            result = original_update(self, n)
            cb = _progress_callback_holder["callback"]
            if cb:
                try:
                    total = getattr(self, "total", None) or 1
                    current = getattr(self, "n", 0)
                    pct = min(100, int(current / total * 100))
                    cb(pct)
                except Exception:
                    pass
            return result

        tqdm_pkg.tqdm.update = patched_update
        _TQDM_PATCHED = True
    except Exception:
        pass


def _set_progress_callback(callback):
    _progress_callback_holder["callback"] = callback


# --------------------------------------------------------------------------
# Lógica de transcripción (hilo aparte)
# --------------------------------------------------------------------------

class TranscriberWorker(threading.Thread):
    """
    Motor de transcripción basado en el Whisper original de OpenAI + PyTorch.
    PyTorch trae su propio runtime de CUDA embebido en el paquete, por lo que
    NO requiere tener el CUDA Toolkit / cuDNN instalados por separado en el
    sistema — solo el driver de NVIDIA normal, el más reciente que ya tengas.

    Nota: es más lento que la versión basada en ctranslate2/faster-whisper,
    pero es compatible con versiones de CUDA más nuevas (12, 13, etc.) sin
    depender de que esa librería externa ya las soporte.
    """

    def __init__(self, filepath, model_size, language, use_gpu, out_dir,
                 on_status, on_segment, on_progress_pct, on_done, on_error,
                 on_clear=None, on_time_update=None, audio_track=None):
        super().__init__(daemon=True)
        self.filepath = filepath
        self.model_size = model_size
        self.language = language
        self.use_gpu = use_gpu
        self.out_dir = out_dir
        self.on_status = on_status
        self.on_segment = on_segment
        self.on_progress_pct = on_progress_pct
        self.on_done = on_done
        self.on_error = on_error
        self.on_clear = on_clear or (lambda: None)
        self.on_time_update = on_time_update or (lambda elapsed, eta: None)
        self.audio_track = audio_track

    CHUNK_SECONDS = 300  # procesar en bloques de 5 minutos

    @staticmethod
    def _fmt_duration(seconds):
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m:02d}m {s:02d}s"
        return f"{m}m {s:02d}s"

    def run(self):
        try:
            self.on_status("Cargando motor de transcripción (Whisper + PyTorch)...")
            import time
            import numpy as np
            import whisper
            import whisper.audio as whisper_audio
            import torch

            device = "cpu"
            if self.use_gpu:
                cuda_ok = torch.cuda.is_available()
                self.on_status(
                    f"Diagnóstico GPU -> torch.cuda.is_available(): {cuda_ok} | "
                    f"CUDA build de PyTorch: {torch.version.cuda}"
                )
                if cuda_ok:
                    try:
                        gpu_name = torch.cuda.get_device_name(0)
                        self.on_status(f"GPU detectada: {gpu_name}")
                    except Exception as diag_err:
                        self.on_status(f"GPU detectada pero no se pudo leer el nombre: {diag_err}")
                    device = "cuda"
                else:
                    self.on_status("No se detectó GPU compatible. Usando CPU...")

            if getattr(sys, "frozen", False):
                base_dir = Path(sys.executable).parent
            else:
                base_dir = Path(__file__).resolve().parent
            models_dir = str(base_dir / "models")
            os.makedirs(models_dir, exist_ok=True)

            _set_progress_callback(None)

            self.on_status("Cargando audio completo en memoria...")
            sr = whisper_audio.SAMPLE_RATE
            if self.audio_track is not None:
                self.on_status(f"Extrayendo pista de audio #{self.audio_track}...")
                audio_array = load_audio_track(self.filepath, track_index=self.audio_track, sr=sr)
            else:
                audio_array = whisper_audio.load_audio(self.filepath)
            total_duration = len(audio_array) / sr
            self.on_status(f"Duración de audio detectada: {total_duration:.1f} segundos")

            chunk_samples = int(self.CHUNK_SECONDS * sr)
            total_samples = len(audio_array)
            num_chunks = max(1, math.ceil(total_samples / chunk_samples))

            def run_pass(dev):
                self.on_status(f"Cargando modelo '{self.model_size}' en {dev.upper()}...")
                m = whisper.load_model(self.model_size, device=dev, download_root=models_dir)
                self.on_status(
                    f"Transcribiendo en {dev.upper()} en {num_chunks} bloque(s) de "
                    f"{self.CHUNK_SECONDS // 60} min..."
                )

                out_txt, out_srt, out_plain, out_segments = [], [], [], []
                detected_lang = self.language
                prev_text_tail = None
                start_time = time.time()

                for idx in range(num_chunks):
                    chunk_start_sample = idx * chunk_samples
                    chunk_end_sample = min(total_samples, chunk_start_sample + chunk_samples)
                    chunk = audio_array[chunk_start_sample:chunk_end_sample]
                    chunk_offset = chunk_start_sample / sr

                    self.on_status(f"Procesando bloque {idx + 1} de {num_chunks}...")

                    result = m.transcribe(
                        chunk.astype(np.float32),
                        language=self.language,
                        fp16=(dev == "cuda"),
                        verbose=False,
                        initial_prompt=prev_text_tail,
                    )

                    if detected_lang is None:
                        detected_lang = result.get("language")

                    chunk_segments = result.get("segments", [])
                    for seg in chunk_segments:
                        start_sec = seg["start"] + chunk_offset
                        end_sec = seg["end"] + chunk_offset
                        text = seg["text"].strip()
                        if not text:
                            continue
                        start_fmt = format_timestamp_txt(start_sec)
                        end_fmt = format_timestamp_txt(end_sec)
                        out_txt.append(f"[{start_fmt} --> {end_fmt}] {text}")
                        out_plain.append(text)
                        out_segments.append({"start": start_sec, "end": end_sec, "text": text})
                        out_srt.append(
                            f"{len(out_plain)}\n{format_timestamp_srt(start_sec)} --> "
                            f"{format_timestamp_srt(end_sec)}\n{text}\n"
                        )
                        self.on_segment(text, start_fmt, end_fmt)

                    chunk_text = result.get("text", "").strip()
                    prev_text_tail = chunk_text[-200:] if chunk_text else prev_text_tail

                    # Progreso, tiempo transcurrido y tiempo restante estimado
                    chunks_done = idx + 1
                    elapsed = time.time() - start_time
                    avg_per_chunk = elapsed / chunks_done
                    remaining_chunks = num_chunks - chunks_done
                    eta = avg_per_chunk * remaining_chunks

                    pct = min(100, int(chunk_end_sample / total_samples * 100))
                    self.on_progress_pct(pct)
                    self.on_time_update(self._fmt_duration(elapsed), self._fmt_duration(eta))

                return detected_lang, out_txt, out_srt, out_plain, out_segments

            try:
                detected_lang, lines_txt, lines_srt, plain_text, segments = run_pass(device)
            except Exception as gpu_err:
                error_text = str(gpu_err).lower()
                gpu_related = any(
                    kw in error_text for kw in ("cuda", "dll", "library", "driver", "gpu")
                )
                if device == "cuda":
                    self.on_status(f"Detalle del error de GPU: {gpu_err}")
                if device == "cuda" and gpu_related:
                    self.on_status(
                        "La GPU falló al procesar (revisa que el driver esté al día). "
                        "Reintentando automáticamente en CPU..."
                    )
                    device = "cpu"
                    self.on_clear()
                    self.on_progress_pct(0)
                    detected_lang, lines_txt, lines_srt, plain_text, segments = run_pass(device)
                else:
                    raise

            base_name = Path(self.filepath).stem
            out_dir = Path(self.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            txt_path = out_dir / f"{base_name}_transcripcion.txt"
            srt_path = out_dir / f"{base_name}_transcripcion.srt"
            md_path = out_dir / f"{base_name}_transcripcion.md"
            plain_txt_path = out_dir / f"{base_name}_texto_plano.txt"

            txt_path.write_text("\n".join(lines_txt), encoding="utf-8")
            srt_path.write_text("\n".join(lines_srt), encoding="utf-8")
            plain_txt_path.write_text("\n".join(plain_text), encoding="utf-8")

            md_content = (
                f"# Transcripción: {base_name}\n\n"
                f"- Idioma detectado/usado: {detected_lang or 'desconocido'}\n"
                f"- Modelo: {self.model_size}\n\n"
                f"## Texto completo\n\n{chr(10).join(plain_text)}\n"
            )
            md_path.write_text(md_content, encoding="utf-8")

            self.on_progress_pct(100)
            # Se pasa también el audio cargado, la config usada y los segmentos,
            # para que un paso de traducción posterior (opcional) no tenga que
            # volver a leer/decodificar el archivo desde cero.
            self.on_done(
                str(txt_path), str(srt_path), str(md_path), str(plain_txt_path),
                {
                    "audio_array": audio_array, "sr": sr, "device": device,
                    "model_size": self.model_size, "base_name": base_name,
                    "segments": segments,
                },
            )
        except Exception as e:
            self.on_error(f"{e}\n\n{traceback.format_exc()}")


class TranslateToEnglishWorker(threading.Thread):
    """Segundo paso opcional: reusa el audio ya cargado en memoria para
    traducir (tarea nativa de Whisper) hacia inglés, sin re-leer el archivo."""

    CHUNK_SECONDS = TranscriberWorker.CHUNK_SECONDS

    def __init__(self, audio_array, sr, device, model_size, out_dir, base_name,
                 on_status, on_segment, on_progress_pct, on_done, on_error, on_clear):
        super().__init__(daemon=True)
        self.audio_array = audio_array
        self.sr = sr
        self.device = device
        self.model_size = model_size
        self.out_dir = out_dir
        self.base_name = base_name
        self.on_status = on_status
        self.on_segment = on_segment
        self.on_progress_pct = on_progress_pct
        self.on_done = on_done
        self.on_error = on_error
        self.on_clear = on_clear

    def run(self):
        try:
            import time
            import numpy as np
            import whisper

            self.on_clear()
            self.on_progress_pct(0)

            if getattr(sys, "frozen", False):
                base_dir = Path(sys.executable).parent
            else:
                base_dir = Path(__file__).resolve().parent
            models_dir = str(base_dir / "models")

            self.on_status(f"Cargando modelo '{self.model_size}' en {self.device.upper()}...")
            m = whisper.load_model(self.model_size, device=self.device, download_root=models_dir)

            chunk_samples = int(self.CHUNK_SECONDS * self.sr)
            total_samples = len(self.audio_array)
            num_chunks = max(1, math.ceil(total_samples / chunk_samples))
            self.on_status(f"Traduciendo al inglés en {num_chunks} bloque(s)...")

            out_txt, out_srt, out_plain = [], [], []
            start_time = time.time()

            for idx in range(num_chunks):
                chunk_start_sample = idx * chunk_samples
                chunk_end_sample = min(total_samples, chunk_start_sample + chunk_samples)
                chunk = self.audio_array[chunk_start_sample:chunk_end_sample]
                chunk_offset = chunk_start_sample / self.sr

                self.on_status(f"Traduciendo bloque {idx + 1} de {num_chunks}...")
                result = m.transcribe(
                    chunk.astype(np.float32),
                    fp16=(self.device == "cuda"),
                    verbose=False,
                    task="translate",
                )
                for seg in result.get("segments", []):
                    start_sec = seg["start"] + chunk_offset
                    end_sec = seg["end"] + chunk_offset
                    text = seg["text"].strip()
                    if not text:
                        continue
                    start_fmt = format_timestamp_txt(start_sec)
                    end_fmt = format_timestamp_txt(end_sec)
                    out_txt.append(f"[{start_fmt} --> {end_fmt}] {text}")
                    out_plain.append(text)
                    out_srt.append(
                        f"{len(out_plain)}\n{format_timestamp_srt(start_sec)} --> "
                        f"{format_timestamp_srt(end_sec)}\n{text}\n"
                    )
                    self.on_segment(text, start_fmt, end_fmt)

                chunks_done = idx + 1
                elapsed = time.time() - start_time
                eta = (elapsed / chunks_done) * (num_chunks - chunks_done)
                pct = min(100, int(chunk_end_sample / total_samples * 100))
                self.on_progress_pct(pct)
                self.on_status(
                    f"Transcurrido: {TranscriberWorker._fmt_duration(elapsed)} · "
                    f"Restante: ~{TranscriberWorker._fmt_duration(eta)}"
                )

            out_dir = Path(self.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            txt_path = out_dir / f"{self.base_name}_ingles.txt"
            srt_path = out_dir / f"{self.base_name}_ingles.srt"
            plain_path = out_dir / f"{self.base_name}_ingles_texto_plano.txt"
            txt_path.write_text("\n".join(out_txt), encoding="utf-8")
            srt_path.write_text("\n".join(out_srt), encoding="utf-8")
            plain_path.write_text("\n".join(out_plain), encoding="utf-8")

            self.on_progress_pct(100)
            self.on_done(str(txt_path), str(srt_path), str(plain_path))
        except Exception as e:
            self.on_error(f"{e}\n\n{traceback.format_exc()}")


class TranslateToSpanishWorker(threading.Thread):
    """Segundo paso opcional: traduce el texto (en inglés) que esté
    actualmente mostrado en el panel hacia español, usando un modelo local
    de traducción de texto (no vuelve a tocar el audio)."""

    def __init__(self, segments, out_dir, base_name,
                 on_status, on_segment, on_progress_pct, on_done, on_error, on_clear):
        super().__init__(daemon=True)
        self.segments = segments  # lista de dicts {start, end, text}
        self.out_dir = out_dir
        self.base_name = base_name
        self.on_status = on_status
        self.on_segment = on_segment
        self.on_progress_pct = on_progress_pct
        self.on_done = on_done
        self.on_error = on_error
        self.on_clear = on_clear

    def run(self):
        try:
            import torch

            self.on_clear()
            self.on_progress_pct(0)

            if not self.segments:
                raise RuntimeError("No hay texto para traducir todavía.")

            if getattr(sys, "frozen", False):
                base_dir = Path(sys.executable).parent
            else:
                base_dir = Path(__file__).resolve().parent
            models_dir = str(base_dir / "models")

            self.on_status(
                "Cargando modelo de traducción inglés→español "
                "(la primera vez descarga ~300MB)..."
            )
            from transformers import MarianMTModel, MarianTokenizer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            name = "Helsinki-NLP/opus-mt-en-es"
            tok = MarianTokenizer.from_pretrained(name, cache_dir=models_dir)
            tmodel = MarianMTModel.from_pretrained(name, cache_dir=models_dir)
            if device == "cuda":
                tmodel = tmodel.to("cuda")

            out_txt, out_srt, out_plain = [], [], []
            batch_size = 16
            total = len(self.segments)

            for start_i in range(0, total, batch_size):
                batch = self.segments[start_i:start_i + batch_size]
                texts = [s["text"] for s in batch]
                self.on_status(f"Traduciendo {start_i + len(batch)} de {total} líneas...")

                inputs = tok(texts, return_tensors="pt", padding=True, truncation=True)
                if device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    generated = tmodel.generate(**inputs, max_length=512)
                translated = [tok.decode(t, skip_special_tokens=True).strip() for t in generated]

                for seg, es_text in zip(batch, translated):
                    start_fmt = format_timestamp_txt(seg["start"])
                    end_fmt = format_timestamp_txt(seg["end"])
                    out_txt.append(f"[{start_fmt} --> {end_fmt}] {es_text}")
                    out_plain.append(es_text)
                    out_srt.append(
                        f"{len(out_plain)}\n{format_timestamp_srt(seg['start'])} --> "
                        f"{format_timestamp_srt(seg['end'])}\n{es_text}\n"
                    )
                    self.on_segment(es_text, start_fmt, end_fmt)

                self.on_progress_pct(min(100, int((start_i + len(batch)) / total * 100)))

            out_dir = Path(self.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            txt_path = out_dir / f"{self.base_name}_espanol.txt"
            srt_path = out_dir / f"{self.base_name}_espanol.srt"
            plain_path = out_dir / f"{self.base_name}_espanol_texto_plano.txt"
            txt_path.write_text("\n".join(out_txt), encoding="utf-8")
            srt_path.write_text("\n".join(out_srt), encoding="utf-8")
            plain_path.write_text("\n".join(out_plain), encoding="utf-8")

            self.on_progress_pct(100)
            self.on_done(str(txt_path), str(srt_path), str(plain_path))
        except Exception as e:
            self.on_error(f"{e}\n\n{traceback.format_exc()}")


class TextDocumentTranslateWorker(threading.Thread):
    """Traduce un documento de texto cualquiera (no producido por la
    transcripción), de forma independiente. target_lang: 'en' o 'es'."""

    def __init__(self, text, target_lang, out_dir, base_name,
                 on_status, on_progress_pct, on_done, on_error):
        super().__init__(daemon=True)
        self.text = text
        self.target_lang = target_lang
        self.out_dir = out_dir
        self.base_name = base_name
        self.on_status = on_status
        self.on_progress_pct = on_progress_pct
        self.on_done = on_done
        self.on_error = on_error

    def run(self):
        try:
            import torch

            if getattr(sys, "frozen", False):
                base_dir = Path(sys.executable).parent
            else:
                base_dir = Path(__file__).resolve().parent
            models_dir = str(base_dir / "models")
            os.makedirs(models_dir, exist_ok=True)

            model_name = (
                "Helsinki-NLP/opus-mt-es-en" if self.target_lang == "en"
                else "Helsinki-NLP/opus-mt-en-es"
            )
            self.on_status(
                f"Cargando modelo de traducción ({model_name.split('/')[-1]}) "
                "(la primera vez descarga ~300MB)..."
            )
            from transformers import MarianMTModel, MarianTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tok = MarianTokenizer.from_pretrained(model_name, cache_dir=models_dir)
            model = MarianMTModel.from_pretrained(model_name, cache_dir=models_dir)
            if device == "cuda":
                model = model.to("cuda")

            lines = [l for l in self.text.splitlines() if l.strip()]
            if not lines:
                lines = [self.text.strip()]

            translated_lines = []
            batch_size = 16
            total = len(lines)

            for i in range(0, total, batch_size):
                batch = lines[i:i + batch_size]
                self.on_status(f"Traduciendo línea {i + len(batch)} de {total}...")
                inputs = tok(batch, return_tensors="pt", padding=True, truncation=True)
                if device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    generated = model.generate(**inputs, max_length=512)
                translated_lines.extend(
                    tok.decode(t, skip_special_tokens=True).strip() for t in generated
                )
                self.on_progress_pct(min(100, int((i + len(batch)) / total * 100)))

            translated_text = "\n".join(translated_lines)

            out_dir = Path(self.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = "ingles" if self.target_lang == "en" else "espanol"
            out_path = out_dir / f"{self.base_name}_traducido_{suffix}.txt"
            out_path.write_text(translated_text, encoding="utf-8")

            self.on_progress_pct(100)
            self.on_done(translated_text, str(out_path))
        except Exception as e:
            self.on_error(f"{e}\n\n{traceback.format_exc()}")


class WebsiteDownloaderWorker(threading.Thread):
    """Descarga las páginas HTML de un sitio (mismo dominio), siguiendo
    enlaces internos, hasta un límite de páginas. Solo HTML por ahora
    (no imágenes/CSS/JS) -- pensado para poder traducir el contenido."""

    def __init__(self, start_url, out_dir, page_limit,
                 on_status, on_progress_pct, on_done, on_error):
        super().__init__(daemon=True)
        self.start_url = start_url
        self.out_dir = out_dir
        self.page_limit = max(1, page_limit)
        self.on_status = on_status
        self.on_progress_pct = on_progress_pct
        self.on_done = on_done
        self.on_error = on_error

    @staticmethod
    def _url_to_local_path(url, base_dir):
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path or "/"
        if path.endswith("/"):
            path = path + "index.html"
        if not os.path.splitext(path)[1]:
            path = path + ".html"
        return Path(base_dir) / parsed.netloc / path.lstrip("/")

    def run(self):
        try:
            import requests
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin, urlparse

            start_url = self.start_url.strip()
            if not start_url.startswith("http"):
                start_url = "https://" + start_url
            domain = urlparse(start_url).netloc
            if not domain:
                raise RuntimeError("La URL no parece válida.")

            out_dir = Path(self.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            visited = set()
            queue = [start_url]
            saved_count = 0
            headers = {"User-Agent": "Mozilla/5.0 (compatible; TranscriptorLocalBot/1.0)"}

            while queue and saved_count < self.page_limit:
                url = queue.pop(0)
                if url in visited:
                    continue
                visited.add(url)

                self.on_status(
                    f"Descargando página {saved_count + 1} de {self.page_limit}: {url}"
                )
                try:
                    resp = requests.get(url, headers=headers, timeout=15)
                    resp.raise_for_status()
                except Exception as fetch_err:
                    self.on_status(f"No se pudo descargar {url}: {fetch_err}")
                    continue

                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    continue

                local_path = self._url_to_local_path(url, out_dir)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(resp.content)
                saved_count += 1

                pct = min(100, int(saved_count / self.page_limit * 100))
                self.on_progress_pct(pct)

                try:
                    soup = BeautifulSoup(resp.content, "html.parser")
                    for a in soup.find_all("a", href=True):
                        link = urljoin(url, a["href"])
                        link = urlparse(link)._replace(fragment="").geturl()
                        if urlparse(link).netloc == domain and link not in visited:
                            queue.append(link)
                except Exception:
                    pass

            self.on_progress_pct(100)
            self.on_done(saved_count, str(out_dir))
        except Exception as e:
            self.on_error(f"{e}\n\n{traceback.format_exc()}")


class WebsiteTranslateWorker(threading.Thread):
    """Traduce todas las páginas .html de una carpeta (por ejemplo, un
    sitio ya descargado con WebsiteDownloaderWorker), conservando la
    estructura HTML y solo reemplazando el texto visible."""

    def __init__(self, site_dir, target_lang, out_dir,
                 on_status, on_progress_pct, on_done, on_error):
        super().__init__(daemon=True)
        self.site_dir = site_dir
        self.target_lang = target_lang
        self.out_dir = out_dir
        self.on_status = on_status
        self.on_progress_pct = on_progress_pct
        self.on_done = on_done
        self.on_error = on_error

    def run(self):
        try:
            import torch
            from bs4 import BeautifulSoup

            if getattr(sys, "frozen", False):
                base_dir = Path(sys.executable).parent
            else:
                base_dir = Path(__file__).resolve().parent
            models_dir = str(base_dir / "models")
            os.makedirs(models_dir, exist_ok=True)

            model_name = (
                "Helsinki-NLP/opus-mt-es-en" if self.target_lang == "en"
                else "Helsinki-NLP/opus-mt-en-es"
            )
            self.on_status(
                f"Cargando modelo de traducción ({model_name.split('/')[-1]}) "
                "(la primera vez descarga ~300MB)..."
            )
            from transformers import MarianMTModel, MarianTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tok = MarianTokenizer.from_pretrained(model_name, cache_dir=models_dir)
            model = MarianMTModel.from_pretrained(model_name, cache_dir=models_dir)
            if device == "cuda":
                model = model.to("cuda")

            def translate_batch(texts):
                texts = [t for t in texts if t.strip()]
                if not texts:
                    return {}
                inputs = tok(texts, return_tensors="pt", padding=True, truncation=True)
                if device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    generated = model.generate(**inputs, max_length=512)
                decoded = [tok.decode(t, skip_special_tokens=True).strip() for t in generated]
                return dict(zip(texts, decoded))

            site_dir = Path(self.site_dir)
            html_files = list(site_dir.rglob("*.html")) + list(site_dir.rglob("*.htm"))
            if not html_files:
                raise RuntimeError("No se encontraron archivos .html en esa carpeta.")

            suffix = "traducido_ingles" if self.target_lang == "en" else "traducido_espanol"
            out_root = Path(self.out_dir) / f"{site_dir.name}_{suffix}"
            out_root.mkdir(parents=True, exist_ok=True)

            skip_tags = {"script", "style", "noscript", "code", "pre"}
            total = len(html_files)

            for i, html_file in enumerate(html_files, start=1):
                rel = html_file.relative_to(site_dir)
                self.on_status(f"Traduciendo página {i} de {total}: {rel}")

                raw = html_file.read_text(encoding="utf-8", errors="ignore")
                soup = BeautifulSoup(raw, "html.parser")

                text_nodes = [
                    node for node in soup.find_all(string=True)
                    if node.parent.name not in skip_tags and node.strip()
                ]
                unique_texts = list({node.strip() for node in text_nodes})

                translations = {}
                batch_size = 16
                for b in range(0, len(unique_texts), batch_size):
                    translations.update(translate_batch(unique_texts[b:b + batch_size]))

                for node in text_nodes:
                    original = node.strip()
                    if original in translations:
                        node.replace_with(translations[original])

                out_path = out_root / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(str(soup), encoding="utf-8")

                self.on_progress_pct(min(100, int(i / total * 100)))

            self.on_progress_pct(100)
            self.on_done(total, str(out_root))
        except Exception as e:
            self.on_error(f"{e}\n\n{traceback.format_exc()}")


class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command, colors, panel_bg,
                 width=150, height=36, use_accent=True, **kwargs):
        super().__init__(parent, width=width, height=height, bg=panel_bg,
                          highlightthickness=0, **kwargs)
        self.command = command
        self.colors = colors
        self.width = width
        self.height = height
        self.text = text
        self.use_accent = use_accent
        self.command_enabled = True
        self._draw(self._fill_color())
        self.bind("<Button-1>", lambda e: self._on_click())
        self.bind("<Enter>", lambda e: self._draw(self._hover_color()))
        self.bind("<Leave>", lambda e: self._draw(self._fill_color()))

    def _fill_color(self):
        if not self.command_enabled:
            return self.colors["btn_disabled"]
        return self.colors["accent"] if self.use_accent else self.colors["surface"]

    def _hover_color(self):
        if not self.command_enabled:
            return self.colors["btn_disabled"]
        return self.colors["accent_hover"] if self.use_accent else self.colors["track"]

    def _round_rect(self, x1, y1, x2, y2, r, **kw):
        points = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
                  x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        return self.create_polygon(points, smooth=True, **kw)

    def _draw(self, color):
        self.delete("all")
        self._round_rect(2, 2, self.width - 2, self.height - 2, 12, fill=color, outline="")
        text_color = "white" if self.use_accent else self.colors["text"]
        self.create_text(self.width / 2, self.height / 2, text=self.text,
                          fill=text_color, font=(FONT_FAMILY, 10, "bold"))

    def set_enabled(self, enabled: bool):
        self.command_enabled = enabled
        self._draw(self._fill_color())

    def set_panel_bg(self, panel_bg):
        self.configure(bg=panel_bg)
        self._draw(self._fill_color())

    def _on_click(self):
        if self.command_enabled and self.command:
            self.command()


# --------------------------------------------------------------------------
# Interfaz gráfica principal
# --------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # Con la ventana ya reconocida como DPI-aware por Windows, hay que
        # decirle a Tk el factor de escala real de la pantalla, o los
        # textos/widgets quedarían diminutos en monitores de alta densidad.
        try:
            dpi_scale = self.winfo_fpixels("1i") / 72.0
            self.tk.call("tk", "scaling", dpi_scale)
        except Exception:
            pass

        self.title(APP_TITLE)
        self.geometry("1150x720")
        self.minsize(900, 560)

        self.theme_name = "dark"
        self.colors = DARK

        self.filepath = tk.StringVar()
        self.out_dir = tk.StringVar(value=str(Path.home() / "Transcripciones"))
        self.model_size = tk.StringVar(value="large-v3")
        self.language_label = tk.StringVar(value="Detectar automáticamente")
        self.use_gpu = tk.BooleanVar(value=True)
        # (la traducción ahora es un paso separado, ver sección "Traducción")
        self.selected_audio_track = None
        self.audio_tracks_info = []
        self.progress_pct = tk.IntVar(value=0)
        self._last_paths = None
        self._translation_cache = None
        self.doc_progress_pct = tk.IntVar(value=0)
        self.doc_filepath = tk.StringVar()
        self.website_url = tk.StringVar()
        self.website_out_dir = tk.StringVar(value=str(Path.home() / "SitiosDescargados"))
        self.website_page_limit = tk.StringVar(value="50")
        self.website_translate_dir = tk.StringVar()
        self.website_progress_pct = tk.IntVar(value=0)

        # Registro de widgets "planos" (no-ttk) que hay que retematizar
        self._bg_frames = []       # frames con color de fondo tipo 'bg'
        self._panel_frames = []    # frames con color de fondo tipo 'panel'
        self._border_frames = []   # frames usados como borde de tarjeta
        self._surface_widgets = []  # Text widgets tipo 'surface' (blanco/gris oscuro)
        self._log_widgets = []
        self._round_buttons_accent = []
        self._round_buttons_plain = []
        self._round_buttons_plain_panel = []
        self._canvases_bg = []      # canvases cuyo fondo es 'bg' (ej. el punto decorativo)

        self.configure(bg=self.colors["bg"])
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self._build_ui()
        self._apply_theme()

    # ---------------------------------------------------------------
    def _card(self, parent):
        outer = tk.Frame(parent, bg=self.colors["border"])
        inner = tk.Frame(outer, bg=self.colors["panel"])
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        self._border_frames.append(outer)
        self._panel_frames.append(inner)
        return outer, inner

    def _panel_frame(self, parent):
        f = tk.Frame(parent, bg=self.colors["panel"])
        self._panel_frames.append(f)
        return f

    # ---------------------------------------------------------------
    def _build_ui(self):
        header = tk.Frame(self, bg=self.colors["bg"])
        header.pack(fill="x", padx=20, pady=(16, 8))
        self._bg_frames.append(header)

        dot = tk.Canvas(header, width=14, height=14, bg=self.colors["bg"], highlightthickness=0)
        dot.create_oval(1, 1, 13, 13, fill=self.colors["accent"], outline="")
        dot.pack(side="left", padx=(0, 8))
        self._canvases_bg.append(dot)

        self.title_label = tk.Label(header, text=APP_TITLE, bg=self.colors["bg"],
                                     fg=self.colors["text"], font=(FONT_FAMILY, 15, "bold"))
        self.title_label.pack(side="left")

        self.theme_btn = RoundedButton(header, "🌙 Modo oscuro", self.toggle_theme,
                                        self.colors, self.colors["bg"], width=150, height=32,
                                        use_accent=False)
        self.theme_btn.pack(side="right")
        self._round_buttons_plain.append(self.theme_btn)

        self.notebook = ttk.Notebook(self, style="App.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        tab_transcribe = tk.Frame(self.notebook, bg=self.colors["bg"])
        tab_translate = tk.Frame(self.notebook, bg=self.colors["bg"])
        tab_website = tk.Frame(self.notebook, bg=self.colors["bg"])
        self._bg_frames.append(tab_transcribe)
        self._bg_frames.append(tab_translate)
        self._bg_frames.append(tab_website)
        self.notebook.add(tab_transcribe, text="  Transcripción  ")
        self.notebook.add(tab_translate, text="  Traducción  ")
        self.notebook.add(tab_website, text="  Sitio web  ")

        paned = ttk.PanedWindow(tab_transcribe, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=0, pady=(12, 0))

        left_panel = self._build_transcript_panel(paned)
        right_panel = self._build_controls_panel(paned)
        paned.add(left_panel, weight=3)
        paned.add(right_panel, weight=2)

        self._build_translation_tab(tab_translate)
        self._build_website_tab(tab_website)

    # ---------------------------------------------------------------
    def _build_transcript_panel(self, parent):
        outer, card = self._card(parent)

        top = self._panel_frame(card)
        top.pack(fill="x", padx=16, pady=(14, 6))

        self.transcript_title = tk.Label(top, text="Transcripción en vivo", bg=self.colors["panel"],
                                          fg=self.colors["text"], font=(FONT_FAMILY, 10, "bold"))
        self.transcript_title.pack(side="left")

        self.pct_label = tk.Label(top, text="0%", bg=self.colors["panel"],
                                   fg=self.colors["accent"], font=(FONT_FAMILY, 12, "bold"))
        self.pct_label.pack(side="right")

        self.time_label = tk.Label(top, text="", bg=self.colors["panel"],
                                    fg=self.colors["text_muted"], font=(FONT_FAMILY, 9))
        self.time_label.pack(side="right", padx=(0, 12))

        self.style.configure("Coral.Horizontal.TProgressbar", troughcolor=self.colors["track"],
                              background=self.colors["accent"], bordercolor=self.colors["track"],
                              lightcolor=self.colors["accent"], darkcolor=self.colors["accent"],
                              thickness=10)
        self.progress = ttk.Progressbar(card, mode="determinate", maximum=100,
                                         variable=self.progress_pct,
                                         style="Coral.Horizontal.TProgressbar")
        self.progress.pack(fill="x", padx=16, pady=(0, 10))

        self.hint_label = tk.Label(
            card, text="Puedes editar el texto libremente mientras se genera o al finalizar.",
            bg=self.colors["panel"], fg=self.colors["text_muted"], font=(FONT_FAMILY, 9)
        )
        self.hint_label.pack(anchor="w", padx=16, pady=(0, 6))

        text_frame = self._panel_frame(card)
        text_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        self.transcript_text = tk.Text(
            text_frame, wrap="word", bg=self.colors["surface"], fg=self.colors["text"],
            font=(FONT_FAMILY, 11), relief="flat", padx=12, pady=12,
            insertbackground=self.colors["text"], yscrollcommand=scrollbar.set,
            undo=True, autoseparators=True, maxundo=-1,
        )
        self.transcript_text.pack(fill="both", expand=True)
        scrollbar.config(command=self.transcript_text.yview)
        self.transcript_text.tag_configure("timestamp", foreground=self.colors["accent"],
                                            font=(FONT_FAMILY, 9, "bold"))
        self._surface_widgets.append(self.transcript_text)
        self._bind_text_edit_shortcuts(self.transcript_text)
        self._add_text_context_menu(self.transcript_text)

        btn_row = self._panel_frame(card)
        btn_row.pack(fill="x", padx=16, pady=(0, 14))

        self.btn_save = RoundedButton(btn_row, "Guardar cambios", self.save_edited_transcript,
                                       self.colors, self.colors["panel"], width=160, height=34)
        self.btn_save.pack(side="left")
        self._round_buttons_accent.append(self.btn_save)

        self.save_hint = tk.Label(btn_row, text="  Sobrescribe los archivos .txt y .md con tus ediciones",
                                   bg=self.colors["panel"], fg=self.colors["text_muted"],
                                   font=(FONT_FAMILY, 9))
        self.save_hint.pack(side="left", padx=8)

        return outer

    # ---------------------------------------------------------------
    def _build_controls_panel(self, parent):
        container = tk.Frame(parent, bg=self.colors["bg"])
        self._bg_frames.append(container)

        outer, card = self._card(container)
        outer.pack(fill="both", expand=True, pady=(0, 10))
        pad = {"padx": 16, "pady": 6}

        self.config_title = tk.Label(card, text="Configuración", bg=self.colors["panel"],
                                      fg=self.colors["text"], font=(FONT_FAMILY, 10, "bold"))
        self.config_title.pack(anchor="w", padx=16, pady=(14, 10))

        self._field_labels = []

        def field_label(text, parent_widget):
            lbl = tk.Label(parent_widget, text=text, bg=self.colors["panel"],
                            fg=self.colors["text"], font=(FONT_FAMILY, 10))
            lbl.pack(anchor="w", **pad)
            self._field_labels.append(lbl)
            return lbl

        field_label("Archivo de audio/video", card)
        file_row = self._panel_frame(card)
        file_row.pack(fill="x", padx=16)
        file_entry = ttk.Entry(file_row, textvariable=self.filepath)
        file_entry.pack(side="left", fill="x", expand=True, ipady=3)
        self._add_entry_context_menu(file_entry)
        b1 = RoundedButton(file_row, "Examinar", self.pick_file, self.colors, self.colors["panel"],
                            width=100, height=30)
        b1.pack(side="left", padx=(8, 0))
        self._round_buttons_accent.append(b1)

        track_row = self._panel_frame(card)
        track_row.pack(fill="x", padx=16, pady=(2, 0))
        self.track_info_label = tk.Label(
            track_row, text="", bg=self.colors["panel"], fg=self.colors["text_muted"],
            font=(FONT_FAMILY, 8), wraplength=280, justify="left"
        )
        self.track_info_label.pack(side="left")
        self.btn_change_track = RoundedButton(
            track_row, "Cambiar pista", self.open_track_selector,
            self.colors, self.colors["panel"], width=110, height=24, use_accent=False
        )
        # Solo se muestra cuando hay más de una pista de audio detectada
        self._round_buttons_plain_panel.append(self.btn_change_track)

        field_label("Carpeta de salida", card)
        out_row = self._panel_frame(card)
        out_row.pack(fill="x", padx=16)
        out_entry = ttk.Entry(out_row, textvariable=self.out_dir)
        out_entry.pack(side="left", fill="x", expand=True, ipady=3)
        self._add_entry_context_menu(out_entry)
        b2 = RoundedButton(out_row, "Elegir", self.pick_out_dir, self.colors, self.colors["panel"],
                            width=100, height=30)
        b2.pack(side="left", padx=(8, 0))
        self._round_buttons_accent.append(b2)

        field_label("Modelo", card)
        model_row = self._panel_frame(card)
        model_row.pack(fill="x", padx=16)
        ttk.Combobox(model_row, textvariable=self.model_size, values=MODEL_OPTIONS,
                     state="readonly").pack(side="left", fill="x", expand=True)
        b_models = RoundedButton(model_row, "Gestionar modelos", self.open_model_manager,
                                  self.colors, self.colors["panel"], width=150, height=30,
                                  use_accent=False)
        b_models.pack(side="left", padx=(8, 0))
        self._round_buttons_plain_panel.append(b_models)

        field_label("Idioma", card)
        ttk.Combobox(card, textvariable=self.language_label,
                     values=list(LANGUAGE_MAP.keys()), state="readonly").pack(fill="x", padx=16)

        gpu_row = self._panel_frame(card)
        gpu_row.pack(fill="x", padx=16, pady=(14, 4))
        ttk.Checkbutton(gpu_row, text="Usar GPU (NVIDIA/CUDA) si está disponible",
                         variable=self.use_gpu).pack(anchor="w")

        run_row = self._panel_frame(card)
        run_row.pack(fill="x", padx=16, pady=(18, 10))
        self.btn_run = RoundedButton(run_row, "Transcribir", self.start_transcription,
                                      self.colors, self.colors["panel"], width=200, height=40)
        self.btn_run.pack(anchor="w")
        self._round_buttons_accent.append(self.btn_run)

        self.status_var = tk.StringVar(value="Listo.")
        self.status_label = tk.Label(card, textvariable=self.status_var, bg=self.colors["panel"],
                                      fg=self.colors["text_muted"], font=(FONT_FAMILY, 9),
                                      wraplength=320, justify="left")
        self.status_label.pack(anchor="w", padx=16, pady=(4, 10))

        self.log_title = tk.Label(card, text="Registro", bg=self.colors["panel"],
                                   fg=self.colors["text"], font=(FONT_FAMILY, 10, "bold"))
        self.log_title.pack(anchor="w", padx=16, pady=(6, 4))

        log_frame = self._panel_frame(card)
        log_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        log_scroll = ttk.Scrollbar(log_frame)
        log_scroll.pack(side="right", fill="y")
        self.log_text = tk.Text(log_frame, height=8, bg=self.colors["log_bg"],
                                 fg=self.colors["text_muted"], font=(FONT_FAMILY, 9),
                                 relief="flat", padx=8, pady=8,
                                 yscrollcommand=log_scroll.set, state="disabled")
        self.log_text.pack(fill="both", expand=True)
        log_scroll.config(command=self.log_text.yview)
        self._log_widgets.append(self.log_text)

        # ---- Tarjeta aparte: Traducción (paso opcional, después de transcribir) ----
        trans_outer, trans_card = self._card(container)
        trans_outer.pack(fill="x")

        self.translate_title = tk.Label(
            trans_card, text="Traducción (paso aparte)", bg=self.colors["panel"],
            fg=self.colors["text"], font=(FONT_FAMILY, 10, "bold")
        )
        self.translate_title.pack(anchor="w", padx=16, pady=(14, 4))

        self.translate_hint_label = tk.Label(
            trans_card,
            text=("Se activa cuando termina una transcripción. Genera archivos nuevos "
                  "aparte, sin tocar los originales."),
            bg=self.colors["panel"], fg=self.colors["text_muted"], font=(FONT_FAMILY, 8),
            wraplength=320, justify="left"
        )
        self.translate_hint_label.pack(anchor="w", padx=16, pady=(0, 10))

        trans_btn_row = self._panel_frame(trans_card)
        trans_btn_row.pack(fill="x", padx=16, pady=(0, 16))

        self.btn_translate_en = RoundedButton(
            trans_btn_row, "Traducir a inglés", self.start_translate_to_english,
            self.colors, self.colors["panel"], width=150, height=34, use_accent=False
        )
        self.btn_translate_en.pack(side="left")
        self.btn_translate_en.set_enabled(False)
        self._round_buttons_plain_panel.append(self.btn_translate_en)

        self.btn_translate_es = RoundedButton(
            trans_btn_row, "Traducir a español", self.start_translate_to_spanish,
            self.colors, self.colors["panel"], width=150, height=34, use_accent=False
        )
        self.btn_translate_es.pack(side="left", padx=(8, 0))
        self.btn_translate_es.set_enabled(False)
        self._round_buttons_plain_panel.append(self.btn_translate_es)

        return container

    # ---------------------------------------------------------------
    # Pestaña de Traducción: independiente, para cualquier documento
    # de texto, sin pasar por la transcripción.
    # ---------------------------------------------------------------
    def _build_translation_tab(self, parent):
        paned = ttk.PanedWindow(parent, orient="horizontal")
        paned.pack(fill="both", expand=True, pady=(12, 0))

        # ---- Panel izquierdo: contenido del documento ----
        left_outer, left_card = self._card(paned)

        top = self._panel_frame(left_card)
        top.pack(fill="x", padx=16, pady=(14, 6))
        self.doc_title_label = tk.Label(
            top, text="Documento", bg=self.colors["panel"], fg=self.colors["text"],
            font=(FONT_FAMILY, 10, "bold")
        )
        self.doc_title_label.pack(side="left")
        self.doc_pct_label = tk.Label(
            top, text="0%", bg=self.colors["panel"], fg=self.colors["accent"],
            font=(FONT_FAMILY, 12, "bold")
        )
        self.doc_pct_label.pack(side="right")

        self.doc_progress = ttk.Progressbar(
            left_card, mode="determinate", maximum=100, variable=self.doc_progress_pct,
            style="Coral.Horizontal.TProgressbar"
        )
        self.doc_progress.pack(fill="x", padx=16, pady=(0, 10))

        self.doc_hint_label = tk.Label(
            left_card,
            text="Cargá un archivo o pegá/escribí texto directamente acá. Editable con Ctrl+Z y clic derecho.",
            bg=self.colors["panel"], fg=self.colors["text_muted"], font=(FONT_FAMILY, 9)
        )
        self.doc_hint_label.pack(anchor="w", padx=16, pady=(0, 6))

        doc_text_frame = self._panel_frame(left_card)
        doc_text_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        doc_scrollbar = ttk.Scrollbar(doc_text_frame)
        doc_scrollbar.pack(side="right", fill="y")

        self.doc_text = tk.Text(
            doc_text_frame, wrap="word", bg=self.colors["surface"], fg=self.colors["text"],
            font=(FONT_FAMILY, 11), relief="flat", padx=12, pady=12,
            insertbackground=self.colors["text"], yscrollcommand=doc_scrollbar.set,
            undo=True, autoseparators=True, maxundo=-1,
        )
        self.doc_text.pack(fill="both", expand=True)
        doc_scrollbar.config(command=self.doc_text.yview)
        self._surface_widgets.append(self.doc_text)
        self._bind_text_edit_shortcuts(self.doc_text)
        self._add_text_context_menu(self.doc_text)

        doc_btn_row = self._panel_frame(left_card)
        doc_btn_row.pack(fill="x", padx=16, pady=(0, 14))
        self.btn_save_translation = RoundedButton(
            doc_btn_row, "Guardar como...", self.save_translation_as,
            self.colors, self.colors["panel"], width=160, height=34
        )
        self.btn_save_translation.pack(side="left")
        self._round_buttons_accent.append(self.btn_save_translation)

        paned.add(left_outer, weight=3)

        # ---- Panel derecho: controles de traducción ----
        right_outer, right_card = self._card(paned)

        self.doc_config_title = tk.Label(
            right_card, text="Traducir documento", bg=self.colors["panel"],
            fg=self.colors["text"], font=(FONT_FAMILY, 10, "bold")
        )
        self.doc_config_title.pack(anchor="w", padx=16, pady=(14, 10))

        self._doc_field_labels = []

        def doc_field_label(text):
            lbl = tk.Label(right_card, text=text, bg=self.colors["panel"],
                            fg=self.colors["text"], font=(FONT_FAMILY, 10))
            lbl.pack(anchor="w", padx=16, pady=6)
            self._doc_field_labels.append(lbl)
            return lbl

        doc_field_label("Archivo (.txt, .md, .srt) — opcional")
        doc_file_row = self._panel_frame(right_card)
        doc_file_row.pack(fill="x", padx=16)
        doc_file_entry = ttk.Entry(doc_file_row, textvariable=self.doc_filepath)
        doc_file_entry.pack(side="left", fill="x", expand=True, ipady=3)
        self._add_entry_context_menu(doc_file_entry)
        b_load = RoundedButton(doc_file_row, "Cargar", self.load_document_for_translation,
                                self.colors, self.colors["panel"], width=100, height=30)
        b_load.pack(side="left", padx=(8, 0))
        self._round_buttons_accent.append(b_load)

        self.doc_load_hint = tk.Label(
            right_card,
            text="No hace falta cargar un archivo: también podés pegar o escribir texto directo en el panel de la izquierda.",
            bg=self.colors["panel"], fg=self.colors["text_muted"], font=(FONT_FAMILY, 8),
            wraplength=290, justify="left"
        )
        self.doc_load_hint.pack(anchor="w", padx=16, pady=(4, 16))

        doc_btn_col = self._panel_frame(right_card)
        doc_btn_col.pack(fill="x", padx=16, pady=(0, 10))

        self.btn_doc_translate_en = RoundedButton(
            doc_btn_col, "Traducir a inglés", lambda: self.start_document_translation("en"),
            self.colors, self.colors["panel"], width=220, height=38
        )
        self.btn_doc_translate_en.pack(anchor="w", pady=(0, 8))
        self._round_buttons_accent.append(self.btn_doc_translate_en)

        self.btn_doc_translate_es = RoundedButton(
            doc_btn_col, "Traducir a español", lambda: self.start_document_translation("es"),
            self.colors, self.colors["panel"], width=220, height=38
        )
        self.btn_doc_translate_es.pack(anchor="w")
        self._round_buttons_accent.append(self.btn_doc_translate_es)

        self.doc_dir_hint = tk.Label(
            right_card,
            text=("\"A inglés\" asume que el texto está en español. \"A español\" asume que "
                  "está en inglés. Descarga un modelo local (~300MB) la primera vez que uses "
                  "cada dirección."),
            bg=self.colors["panel"], fg=self.colors["text_muted"], font=(FONT_FAMILY, 8),
            wraplength=290, justify="left"
        )
        self.doc_dir_hint.pack(anchor="w", padx=16, pady=(10, 16))

        self.doc_status_var = tk.StringVar(value="Listo.")
        self.doc_status_label = tk.Label(
            right_card, textvariable=self.doc_status_var, bg=self.colors["panel"],
            fg=self.colors["text_muted"], font=(FONT_FAMILY, 9), wraplength=290, justify="left"
        )
        self.doc_status_label.pack(anchor="w", padx=16, pady=(4, 16))

        paned.add(right_outer, weight=2)
        return paned

    def load_document_for_translation(self):
        path = filedialog.askopenfilename(
            title="Selecciona un documento de texto",
            filetypes=[
                ("Texto/Markdown/Subtítulos", "*.txt *.md *.srt"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not path:
            return
        try:
            raw = Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"No se pudo leer el archivo:\n\n{e}")
            return

        if path.lower().endswith(".srt"):
            lines = []
            for line in raw.splitlines():
                stripped = line.strip()
                if not stripped or stripped.isdigit() or "-->" in stripped:
                    continue
                lines.append(stripped)
            content = "\n".join(lines)
        else:
            content = raw

        self.doc_filepath.set(path)
        self.doc_text.delete("1.0", "end")
        self.doc_text.insert("1.0", content)
        self.doc_status_var.set(f"Documento cargado: {Path(path).name}")

    def start_document_translation(self, target_lang):
        text = self.doc_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning(APP_TITLE, "Cargá un documento o escribí/pegá texto primero.")
            return

        base_name = Path(self.doc_filepath.get()).stem if self.doc_filepath.get() else "documento"
        out_dir = self.out_dir.get() or str(Path.home() / "Transcripciones")

        self.btn_doc_translate_en.set_enabled(False)
        self.btn_doc_translate_es.set_enabled(False)
        self.doc_progress_pct.set(0)
        self.doc_pct_label.configure(text="0%")
        self.doc_status_var.set("Iniciando traducción...")

        worker = TextDocumentTranslateWorker(
            text=text, target_lang=target_lang, out_dir=out_dir, base_name=base_name,
            on_status=lambda msg: self.after(0, self.doc_status_var.set, msg),
            on_progress_pct=lambda pct: self.after(0, self._update_doc_progress, pct),
            on_done=lambda translated_text, out_path: self.after(
                0, self.on_document_translation_done, translated_text, out_path
            ),
            on_error=lambda err: self.after(0, self.on_document_translation_error, err),
        )
        worker.start()

    def _update_doc_progress(self, pct):
        self.doc_progress_pct.set(pct)
        self.doc_pct_label.configure(text=f"{pct}%")

    def on_document_translation_done(self, translated_text, out_path):
        self.btn_doc_translate_en.set_enabled(True)
        self.btn_doc_translate_es.set_enabled(True)
        self.doc_text.delete("1.0", "end")
        self.doc_text.insert("1.0", translated_text)
        self.doc_status_var.set(f"¡Traducción completada! Guardada en:\n{out_path}")
        messagebox.showinfo(APP_TITLE, f"Traducción completada.\n\nGuardada en:\n{out_path}")

    def on_document_translation_error(self, error_message):
        self.btn_doc_translate_en.set_enabled(True)
        self.btn_doc_translate_es.set_enabled(True)
        self.doc_status_var.set("Ocurrió un error durante la traducción.")
        messagebox.showerror(APP_TITLE, f"Error durante la traducción:\n\n{error_message[:500]}")

    def save_translation_as(self):
        content = self.doc_text.get("1.0", "end").strip()
        if not content:
            messagebox.showwarning(APP_TITLE, "No hay texto para guardar.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar como...", defaultextension=".txt",
            filetypes=[("Texto", "*.txt"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        Path(path).write_text(content, encoding="utf-8")
        messagebox.showinfo(APP_TITLE, "Archivo guardado correctamente.")

    # ---------------------------------------------------------------
    # Pestaña de Sitio web: descargar un sitio y, aparte, traducir uno
    # ya descargado (con este mismo motor de traducción local).
    # ---------------------------------------------------------------
    def _build_website_tab(self, parent):
        container = tk.Frame(parent, bg=self.colors["bg"])
        container.pack(fill="both", expand=True, pady=(12, 0))

        # ---- Tarjeta: Descargar sitio ----
        dl_outer, dl_card = self._card(container)
        dl_outer.pack(fill="x", pady=(0, 10))

        self.website_dl_title = tk.Label(
            dl_card, text="Descargar sitio web completo", bg=self.colors["panel"],
            fg=self.colors["text"], font=(FONT_FAMILY, 10, "bold")
        )
        self.website_dl_title.pack(anchor="w", padx=16, pady=(14, 4))
        self.website_dl_hint = tk.Label(
            dl_card,
            text="Descarga las páginas HTML del sitio (mismo dominio, siguiendo enlaces internos). No incluye imágenes/CSS/JS.",
            bg=self.colors["panel"], fg=self.colors["text_muted"], font=(FONT_FAMILY, 8),
            wraplength=700, justify="left"
        )
        self.website_dl_hint.pack(anchor="w", padx=16, pady=(0, 10))

        url_row = self._panel_frame(dl_card)
        url_row.pack(fill="x", padx=16, pady=4)
        tk.Label(url_row, text="URL:", bg=self.colors["panel"], fg=self.colors["text"],
                 font=(FONT_FAMILY, 10), width=14, anchor="w").pack(side="left")
        url_entry = ttk.Entry(url_row, textvariable=self.website_url)
        url_entry.pack(side="left", fill="x", expand=True)
        self._add_entry_context_menu(url_entry)

        dlout_row = self._panel_frame(dl_card)
        dlout_row.pack(fill="x", padx=16, pady=4)
        tk.Label(dlout_row, text="Carpeta destino:", bg=self.colors["panel"], fg=self.colors["text"],
                 font=(FONT_FAMILY, 10), width=14, anchor="w").pack(side="left")
        dlout_entry = ttk.Entry(dlout_row, textvariable=self.website_out_dir)
        dlout_entry.pack(side="left", fill="x", expand=True)
        self._add_entry_context_menu(dlout_entry)
        b_dlout = RoundedButton(dlout_row, "Elegir", self.pick_website_out_dir,
                                 self.colors, self.colors["panel"], width=90, height=28)
        b_dlout.pack(side="left", padx=(8, 0))
        self._round_buttons_accent.append(b_dlout)

        limit_row = self._panel_frame(dl_card)
        limit_row.pack(fill="x", padx=16, pady=4)
        tk.Label(limit_row, text="Máx. de páginas:", bg=self.colors["panel"], fg=self.colors["text"],
                 font=(FONT_FAMILY, 10), width=14, anchor="w").pack(side="left")
        limit_entry = ttk.Entry(limit_row, textvariable=self.website_page_limit, width=10)
        limit_entry.pack(side="left")
        self._add_entry_context_menu(limit_entry)

        dl_btn_row = self._panel_frame(dl_card)
        dl_btn_row.pack(fill="x", padx=16, pady=(10, 6))
        self.btn_website_download = RoundedButton(
            dl_btn_row, "Descargar sitio", self.start_website_download,
            self.colors, self.colors["panel"], width=180, height=36
        )
        self.btn_website_download.pack(side="left")
        self._round_buttons_accent.append(self.btn_website_download)

        self.website_dl_pct_label = tk.Label(
            dl_btn_row, text="0%", bg=self.colors["panel"], fg=self.colors["accent"],
            font=(FONT_FAMILY, 11, "bold")
        )
        self.website_dl_pct_label.pack(side="left", padx=12)

        self.website_dl_progress = ttk.Progressbar(
            dl_card, mode="determinate", maximum=100, variable=self.website_progress_pct,
            style="Coral.Horizontal.TProgressbar"
        )
        self.website_dl_progress.pack(fill="x", padx=16, pady=(0, 8))

        self.website_dl_status_var = tk.StringVar(value="Listo.")
        self.website_dl_status_label = tk.Label(
            dl_card, textvariable=self.website_dl_status_var, bg=self.colors["panel"],
            fg=self.colors["text_muted"], font=(FONT_FAMILY, 9), wraplength=700,
            justify="left", anchor="w"
        )
        self.website_dl_status_label.pack(anchor="w", padx=16, pady=(0, 16), fill="x")

        # ---- Tarjeta: Traducir sitio ya descargado ----
        tr_outer, tr_card = self._card(container)
        tr_outer.pack(fill="x")

        self.website_tr_title = tk.Label(
            tr_card, text="Traducir sitio ya descargado", bg=self.colors["panel"],
            fg=self.colors["text"], font=(FONT_FAMILY, 10, "bold")
        )
        self.website_tr_title.pack(anchor="w", padx=16, pady=(14, 4))
        self.website_tr_hint = tk.Label(
            tr_card,
            text="Elegí la carpeta de un sitio ya descargado (con esta app u otra herramienta) y traducí su contenido.",
            bg=self.colors["panel"], fg=self.colors["text_muted"], font=(FONT_FAMILY, 8),
            wraplength=700, justify="left"
        )
        self.website_tr_hint.pack(anchor="w", padx=16, pady=(0, 10))

        trdir_row = self._panel_frame(tr_card)
        trdir_row.pack(fill="x", padx=16, pady=4)
        tk.Label(trdir_row, text="Carpeta del sitio:", bg=self.colors["panel"], fg=self.colors["text"],
                 font=(FONT_FAMILY, 10), width=14, anchor="w").pack(side="left")
        trdir_entry = ttk.Entry(trdir_row, textvariable=self.website_translate_dir)
        trdir_entry.pack(side="left", fill="x", expand=True)
        self._add_entry_context_menu(trdir_entry)
        b_trdir = RoundedButton(trdir_row, "Elegir", self.pick_website_translate_dir,
                                 self.colors, self.colors["panel"], width=90, height=28)
        b_trdir.pack(side="left", padx=(8, 0))
        self._round_buttons_accent.append(b_trdir)

        tr_btn_row = self._panel_frame(tr_card)
        tr_btn_row.pack(fill="x", padx=16, pady=(10, 6))
        self.btn_website_translate_en = RoundedButton(
            tr_btn_row, "Traducir sitio a inglés", lambda: self.start_website_translation("en"),
            self.colors, self.colors["panel"], width=190, height=36
        )
        self.btn_website_translate_en.pack(side="left")
        self._round_buttons_accent.append(self.btn_website_translate_en)

        self.btn_website_translate_es = RoundedButton(
            tr_btn_row, "Traducir sitio a español", lambda: self.start_website_translation("es"),
            self.colors, self.colors["panel"], width=190, height=36
        )
        self.btn_website_translate_es.pack(side="left", padx=(8, 0))
        self._round_buttons_accent.append(self.btn_website_translate_es)

        self.website_tr_pct_label = tk.Label(
            tr_btn_row, text="0%", bg=self.colors["panel"], fg=self.colors["accent"],
            font=(FONT_FAMILY, 11, "bold")
        )
        self.website_tr_pct_label.pack(side="left", padx=12)

        self.website_tr_progress_var = tk.IntVar(value=0)
        self.website_tr_progress = ttk.Progressbar(
            tr_card, mode="determinate", maximum=100, variable=self.website_tr_progress_var,
            style="Coral.Horizontal.TProgressbar"
        )
        self.website_tr_progress.pack(fill="x", padx=16, pady=(0, 8))

        self.website_tr_status_var = tk.StringVar(value="Listo.")
        self.website_tr_status_label = tk.Label(
            tr_card, textvariable=self.website_tr_status_var, bg=self.colors["panel"],
            fg=self.colors["text_muted"], font=(FONT_FAMILY, 9), wraplength=700,
            justify="left", anchor="w"
        )
        self.website_tr_status_label.pack(anchor="w", padx=16, pady=(0, 16), fill="x")

        return container

    def pick_website_out_dir(self):
        path = filedialog.askdirectory(title="Selecciona carpeta de destino")
        if path:
            self.website_out_dir.set(path)

    def pick_website_translate_dir(self):
        path = filedialog.askdirectory(title="Selecciona la carpeta del sitio a traducir")
        if path:
            self.website_translate_dir.set(path)

    def start_website_download(self):
        url = self.website_url.get().strip()
        if not url:
            messagebox.showwarning(APP_TITLE, "Escribí primero la URL del sitio.")
            return
        try:
            page_limit = int(self.website_page_limit.get().strip())
        except ValueError:
            messagebox.showwarning(APP_TITLE, "El máximo de páginas debe ser un número.")
            return

        self.btn_website_download.set_enabled(False)
        self.website_progress_pct.set(0)
        self.website_dl_pct_label.configure(text="0%")
        self.website_dl_status_var.set("Iniciando descarga...")

        worker = WebsiteDownloaderWorker(
            start_url=url, out_dir=self.website_out_dir.get(), page_limit=page_limit,
            on_status=lambda msg: self.after(0, self._update_website_dl_status, msg),
            on_progress_pct=lambda pct: self.after(0, self._update_website_dl_progress, pct),
            on_done=lambda count, out_dir: self.after(0, self.on_website_download_done, count, out_dir),
            on_error=lambda err: self.after(0, self.on_website_download_error, err),
        )
        worker.start()

    def _update_website_dl_status(self, msg):
        self.website_dl_status_var.set(msg)

    def _update_website_dl_progress(self, pct):
        self.website_progress_pct.set(pct)
        self.website_dl_pct_label.configure(text=f"{pct}%")

    def on_website_download_done(self, count, out_dir):
        self.btn_website_download.set_enabled(True)
        self.website_dl_status_var.set(f"¡Listo! Se descargaron {count} página(s) en: {out_dir}")
        self.website_translate_dir.set(out_dir)
        messagebox.showinfo(APP_TITLE, f"Se descargaron {count} página(s).\n\nGuardadas en:\n{out_dir}")

    def on_website_download_error(self, error_message):
        self.btn_website_download.set_enabled(True)
        self.website_dl_status_var.set("Ocurrió un error durante la descarga.")
        messagebox.showerror(APP_TITLE, f"Error al descargar el sitio:\n\n{error_message[:500]}")

    def start_website_translation(self, target_lang):
        site_dir = self.website_translate_dir.get().strip()
        if not site_dir or not os.path.isdir(site_dir):
            messagebox.showwarning(APP_TITLE, "Elegí una carpeta válida de un sitio ya descargado.")
            return

        self.btn_website_translate_en.set_enabled(False)
        self.btn_website_translate_es.set_enabled(False)
        self.website_tr_progress_var.set(0)
        self.website_tr_pct_label.configure(text="0%")
        self.website_tr_status_var.set("Iniciando traducción del sitio...")

        worker = WebsiteTranslateWorker(
            site_dir=site_dir, target_lang=target_lang,
            out_dir=self.website_out_dir.get() or str(Path.home() / "SitiosDescargados"),
            on_status=lambda msg: self.after(0, self._update_website_tr_status, msg),
            on_progress_pct=lambda pct: self.after(0, self._update_website_tr_progress, pct),
            on_done=lambda count, out_dir: self.after(0, self.on_website_translation_done, count, out_dir),
            on_error=lambda err: self.after(0, self.on_website_translation_error, err),
        )
        worker.start()

    def _update_website_tr_status(self, msg):
        self.website_tr_status_var.set(msg)

    def _update_website_tr_progress(self, pct):
        self.website_tr_progress_var.set(pct)
        self.website_tr_pct_label.configure(text=f"{pct}%")

    def on_website_translation_done(self, count, out_dir):
        self.btn_website_translate_en.set_enabled(True)
        self.btn_website_translate_es.set_enabled(True)
        self.website_tr_status_var.set(f"¡Listo! Se tradujeron {count} página(s) en: {out_dir}")
        messagebox.showinfo(APP_TITLE, f"Se tradujeron {count} página(s).\n\nGuardadas en:\n{out_dir}")

    def on_website_translation_error(self, error_message):
        self.btn_website_translate_en.set_enabled(True)
        self.btn_website_translate_es.set_enabled(True)
        self.website_tr_status_var.set("Ocurrió un error durante la traducción.")
        messagebox.showerror(APP_TITLE, f"Error al traducir el sitio:\n\n{error_message[:500]}")

    # ---------------------------------------------------------------
    # Tema claro/oscuro
    # ---------------------------------------------------------------
    def toggle_theme(self):
        self.theme_name = "dark" if self.theme_name == "light" else "light"
        self.colors = DARK if self.theme_name == "dark" else LIGHT
        self._apply_theme()

    def _apply_theme(self):
        c = self.colors
        self.configure(bg=c["bg"])

        for f in self._bg_frames:
            f.configure(bg=c["bg"])
        for f in self._panel_frames:
            f.configure(bg=c["panel"])
        for f in self._border_frames:
            f.configure(bg=c["border"])
        for cv in self._canvases_bg:
            cv.configure(bg=c["bg"])
            cv.itemconfig(1, fill=c["accent"])

        self.title_label.configure(bg=c["bg"], fg=c["text"])
        self.transcript_title.configure(bg=c["panel"], fg=c["text"])
        self.pct_label.configure(bg=c["panel"], fg=c["accent"])
        self.time_label.configure(bg=c["panel"], fg=c["text_muted"])
        self.hint_label.configure(bg=c["panel"], fg=c["text_muted"])
        self.save_hint.configure(bg=c["panel"], fg=c["text_muted"])
        self.config_title.configure(bg=c["panel"], fg=c["text"])
        self.status_label.configure(bg=c["panel"], fg=c["text_muted"])
        self.log_title.configure(bg=c["panel"], fg=c["text"])

        for lbl in self._field_labels:
            lbl.configure(bg=c["panel"], fg=c["text"])

        for tw in self._surface_widgets:
            tw.configure(bg=c["surface"], fg=c["text"], insertbackground=c["text"])
            tw.tag_configure("timestamp", foreground=c["accent"])

        for tw in self._log_widgets:
            tw.configure(bg=c["log_bg"], fg=c["text_muted"])

        self.style.configure("Coral.Horizontal.TProgressbar", troughcolor=c["track"],
                              background=c["accent"], bordercolor=c["track"],
                              lightcolor=c["accent"], darkcolor=c["accent"])

        self.style.configure("App.TNotebook", background=c["bg"], borderwidth=0)
        self.style.configure("App.TNotebook.Tab", background=c["bg"], foreground=c["text_muted"],
                              padding=(14, 8), font=(FONT_FAMILY, 10, "bold"), borderwidth=0)
        self.style.map(
            "App.TNotebook.Tab",
            background=[("selected", c["panel"])],
            foreground=[("selected", c["accent"])],
        )

        self.style.configure("TEntry", fieldbackground=c["surface"], foreground=c["text"],
                              bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"])
        self.style.configure("TCombobox", fieldbackground=c["surface"], foreground=c["text"])
        self.style.map("TCombobox", fieldbackground=[("readonly", c["surface"])])
        self.style.configure("TCheckbutton", background=c["panel"], foreground=c["text"])
        self.style.map("TCheckbutton", background=[("active", c["panel"])])

        for btn in self._round_buttons_accent:
            btn.colors = c
            btn.set_panel_bg(c["panel"])
        for btn in self._round_buttons_plain:
            btn.colors = c
            btn.set_panel_bg(c["bg"])
        for btn in self._round_buttons_plain_panel:
            btn.colors = c
            btn.set_panel_bg(c["panel"])

        self.theme_btn.text = "☀️ Modo claro" if self.theme_name == "dark" else "🌙 Modo oscuro"
        self.theme_btn._draw(self.theme_btn._fill_color())

    # ---------------------------------------------------------------
    # Acciones
    # ---------------------------------------------------------------
    def _bind_text_edit_shortcuts(self, widget):
        """Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z para deshacer/rehacer en un widget Text."""
        widget.bind("<Control-z>", lambda e: (widget.event_generate("<<Undo>>"), "break"))
        widget.bind("<Control-y>", lambda e: (widget.event_generate("<<Redo>>"), "break"))
        widget.bind("<Control-Shift-Z>", lambda e: (widget.event_generate("<<Redo>>"), "break"))

    def _add_text_context_menu(self, widget):
        """Menú de clic derecho con deshacer/rehacer/cortar/copiar/pegar para un Text."""
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Deshacer", command=lambda: widget.event_generate("<<Undo>>"))
        menu.add_command(label="Rehacer", command=lambda: widget.event_generate("<<Redo>>"))
        menu.add_separator()
        menu.add_command(label="Cortar", command=lambda: widget.event_generate("<<Cut>>"))
        menu.add_command(label="Copiar", command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label="Pegar", command=lambda: widget.event_generate("<<Paste>>"))

        def show_menu(event):
            widget.focus_set()
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        widget.bind("<Button-3>", show_menu)
        return menu

    def _add_entry_context_menu(self, widget):
        """Menú de clic derecho con cortar/copiar/pegar para un Entry (rutas/carpetas)."""
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Cortar", command=lambda: widget.event_generate("<<Cut>>"))
        menu.add_command(label="Copiar", command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label="Pegar", command=lambda: widget.event_generate("<<Paste>>"))

        def show_menu(event):
            widget.focus_set()
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        widget.bind("<Button-3>", show_menu)
        return menu

    def _refresh_track_indicator(self):
        if len(self.audio_tracks_info) > 1:
            selected = self.selected_audio_track
            if selected is None:
                self.track_info_label.configure(text="Varias pistas de audio detectadas — elige una.")
            else:
                track = next(
                    (t for t in self.audio_tracks_info if t["track_index"] == selected), None
                )
                lang = (track or {}).get("language") or "desconocido"
                self.track_info_label.configure(text=f"Usando pista #{selected} (idioma: {lang})")
            self.btn_change_track.pack(side="left", padx=(8, 0))
        else:
            self.track_info_label.configure(text="")
            self.btn_change_track.pack_forget()

    def pick_file(self):
        path = filedialog.askopenfilename(
            title="Selecciona un archivo de audio o video",
            filetypes=[
                ("Audio/Video", "*.mp3 *.wav *.m4a *.flac *.ogg *.mp4 *.mkv *.mov *.avi"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if path:
            self.filepath.set(path)
            self.selected_audio_track = None
            self.audio_tracks_info = probe_audio_tracks(path)
            self._refresh_track_indicator()
            if len(self.audio_tracks_info) > 1:
                self.open_track_selector()

    def open_track_selector(self):
        c = self.colors
        win = tk.Toplevel(self)
        win.title("Seleccionar pista de audio")
        win.configure(bg=c["bg"])
        win.geometry("480x360")
        win.minsize(420, 300)
        win.transient(self)
        win.grab_set()

        tk.Label(
            win, text="Este video tiene varias pistas de audio", bg=c["bg"], fg=c["text"],
            font=(FONT_FAMILY, 13, "bold")
        ).pack(anchor="w", padx=18, pady=(18, 4))
        tk.Label(
            win, text="Elige cuál quieres transcribir (evita mezclar idiomas):",
            bg=c["bg"], fg=c["text_muted"], font=(FONT_FAMILY, 9)
        ).pack(anchor="w", padx=18, pady=(0, 12))

        rows = tk.Frame(win, bg=c["bg"])
        rows.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        def choose(track_index):
            self.selected_audio_track = track_index
            self.set_status(f"Pista de audio seleccionada: #{track_index}")
            self._refresh_track_indicator()
            win.destroy()

        for track in self.audio_tracks_info:
            idx = track["track_index"]
            lang = track.get("language") or "desconocido"
            title = track.get("title") or ""
            label = f"Pista {idx} — idioma: {lang}"
            if title:
                label += f" ({title})"

            outer = tk.Frame(rows, bg=c["border"])
            outer.pack(fill="x", pady=5)
            row = tk.Frame(outer, bg=c["panel"])
            row.pack(fill="both", expand=True, padx=1, pady=1)

            tk.Label(row, text=label, bg=c["panel"], fg=c["text"],
                     font=(FONT_FAMILY, 10)).pack(side="left", padx=12, pady=10)
            btn = RoundedButton(row, "Usar esta", lambda i=idx: choose(i),
                                 c, c["panel"], width=100, height=30)
            btn.pack(side="right", padx=12, pady=10)

    def pick_out_dir(self):
        path = filedialog.askdirectory(title="Selecciona carpeta de salida")
        if path:
            self.out_dir.set(path)

    def log(self, message: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def set_status(self, message: str):
        self.status_var.set(message)
        self.log(message)

    def append_segment(self, text, start, end):
        self.transcript_text.insert("end", f"[{start} → {end}] ", "timestamp")
        self.transcript_text.insert("end", text + "\n\n")
        self.transcript_text.see("end")

    def update_progress(self, pct):
        self.progress_pct.set(pct)
        self.pct_label.configure(text=f"{pct}%")

    def update_time_info(self, elapsed_str, eta_str):
        self.time_label.configure(text=f"Transcurrido: {elapsed_str}  ·  Restante: ~{eta_str}")

    def start_transcription(self):
        filepath = self.filepath.get().strip()
        if not filepath or not os.path.isfile(filepath):
            messagebox.showerror(APP_TITLE, "Selecciona un archivo válido primero.")
            return

        self.btn_run.set_enabled(False)
        self.btn_translate_en.set_enabled(False)
        self.btn_translate_es.set_enabled(False)
        self.transcript_text.delete("1.0", "end")
        self.update_progress(0)
        self.time_label.configure(text="")
        self.set_status("Iniciando...")
        self._last_paths = None
        self._translation_cache = None

        language = LANGUAGE_MAP.get(self.language_label.get())

        worker = TranscriberWorker(
            filepath=filepath,
            model_size=self.model_size.get(),
            language=language,
            use_gpu=self.use_gpu.get(),
            out_dir=self.out_dir.get(),
            on_status=lambda msg: self.after(0, self.set_status, msg),
            on_segment=lambda text, start, end: self.after(0, self.append_segment, text, start, end),
            on_progress_pct=lambda pct: self.after(0, self.update_progress, pct),
            on_done=lambda txt, srt, md, plain, cache: self.after(0, self.on_done, txt, srt, md, plain, cache),
            on_error=lambda err: self.after(0, self.on_error, err),
            on_clear=lambda: self.after(0, lambda: self.transcript_text.delete("1.0", "end")),
            on_time_update=lambda elapsed, eta: self.after(0, self.update_time_info, elapsed, eta),
            audio_track=self.selected_audio_track,
        )
        worker.start()

    def on_done(self, txt_path, srt_path, md_path, plain_txt_path, cache):
        self.btn_run.set_enabled(True)
        self.set_status("¡Transcripción completada!")
        self._last_paths = (txt_path, srt_path, md_path, plain_txt_path)
        self._translation_cache = cache  # audio_array, sr, device, model_size, base_name, segments
        self.log(f"TXT (con tiempos): {txt_path}")
        self.log(f"SRT: {srt_path}")
        self.log(f"MD:  {md_path}")
        self.log(f"TXT (texto plano): {plain_txt_path}")
        self.btn_translate_en.set_enabled(True)
        self.btn_translate_es.set_enabled(True)
        messagebox.showinfo(APP_TITLE, f"Transcripción completada.\n\nArchivos guardados en:\n{Path(txt_path).parent}")

    def on_error(self, error_message):
        self.btn_run.set_enabled(True)
        self.set_status("Ocurrió un error.")
        self.log(error_message)
        messagebox.showerror(APP_TITLE, f"Error durante la transcripción:\n\n{error_message[:500]}")

    # ---------------------------------------------------------------
    # Traducción (paso aparte, usa lo ya transcrito)
    # ---------------------------------------------------------------
    def _parse_current_transcript_segments(self):
        """Convierte el contenido actual del panel (líneas '[HH:MM:SS --> HH:MM:SS] texto')
        en una lista de dicts {start, end, text}, para poder traducirlo."""
        import re

        def hms_to_seconds(ts):
            parts = [float(p) for p in ts.split(":")]
            while len(parts) < 3:
                parts.insert(0, 0.0)
            h, m, s = parts
            return h * 3600 + m * 60 + s

        pattern = re.compile(r"^\[(.+?) --> (.+?)\] (.*)$")
        content = self.transcript_text.get("1.0", "end")
        segments = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if match:
                start_s, end_s, text = match.group(1), match.group(2), match.group(3)
                segments.append({
                    "start": hms_to_seconds(start_s),
                    "end": hms_to_seconds(end_s),
                    "text": text.strip(),
                })
        return segments

    def start_translate_to_english(self):
        if not self._translation_cache:
            messagebox.showwarning(APP_TITLE, "Primero transcribe un archivo.")
            return
        self.btn_translate_en.set_enabled(False)
        self.btn_translate_es.set_enabled(False)
        self.btn_run.set_enabled(False)
        self.update_progress(0)
        self.set_status("Iniciando traducción al inglés...")

        c = self._translation_cache
        worker = TranslateToEnglishWorker(
            audio_array=c["audio_array"], sr=c["sr"], device=c["device"],
            model_size=c["model_size"], out_dir=self.out_dir.get(), base_name=c["base_name"],
            on_status=lambda msg: self.after(0, self.set_status, msg),
            on_segment=lambda text, start, end: self.after(0, self.append_segment, text, start, end),
            on_progress_pct=lambda pct: self.after(0, self.update_progress, pct),
            on_done=lambda txt, srt, plain: self.after(0, self.on_translation_done, "inglés", txt, srt, plain),
            on_error=lambda err: self.after(0, self.on_translation_error, err),
            on_clear=lambda: self.after(0, lambda: self.transcript_text.delete("1.0", "end")),
        )
        worker.start()

    def start_translate_to_spanish(self):
        segments = self._parse_current_transcript_segments()
        if not segments:
            messagebox.showwarning(
                APP_TITLE, "No hay texto en el panel para traducir todavía."
            )
            return
        base_name = (self._translation_cache or {}).get("base_name")
        if not base_name and self._last_paths:
            base_name = Path(self._last_paths[0]).stem.replace("_transcripcion", "")
        base_name = base_name or "transcripcion"

        self.btn_translate_en.set_enabled(False)
        self.btn_translate_es.set_enabled(False)
        self.btn_run.set_enabled(False)
        self.update_progress(0)
        self.set_status("Iniciando traducción al español...")

        worker = TranslateToSpanishWorker(
            segments=segments, out_dir=self.out_dir.get(), base_name=base_name,
            on_status=lambda msg: self.after(0, self.set_status, msg),
            on_segment=lambda text, start, end: self.after(0, self.append_segment, text, start, end),
            on_progress_pct=lambda pct: self.after(0, self.update_progress, pct),
            on_done=lambda txt, srt, plain: self.after(0, self.on_translation_done, "español", txt, srt, plain),
            on_error=lambda err: self.after(0, self.on_translation_error, err),
            on_clear=lambda: self.after(0, lambda: self.transcript_text.delete("1.0", "end")),
        )
        worker.start()

    def on_translation_done(self, lang_label, txt_path, srt_path, plain_path):
        self.btn_run.set_enabled(True)
        self.btn_translate_en.set_enabled(True)
        self.btn_translate_es.set_enabled(True)
        self.set_status(f"¡Traducción al {lang_label} completada!")
        self.log(f"TXT ({lang_label}): {txt_path}")
        self.log(f"SRT ({lang_label}): {srt_path}")
        self.log(f"TXT plano ({lang_label}): {plain_path}")
        messagebox.showinfo(
            APP_TITLE,
            f"Traducción al {lang_label} completada.\n\nArchivos guardados en:\n{Path(txt_path).parent}",
        )

    def on_translation_error(self, error_message):
        self.btn_run.set_enabled(True)
        self.btn_translate_en.set_enabled(True)
        self.btn_translate_es.set_enabled(True)
        self.set_status("Ocurrió un error durante la traducción.")
        self.log(error_message)
        messagebox.showerror(APP_TITLE, f"Error durante la traducción:\n\n{error_message[:500]}")

    def save_edited_transcript(self):
        if not self._last_paths:
            messagebox.showwarning(APP_TITLE, "Todavía no hay una transcripción generada para guardar.")
            return
        txt_path, srt_path, md_path, plain_txt_path = self._last_paths
        content = self.transcript_text.get("1.0", "end").strip()

        Path(txt_path).write_text(content, encoding="utf-8")

        plain = "\n".join(line.split("] ", 1)[-1] for line in content.splitlines() if line.strip())
        md_text = Path(md_path).read_text(encoding="utf-8") if Path(md_path).exists() else ""
        header = md_text.split("## Texto completo")[0] if "## Texto completo" in md_text else ""
        Path(md_path).write_text(header + "## Texto completo\n\n" + plain + "\n", encoding="utf-8")
        Path(plain_txt_path).write_text(plain, encoding="utf-8")

        self.set_status("Cambios guardados en los archivos .txt, .md y texto plano")
        messagebox.showinfo(APP_TITLE, "Cambios guardados correctamente.")

    # ---------------------------------------------------------------
    # Gestor de modelos: descargar / eliminar modelos permanentemente
    # junto al .exe, sin tener que precargarlos durante la compilación.
    # ---------------------------------------------------------------
    def get_models_dir(self) -> Path:
        if getattr(sys, "frozen", False):
            base_dir = Path(sys.executable).parent
        else:
            base_dir = Path(__file__).resolve().parent
        d = base_dir / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def model_file_path(self, name: str) -> Path:
        import whisper
        url = whisper._MODELS[name]
        return self.get_models_dir() / os.path.basename(url)

    def is_model_downloaded(self, name: str) -> bool:
        try:
            return self.model_file_path(name).exists()
        except Exception:
            return False

    def open_model_manager(self):
        c = self.colors
        win = tk.Toplevel(self)
        win.title("Gestor de modelos")
        win.configure(bg=c["bg"])
        win.geometry("560x460")
        win.minsize(500, 380)
        win.transient(self)

        tk.Label(win, text="Modelos disponibles", bg=c["bg"], fg=c["text"],
                 font=(FONT_FAMILY, 13, "bold")).pack(anchor="w", padx=18, pady=(18, 4))
        tk.Label(win, text=f"Carpeta: {self.get_models_dir()}", bg=c["bg"], fg=c["text_muted"],
                 font=(FONT_FAMILY, 8)).pack(anchor="w", padx=18, pady=(0, 10))

        rows_container = tk.Frame(win, bg=c["bg"])
        rows_container.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        for name in MODEL_OPTIONS:
            self._build_model_row(rows_container, name)

    def _build_model_row(self, parent, name):
        c = self.colors
        outer = tk.Frame(parent, bg=c["border"])
        outer.pack(fill="x", pady=5)
        row = tk.Frame(outer, bg=c["panel"])
        row.pack(fill="both", expand=True, padx=1, pady=1)

        left = tk.Frame(row, bg=c["panel"])
        left.pack(side="left", fill="both", expand=True, padx=12, pady=10)

        name_lbl = tk.Label(left, text=f"{name}   ({MODEL_SIZE_LABEL.get(name, '')})",
                             bg=c["panel"], fg=c["text"], font=(FONT_FAMILY, 10, "bold"))
        name_lbl.pack(anchor="w")

        downloaded = self.is_model_downloaded(name)
        status_lbl = tk.Label(
            left, text="Descargado" if downloaded else "No descargado",
            bg=c["panel"], fg=(c["accent"] if downloaded else c["text_muted"]),
            font=(FONT_FAMILY, 9),
        )
        status_lbl.pack(anchor="w")

        right = tk.Frame(row, bg=c["panel"])
        right.pack(side="right", padx=12, pady=10)

        widgets = {"status_lbl": status_lbl, "btn": None}

        def make_btn():
            is_dl = self.is_model_downloaded(name)
            label = "Eliminar" if is_dl else "Descargar"
            action = (lambda: self.delete_model(name, widgets)) if is_dl else \
                     (lambda: self.download_model(name, widgets))
            btn = RoundedButton(right, label, action, c, c["panel"], width=110, height=30,
                                 use_accent=not is_dl)
            btn.pack()
            widgets["btn"] = btn

        make_btn()
        widgets["make_btn"] = make_btn
        widgets["right"] = right

    def download_model(self, name, widgets):
        widgets["btn"].destroy()
        pct_lbl = tk.Label(widgets["right"], text="0%", bg=self.colors["panel"],
                            fg=self.colors["accent"], font=(FONT_FAMILY, 9, "bold"))
        pct_lbl.pack()
        widgets["status_lbl"].configure(text="Descargando...", fg=self.colors["accent"])

        def on_progress(pct):
            self.after(0, lambda: pct_lbl.configure(text=f"{pct}%"))

        def on_done():
            def _finish():
                pct_lbl.destroy()
                widgets["status_lbl"].configure(text="Descargado", fg=self.colors["accent"])
                widgets["make_btn"]()
            self.after(0, _finish)

        def on_error(err):
            def _fail():
                pct_lbl.destroy()
                widgets["status_lbl"].configure(text="Error al descargar", fg=self.colors["text_muted"])
                widgets["make_btn"]()
                messagebox.showerror(APP_TITLE, f"No se pudo descargar el modelo:\n\n{err[:400]}")
            self.after(0, _fail)

        def _run():
            try:
                import whisper
                _install_tqdm_progress_hook()
                _set_progress_callback(on_progress)
                whisper._download(whisper._MODELS[name], str(self.get_models_dir()), False)
                on_done()
            except Exception as e:
                # Si algo falló a mitad de la descarga, el archivo puede haber
                # quedado a medias/corrupto. Lo borramos para evitar errores
                # confusos más adelante al intentar usarlo.
                try:
                    partial = self.model_file_path(name)
                    if partial.exists():
                        partial.unlink()
                except Exception:
                    pass
                on_error(str(e))

        threading.Thread(target=_run, daemon=True).start()

    def delete_model(self, name, widgets):
        if not messagebox.askyesno(APP_TITLE, f"¿Eliminar el modelo '{name}' descargado?"):
            return
        try:
            path = self.model_file_path(name)
            if path.exists():
                path.unlink()
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"No se pudo eliminar el modelo:\n\n{e}")
            return
        widgets["status_lbl"].configure(text="No descargado", fg=self.colors["text_muted"])
        widgets["make_btn"]()


if __name__ == "__main__":
    app = App()
    app.mainloop()
