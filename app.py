"""
Transcriptor de Audio Local - GUI
Usa faster-whisper para transcribir audio/video localmente, con GPU (CUDA) o CPU.
"""

import os
import sys
import threading
import traceback
from datetime import timedelta
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "Transcriptor Local (Whisper)"

# --------------------------------------------------------------------------
# Utilidades de formato
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Lógica de transcripción (corre en un hilo aparte para no congelar la GUI)
# --------------------------------------------------------------------------

class TranscriberWorker(threading.Thread):
    def __init__(self, filepath, model_size, language, use_gpu, out_dir,
                 on_progress, on_done, on_error):
        super().__init__(daemon=True)
        self.filepath = filepath
        self.model_size = model_size
        self.language = language
        self.use_gpu = use_gpu
        self.out_dir = out_dir
        self.on_progress = on_progress
        self.on_done = on_done
        self.on_error = on_error

    def run(self):
        try:
            self.on_progress("Cargando modelo (puede tardar la primera vez)...")
            from faster_whisper import WhisperModel

            device = "cpu"
            compute_type = "int8"

            if self.use_gpu:
                try:
                    import ctranslate2
                    supported = ctranslate2.get_supported_compute_types("cuda")
                    if supported:
                        device = "cuda"
                        compute_type = "float16" if "float16" in supported else supported[0]
                except Exception:
                    device = "cpu"
                    compute_type = "int8"

            if device == "cpu" and self.use_gpu:
                self.on_progress("No se detectó GPU compatible. Usando CPU...")

            model = WhisperModel(self.model_size, device=device, compute_type=compute_type)

            self.on_progress(f"Transcribiendo en {device.upper()}... esto puede tardar unos minutos.")

            segments, info = model.transcribe(
                self.filepath,
                language=self.language,
                vad_filter=True,
            )

            lines_txt = []
            lines_srt = []
            plain_text = []

            for i, seg in enumerate(segments, start=1):
                start = format_timestamp_txt(seg.start)
                end = format_timestamp_txt(seg.end)
                lines_txt.append(f"[{start} --> {end}] {seg.text.strip()}")
                plain_text.append(seg.text.strip())

                srt_start = format_timestamp_srt(seg.start)
                srt_end = format_timestamp_srt(seg.end)
                lines_srt.append(f"{i}\n{srt_start} --> {srt_end}\n{seg.text.strip()}\n")

                self.on_progress(f"Procesado segmento {i} ({start})...")

            base_name = Path(self.filepath).stem
            out_dir = Path(self.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            txt_path = out_dir / f"{base_name}_transcripcion.txt"
            srt_path = out_dir / f"{base_name}_transcripcion.srt"
            md_path = out_dir / f"{base_name}_transcripcion.md"

            txt_path.write_text("\n".join(lines_txt), encoding="utf-8")
            srt_path.write_text("\n".join(lines_srt), encoding="utf-8")

            detected_lang = getattr(info, "language", self.language or "desconocido")
            md_content = (
                f"# Transcripción: {base_name}\n\n"
                f"- Idioma detectado/usado: {detected_lang}\n"
                f"- Modelo: {self.model_size}\n\n"
                f"## Texto completo\n\n"
                f"{' '.join(plain_text)}\n"
            )
            md_path.write_text(md_content, encoding="utf-8")

            self.on_done(str(txt_path), str(srt_path), str(md_path))

        except Exception as e:
            self.on_error(f"{e}\n\n{traceback.format_exc()}")


# --------------------------------------------------------------------------
# Interfaz gráfica
# --------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("640x480")
        self.resizable(False, False)

        self.filepath = tk.StringVar()
        self.out_dir = tk.StringVar(value=str(Path.home() / "Transcripciones"))
        self.model_size = tk.StringVar(value="medium")
        self.language_label = tk.StringVar(value="Detectar automáticamente")
        self.use_gpu = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 12, "pady": 8}

        # Archivo de entrada
        frame_file = ttk.LabelFrame(self, text="Archivo de audio/video")
        frame_file.pack(fill="x", **pad)

        entry = ttk.Entry(frame_file, textvariable=self.filepath, width=60)
        entry.pack(side="left", padx=8, pady=8, fill="x", expand=True)

        ttk.Button(frame_file, text="Examinar...", command=self.pick_file).pack(
            side="left", padx=8, pady=8
        )

        # Carpeta de salida
        frame_out = ttk.LabelFrame(self, text="Carpeta de salida")
        frame_out.pack(fill="x", **pad)

        ttk.Entry(frame_out, textvariable=self.out_dir, width=60).pack(
            side="left", padx=8, pady=8, fill="x", expand=True
        )
        ttk.Button(frame_out, text="Elegir...", command=self.pick_out_dir).pack(
            side="left", padx=8, pady=8
        )

        # Opciones
        frame_opts = ttk.LabelFrame(self, text="Opciones")
        frame_opts.pack(fill="x", **pad)

        ttk.Label(frame_opts, text="Modelo:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        model_combo = ttk.Combobox(
            frame_opts, textvariable=self.model_size, values=MODEL_OPTIONS,
            state="readonly", width=15
        )
        model_combo.grid(row=0, column=1, sticky="w", padx=8, pady=6)

        ttk.Label(frame_opts, text="Idioma:").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        lang_combo = ttk.Combobox(
            frame_opts, textvariable=self.language_label,
            values=list(LANGUAGE_MAP.keys()), state="readonly", width=22
        )
        lang_combo.grid(row=0, column=3, sticky="w", padx=8, pady=6)

        gpu_check = ttk.Checkbutton(
            frame_opts, text="Usar GPU (NVIDIA/CUDA) si está disponible",
            variable=self.use_gpu
        )
        gpu_check.grid(row=1, column=0, columnspan=4, sticky="w", padx=8, pady=6)

        # Botón transcribir
        self.btn_run = ttk.Button(self, text="Transcribir", command=self.start_transcription)
        self.btn_run.pack(pady=10)

        # Barra de progreso + estado
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill="x", padx=12, pady=4)

        self.status_var = tk.StringVar(value="Listo.")
        ttk.Label(self, textvariable=self.status_var, wraplength=600, justify="left").pack(
            padx=12, pady=4, anchor="w"
        )

        # Log
        frame_log = ttk.LabelFrame(self, text="Registro")
        frame_log.pack(fill="both", expand=True, padx=12, pady=8)

        self.log_text = tk.Text(frame_log, height=8, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)

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

    def start_transcription(self):
        filepath = self.filepath.get().strip()
        if not filepath or not os.path.isfile(filepath):
            messagebox.showerror(APP_TITLE, "Selecciona un archivo válido primero.")
            return

        self.btn_run.configure(state="disabled")
        self.progress.start(10)
        self.set_status("Iniciando...")

        language = LANGUAGE_MAP.get(self.language_label.get())

        worker = TranscriberWorker(
            filepath=filepath,
            model_size=self.model_size.get(),
            language=language,
            use_gpu=self.use_gpu.get(),
            out_dir=self.out_dir.get(),
            on_progress=lambda msg: self.after(0, self.set_status, msg),
            on_done=lambda txt, srt, md: self.after(0, self.on_done, txt, srt, md),
            on_error=lambda err: self.after(0, self.on_error, err),
        )
        worker.start()

    def on_done(self, txt_path, srt_path, md_path):
        self.progress.stop()
        self.btn_run.configure(state="normal")
        self.set_status("¡Transcripción completada!")
        self.log(f"TXT: {txt_path}")
        self.log(f"SRT: {srt_path}")
        self.log(f"MD:  {md_path}")
        messagebox.showinfo(
            APP_TITLE,
            f"Transcripción completada.\n\nArchivos guardados en:\n{Path(txt_path).parent}",
        )

    def on_error(self, error_message):
        self.progress.stop()
        self.btn_run.configure(state="normal")
        self.set_status("Ocurrió un error.")
        self.log(error_message)
        messagebox.showerror(APP_TITLE, f"Error durante la transcripción:\n\n{error_message[:500]}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
