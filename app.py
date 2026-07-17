"""
Transcriptor de Audio Local - GUI estilo Claude
Usa faster-whisper para transcribir audio/video localmente, con GPU (CUDA) o CPU.
Panel dividido: transcripcion en vivo (editable) + controles.
"""

import os
import threading
import traceback
from datetime import timedelta
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "Transcriptor Local"

# --------------------------------------------------------------------------
# Paleta estilo Claude
# --------------------------------------------------------------------------
BG = "#FAF9F5"          # crema de fondo
PANEL = "#FFFFFF"       # panel blanco
BORDER = "#E5E4DF"      # bordes suaves
TEXT = "#3D3929"        # texto principal
TEXT_MUTED = "#87867F"  # texto secundario
ACCENT = "#D97757"      # coral / naranja Claude
ACCENT_HOVER = "#C4633F"
TRACK = "#EDEBE3"       # fondo de barra de progreso

FONT_FAMILY = "Segoe UI"

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
                 on_status, on_segment, on_progress_pct, on_done, on_error):
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

    def run(self):
        try:
            self.on_status("Cargando modelo (puede tardar la primera vez)...")
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
                self.on_status("No se detectó GPU compatible. Usando CPU...")

            def run_pass(dev, ctype):
                m = WhisperModel(self.model_size, device=dev, compute_type=ctype)
                self.on_status(f"Transcribiendo en {dev.upper()}...")
                segs, inf = m.transcribe(
                    self.filepath,
                    language=self.language,
                    vad_filter=True,
                )
                total_duration = getattr(inf, "duration", None) or 0
                out_txt, out_srt, out_plain = [], [], []
                for i, seg in enumerate(segs, start=1):
                    start = format_timestamp_txt(seg.start)
                    end = format_timestamp_txt(seg.end)
                    text = seg.text.strip()
                    out_txt.append(f"[{start} --> {end}] {text}")
                    out_plain.append(text)

                    srt_start = format_timestamp_srt(seg.start)
                    srt_end = format_timestamp_srt(seg.end)
                    out_srt.append(f"{i}\n{srt_start} --> {srt_end}\n{text}\n")

                    pct = 0
                    if total_duration > 0:
                        pct = min(100, round((seg.end / total_duration) * 100))
                    self.on_segment(text, start, end)
                    self.on_progress_pct(pct)
                return inf, out_txt, out_srt, out_plain

            try:
                info, lines_txt, lines_srt, plain_text = run_pass(device, compute_type)
            except Exception as gpu_err:
                error_text = str(gpu_err).lower()
                gpu_related = any(
                    kw in error_text
                    for kw in ("cublas", "cudnn", "cuda", "dll", "library")
                )
                if device == "cuda" and gpu_related:
                    self.on_status(
                        "La GPU falló al cargar librerías CUDA/cuDNN "
                        "(probablemente falta el CUDA Toolkit). "
                        "Reintentando automáticamente en CPU..."
                    )
                    device, compute_type = "cpu", "int8"
                    info, lines_txt, lines_srt, plain_text = run_pass(device, compute_type)
                else:
                    raise

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

            self.on_progress_pct(100)
            self.on_done(str(txt_path), str(srt_path), str(md_path))

        except Exception as e:
            self.on_error(f"{e}\n\n{traceback.format_exc()}")


# --------------------------------------------------------------------------
# Widgets auxiliares con estética "Claude"
# --------------------------------------------------------------------------

class RoundedButton(tk.Canvas):
    """Boton con esquinas redondeadas dibujado a mano (Tkinter no las trae nativas)."""

    def __init__(self, parent, text, command, bg=ACCENT, fg="white",
                 hover_bg=ACCENT_HOVER, width=150, height=36, **kwargs):
        super().__init__(parent, width=width, height=height, bg=PANEL,
                          highlightthickness=0, **kwargs)
        self.command = command
        self.bg_color = bg
        self.hover_color = hover_bg
        self.fg = fg
        self.width = width
        self.height = height
        self.text = text
        self._draw(bg)
        self.bind("<Button-1>", lambda e: self._on_click())
        self.bind("<Enter>", lambda e: self._draw(self.hover_color))
        self.bind("<Leave>", lambda e: self._draw(self.bg_color))

    def _round_rect(self, x1, y1, x2, y2, r, **kw):
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kw)

    def _draw(self, color):
        self.delete("all")
        self._round_rect(2, 2, self.width - 2, self.height - 2, 10, fill=color, outline="")
        self.create_text(self.width / 2, self.height / 2, text=self.text,
                          fill=self.fg, font=(FONT_FAMILY, 10, "bold"))

    def set_enabled(self, enabled: bool):
        self.command_enabled = enabled
        self._draw(self.bg_color if enabled else TEXT_MUTED)

    def _on_click(self):
        if getattr(self, "command_enabled", True) and self.command:
            self.command()


# --------------------------------------------------------------------------
# Interfaz gráfica principal
# --------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1150x720")
        self.minsize(900, 560)
        self.configure(bg=BG)

        self.filepath = tk.StringVar()
        self.out_dir = tk.StringVar(value=str(Path.home() / "Transcripciones"))
        self.model_size = tk.StringVar(value="medium")
        self.language_label = tk.StringVar(value="Detectar automáticamente")
        self.use_gpu = tk.BooleanVar(value=True)
        self.progress_pct = tk.IntVar(value=0)

        self._setup_style()
        self._build_ui()

    # ---------------------------------------------------------------
    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT, font=(FONT_FAMILY, 10))
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT, font=(FONT_FAMILY, 10))
        style.configure("Muted.TLabel", background=PANEL, foreground=TEXT_MUTED, font=(FONT_FAMILY, 9))
        style.configure("Header.TLabel", background=BG, foreground=TEXT,
                         font=(FONT_FAMILY, 15, "bold"))
        style.configure("SectionTitle.TLabel", background=PANEL, foreground=TEXT,
                         font=(FONT_FAMILY, 10, "bold"))
        style.configure("Pct.TLabel", background=PANEL, foreground=ACCENT,
                         font=(FONT_FAMILY, 12, "bold"))

        style.configure("TEntry", fieldbackground="white", foreground=TEXT,
                         bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                         padding=6)
        style.configure("TCombobox", fieldbackground="white", foreground=TEXT,
                         padding=6)
        style.map("TCombobox", fieldbackground=[("readonly", "white")])

        style.configure("TCheckbutton", background=PANEL, foreground=TEXT,
                         font=(FONT_FAMILY, 10))
        style.map("TCheckbutton", background=[("active", PANEL)])

        style.configure("Coral.Horizontal.TProgressbar", troughcolor=TRACK,
                         background=ACCENT, bordercolor=TRACK, lightcolor=ACCENT,
                         darkcolor=ACCENT, thickness=10)

    # ---------------------------------------------------------------
    def _build_ui(self):
        # ---- Encabezado ----
        header = ttk.Frame(self, style="TFrame")
        header.pack(fill="x", padx=20, pady=(16, 8))

        dot = tk.Canvas(header, width=14, height=14, bg=BG, highlightthickness=0)
        dot.create_oval(1, 1, 13, 13, fill=ACCENT, outline="")
        dot.pack(side="left", padx=(0, 8))

        ttk.Label(header, text=APP_TITLE, style="Header.TLabel").pack(side="left")

        # ---- Cuerpo: paneles divididos y ajustables ----
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        left_panel = self._build_transcript_panel(paned)
        right_panel = self._build_controls_panel(paned)

        paned.add(left_panel, weight=3)
        paned.add(right_panel, weight=2)

    # ---------------------------------------------------------------
    def _card(self, parent):
        """Contenedor tipo tarjeta blanca con borde suave."""
        outer = tk.Frame(parent, bg=BORDER)
        inner = tk.Frame(outer, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        return outer, inner

    # ---------------------------------------------------------------
    def _build_transcript_panel(self, parent):
        outer, card = self._card(parent)

        top = tk.Frame(card, bg=PANEL)
        top.pack(fill="x", padx=16, pady=(14, 6))

        ttk.Label(top, text="Transcripción en vivo", style="SectionTitle.TLabel").pack(side="left")
        self.pct_label = ttk.Label(top, text="0%", style="Pct.TLabel")
        self.pct_label.pack(side="right")

        self.progress = ttk.Progressbar(
            card, mode="determinate", maximum=100, variable=self.progress_pct,
            style="Coral.Horizontal.TProgressbar"
        )
        self.progress.pack(fill="x", padx=16, pady=(0, 10))

        ttk.Label(card, text="Puedes editar el texto libremente mientras se genera o al finalizar.",
                  style="Muted.TLabel").pack(anchor="w", padx=16, pady=(0, 6))

        text_frame = tk.Frame(card, bg=PANEL)
        text_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        self.transcript_text = tk.Text(
            text_frame, wrap="word", bg="white", fg=TEXT,
            font=(FONT_FAMILY, 11), relief="flat", padx=12, pady=12,
            insertbackground=TEXT, yscrollcommand=scrollbar.set,
        )
        self.transcript_text.pack(fill="both", expand=True)
        scrollbar.config(command=self.transcript_text.yview)
        self.transcript_text.tag_configure("timestamp", foreground=ACCENT,
                                            font=(FONT_FAMILY, 9, "bold"))

        # Botones bajo el texto
        btn_row = tk.Frame(card, bg=PANEL)
        btn_row.pack(fill="x", padx=16, pady=(0, 14))

        self.btn_save = RoundedButton(btn_row, "Guardar cambios", self.save_edited_transcript,
                                       bg=ACCENT, width=160, height=34)
        self.btn_save.pack(side="left")

        ttk.Label(btn_row, text="  Sobrescribe los archivos .txt y .md con tus ediciones",
                  style="Muted.TLabel").pack(side="left", padx=8)

        return outer

    # ---------------------------------------------------------------
    def _build_controls_panel(self, parent):
        outer, card = self._card(parent)
        pad = {"padx": 16, "pady": 6}

        ttk.Label(card, text="Configuración", style="SectionTitle.TLabel").pack(
            anchor="w", padx=16, pady=(14, 10)
        )

        # Archivo
        ttk.Label(card, text="Archivo de audio/video", style="Panel.TLabel").pack(anchor="w", **pad)
        file_row = tk.Frame(card, bg=PANEL)
        file_row.pack(fill="x", padx=16)
        ttk.Entry(file_row, textvariable=self.filepath).pack(side="left", fill="x", expand=True, ipady=3)
        RoundedButton(file_row, "Examinar", self.pick_file, width=100, height=30).pack(side="left", padx=(8, 0))

        # Carpeta de salida
        ttk.Label(card, text="Carpeta de salida", style="Panel.TLabel").pack(anchor="w", **pad)
        out_row = tk.Frame(card, bg=PANEL)
        out_row.pack(fill="x", padx=16)
        ttk.Entry(out_row, textvariable=self.out_dir).pack(side="left", fill="x", expand=True, ipady=3)
        RoundedButton(out_row, "Elegir", self.pick_out_dir, width=100, height=30).pack(side="left", padx=(8, 0))

        # Modelo
        ttk.Label(card, text="Modelo", style="Panel.TLabel").pack(anchor="w", **pad)
        ttk.Combobox(card, textvariable=self.model_size, values=MODEL_OPTIONS,
                     state="readonly").pack(fill="x", padx=16)

        # Idioma
        ttk.Label(card, text="Idioma", style="Panel.TLabel").pack(anchor="w", **pad)
        ttk.Combobox(card, textvariable=self.language_label,
                     values=list(LANGUAGE_MAP.keys()), state="readonly").pack(fill="x", padx=16)

        # GPU
        gpu_row = tk.Frame(card, bg=PANEL)
        gpu_row.pack(fill="x", padx=16, pady=(14, 4))
        ttk.Checkbutton(gpu_row, text="Usar GPU (NVIDIA/CUDA) si está disponible",
                         variable=self.use_gpu).pack(anchor="w")

        # Botón transcribir
        run_row = tk.Frame(card, bg=PANEL)
        run_row.pack(fill="x", padx=16, pady=(18, 10))
        self.btn_run = RoundedButton(run_row, "Transcribir", self.start_transcription,
                                      width=200, height=40)
        self.btn_run.pack(anchor="w")

        # Estado
        self.status_var = tk.StringVar(value="Listo.")
        ttk.Label(card, textvariable=self.status_var, style="Muted.TLabel",
                  wraplength=320, justify="left").pack(anchor="w", padx=16, pady=(4, 10))

        # Log
        ttk.Label(card, text="Registro", style="SectionTitle.TLabel").pack(anchor="w", padx=16, pady=(6, 4))
        log_frame = tk.Frame(card, bg=PANEL)
        log_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        log_scroll = ttk.Scrollbar(log_frame)
        log_scroll.pack(side="right", fill="y")
        self.log_text = tk.Text(log_frame, height=8, bg="#F5F4EF", fg=TEXT_MUTED,
                                 font=(FONT_FAMILY, 9), relief="flat", padx=8, pady=8,
                                 yscrollcommand=log_scroll.set, state="disabled")
        self.log_text.pack(fill="both", expand=True)
        log_scroll.config(command=self.log_text.yview)

        return outer

    # ---------------------------------------------------------------
    # Acciones
    # ---------------------------------------------------------------
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

    def append_segment(self, text, start, end):
        self.transcript_text.insert("end", f"[{start} → {end}] ", "timestamp")
        self.transcript_text.insert("end", text + "\n\n")
        self.transcript_text.see("end")

    def update_progress(self, pct):
        self.progress_pct.set(pct)
        self.pct_label.configure(text=f"{pct}%")

    def start_transcription(self):
        filepath = self.filepath.get().strip()
        if not filepath or not os.path.isfile(filepath):
            messagebox.showerror(APP_TITLE, "Selecciona un archivo válido primero.")
            return

        self.btn_run.set_enabled(False)
        self.transcript_text.delete("1.0", "end")
        self.update_progress(0)
        self.set_status("Iniciando...")
        self._last_paths = None

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
            on_done=lambda txt, srt, md: self.after(0, self.on_done, txt, srt, md),
            on_error=lambda err: self.after(0, self.on_error, err),
        )
        worker.start()

    def on_done(self, txt_path, srt_path, md_path):
        self.btn_run.set_enabled(True)
        self.set_status("¡Transcripción completada!")
        self._last_paths = (txt_path, srt_path, md_path)
        self.log(f"TXT: {txt_path}")
        self.log(f"SRT: {srt_path}")
        self.log(f"MD:  {md_path}")
        messagebox.showinfo(
            APP_TITLE,
            f"Transcripción completada.\n\nArchivos guardados en:\n{Path(txt_path).parent}",
        )

    def on_error(self, error_message):
        self.btn_run.set_enabled(True)
        self.set_status("Ocurrió un error.")
        self.log(error_message)
        messagebox.showerror(APP_TITLE, f"Error durante la transcripción:\n\n{error_message[:500]}")

    def save_edited_transcript(self):
        if not getattr(self, "_last_paths", None):
            messagebox.showwarning(APP_TITLE, "Todavía no hay una transcripción generada para guardar.")
            return
        txt_path, srt_path, md_path = self._last_paths
        content = self.transcript_text.get("1.0", "end").strip()

        Path(txt_path).write_text(content, encoding="utf-8")

        plain = " ".join(
            line.split("] ", 1)[-1] for line in content.splitlines() if line.strip()
        )
        md_text = Path(md_path).read_text(encoding="utf-8") if Path(md_path).exists() else ""
        header = md_text.split("## Texto completo")[0] if "## Texto completo" in md_text else ""
        Path(md_path).write_text(header + "## Texto completo\n\n" + plain + "\n", encoding="utf-8")

        self.set_status("Cambios guardados en los archivos .txt y .md")
        messagebox.showinfo(APP_TITLE, "Cambios guardados correctamente.")


if __name__ == "__main__":
    app = App()
    app.mainloop()
