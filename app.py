"""
Transcriptor de Audio Local - GUI estilo Claude (con modo claro/oscuro)
Usa Whisper (OpenAI) + PyTorch para transcribir localmente, con GPU (CUDA) o CPU.
PyTorch trae su propio runtime CUDA embebido, no requiere CUDA Toolkit/cuDNN
instalados por separado en el sistema.
"""

import os
import sys
import threading
import traceback
from datetime import timedelta
from pathlib import Path

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
            self.on_status("Cargando motor de transcripción (Whisper + PyTorch)...")
            import whisper
            import torch
            import whisper.transcribe as whisper_transcribe_module

            device = "cpu"
            if self.use_gpu:
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    self.on_status("No se detectó GPU compatible. Usando CPU...")

            if getattr(sys, "frozen", False):
                base_dir = Path(sys.executable).parent
            else:
                base_dir = Path(__file__).resolve().parent
            models_dir = str(base_dir / "models")
            os.makedirs(models_dir, exist_ok=True)

            outer_self = self

            class _ProgressBridge:
                """Reemplaza la barra tqdm interna de whisper para reportar % real."""
                def __init__(self, total=0, unit=None, disable=False, **kwargs):
                    self.total = total or 1
                    self.n = 0

                def update(self, n):
                    self.n += n
                    pct = min(100, int(self.n / self.total * 100))
                    outer_self.on_progress_pct(pct)

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def close(self):
                    pass

            def _patch_progress(module, bridge_cls):
                """Reemplaza la barra tqdm que usa `module` internamente, sin importar
                si esa lib la importó como `import tqdm` o `from tqdm import tqdm`."""
                current = getattr(module, "tqdm", None)
                if current is not None and hasattr(current, "tqdm"):
                    # Estilo "import tqdm" -> module.tqdm es el submódulo, .tqdm es la clase
                    current.tqdm = bridge_cls
                else:
                    # Estilo "from tqdm import tqdm" -> module.tqdm ya es la clase/función
                    module.tqdm = bridge_cls

            _patch_progress(whisper_transcribe_module, _ProgressBridge)

            def run_pass(dev):
                self.on_status(f"Cargando modelo '{self.model_size}' en {dev.upper()}...")
                m = whisper.load_model(self.model_size, device=dev, download_root=models_dir)
                self.on_status(f"Transcribiendo en {dev.upper()}... esto puede tardar unos minutos.")
                result = m.transcribe(
                    self.filepath,
                    language=self.language,
                    fp16=(dev == "cuda"),
                    verbose=False,
                )
                segs = result.get("segments", [])
                out_txt, out_srt, out_plain = [], [], []
                for i, seg in enumerate(segs, start=1):
                    start = format_timestamp_txt(seg["start"])
                    end = format_timestamp_txt(seg["end"])
                    text = seg["text"].strip()
                    out_txt.append(f"[{start} --> {end}] {text}")
                    out_plain.append(text)

                    srt_start = format_timestamp_srt(seg["start"])
                    srt_end = format_timestamp_srt(seg["end"])
                    out_srt.append(f"{i}\n{srt_start} --> {srt_end}\n{text}\n")

                    self.on_segment(text, start, end)
                return result.get("language", self.language), out_txt, out_srt, out_plain

            try:
                detected_lang, lines_txt, lines_srt, plain_text = run_pass(device)
            except Exception as gpu_err:
                error_text = str(gpu_err).lower()
                gpu_related = any(
                    kw in error_text for kw in ("cuda", "dll", "library", "driver", "gpu")
                )
                if device == "cuda" and gpu_related:
                    self.on_status(
                        "La GPU falló al procesar (revisa que el driver esté al día). "
                        "Reintentando automáticamente en CPU..."
                    )
                    device = "cpu"
                    detected_lang, lines_txt, lines_srt, plain_text = run_pass(device)
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

            md_content = (
                f"# Transcripción: {base_name}\n\n"
                f"- Idioma detectado/usado: {detected_lang or 'desconocido'}\n"
                f"- Modelo: {self.model_size}\n\n"
                f"## Texto completo\n\n{' '.join(plain_text)}\n"
            )
            md_path.write_text(md_content, encoding="utf-8")

            self.on_progress_pct(100)
            self.on_done(str(txt_path), str(srt_path), str(md_path))
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
        self._round_rect(2, 2, self.width - 2, self.height - 2, 10, fill=color, outline="")
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
        self.title(APP_TITLE)
        self.geometry("1150x720")
        self.minsize(900, 560)

        self.theme_name = "light"
        self.colors = LIGHT

        self.filepath = tk.StringVar()
        self.out_dir = tk.StringVar(value=str(Path.home() / "Transcripciones"))
        self.model_size = tk.StringVar(value="large-v3")
        self.language_label = tk.StringVar(value="Detectar automáticamente")
        self.use_gpu = tk.BooleanVar(value=True)
        self.progress_pct = tk.IntVar(value=0)
        self._last_paths = None

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

        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        left_panel = self._build_transcript_panel(paned)
        right_panel = self._build_controls_panel(paned)
        paned.add(left_panel, weight=3)
        paned.add(right_panel, weight=2)

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
        )
        self.transcript_text.pack(fill="both", expand=True)
        scrollbar.config(command=self.transcript_text.yview)
        self.transcript_text.tag_configure("timestamp", foreground=self.colors["accent"],
                                            font=(FONT_FAMILY, 9, "bold"))
        self._surface_widgets.append(self.transcript_text)

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
        outer, card = self._card(parent)
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
        ttk.Entry(file_row, textvariable=self.filepath).pack(side="left", fill="x", expand=True, ipady=3)
        b1 = RoundedButton(file_row, "Examinar", self.pick_file, self.colors, self.colors["panel"],
                            width=100, height=30)
        b1.pack(side="left", padx=(8, 0))
        self._round_buttons_accent.append(b1)

        field_label("Carpeta de salida", card)
        out_row = self._panel_frame(card)
        out_row.pack(fill="x", padx=16)
        ttk.Entry(out_row, textvariable=self.out_dir).pack(side="left", fill="x", expand=True, ipady=3)
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

        return outer

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
        messagebox.showinfo(APP_TITLE, f"Transcripción completada.\n\nArchivos guardados en:\n{Path(txt_path).parent}")

    def on_error(self, error_message):
        self.btn_run.set_enabled(True)
        self.set_status("Ocurrió un error.")
        self.log(error_message)
        messagebox.showerror(APP_TITLE, f"Error durante la transcripción:\n\n{error_message[:500]}")

    def save_edited_transcript(self):
        if not self._last_paths:
            messagebox.showwarning(APP_TITLE, "Todavía no hay una transcripción generada para guardar.")
            return
        txt_path, srt_path, md_path = self._last_paths
        content = self.transcript_text.get("1.0", "end").strip()

        Path(txt_path).write_text(content, encoding="utf-8")

        plain = " ".join(line.split("] ", 1)[-1] for line in content.splitlines() if line.strip())
        md_text = Path(md_path).read_text(encoding="utf-8") if Path(md_path).exists() else ""
        header = md_text.split("## Texto completo")[0] if "## Texto completo" in md_text else ""
        Path(md_path).write_text(header + "## Texto completo\n\n" + plain + "\n", encoding="utf-8")

        self.set_status("Cambios guardados en los archivos .txt y .md")
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
                outer_self = self

                class _ProgressBridge:
                    def __init__(self, total=0, **kwargs):
                        self.total = total or 1
                        self.n = 0

                    def update(self, n):
                        self.n += n
                        pct = min(100, int(self.n / self.total * 100))
                        on_progress(pct)

                    def __enter__(self):
                        return self

                    def __exit__(self, *exc):
                        return False

                    def close(self):
                        pass

                whisper.tqdm = _ProgressBridge
                whisper._download(whisper._MODELS[name], str(self.get_models_dir()), False)
                on_done()
            except Exception as e:
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
