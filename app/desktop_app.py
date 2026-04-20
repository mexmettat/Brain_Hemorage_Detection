import os
import sys
import json
import threading
import time
from pathlib import Path
from typing import Optional

# ── Proje kökünü ekle ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# ── GUI kütüphaneleri ──────────────────────────────────────
try:
    import customtkinter as ctk
except ImportError:
    print("ERROR: customtkinter not found. To install: pip install customtkinter")
    sys.exit(1)

import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from PIL import Image, ImageTk, ImageOps
    PIL_OK = True
except ImportError:
    PIL_OK = False

# ── Renk Paleti (TAMAMEN ORİJİNAL TASARIM) ──────────────────
COLORS = {
    "bg":          "#0d1117",
    "bg_panel":    "#161b22",
    "bg_card":     "#1c2333",
    "bg_input":    "#21262d",
    "border":      "#30363d",
    "accent_red":  "#ef4444",
    "accent_red2": "#dc2626",
    "accent_green":"#22c55e",
    "accent_blue": "#3b82f6",
    "text_primary":"#f0f6fc",
    "text_sec":    "#8b949e",
    "text_muted":  "#484f58",
    "hover":       "#2d333b",
    "normal_bar":  "#22c55e",
    "hemo_bar":    "#ef4444",
    "warn_bar":    "#fbbf24", # Amber/Yellow for disagreement
    "bg_normal":   "#0f2d1a",
    "bg_hemo":     "#3d1515",
    "bg_warn":     "#3d2d0d",
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ══════════════════════════════════════════════════════════
# Yardımcı: Görüntüyü CTkImage olarak yükle
# ══════════════════════════════════════════════════════════
def load_ctk_image(path: str, size: tuple) -> Optional[ctk.CTkImage]:
    if not PIL_OK: return None
    try:
        img = Image.open(path).convert("RGB")
        img = ImageOps.fit(img, size, Image.LANCZOS)
        return ctk.CTkImage(light_image=img, dark_image=img, size=size)
    except Exception: return None

# ══════════════════════════════════════════════════════════
# Model Yöneticisi
# ══════════════════════════════════════════════════════════
class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self._initialized = True
        self.models = {}
        self.device = None
        self.load_error = None

    def load(self, callback=None):
        def _load():
            try:
                import torch
                from models import get_pretrained_model, CustomCNN
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # ConvNeXt Load
                path_cv = ROOT / "output" / "convnext_hemorrhage.pth"
                if path_cv.exists():
                    model = get_pretrained_model(fine_tune=True)
                    model.load_state_dict(torch.load(path_cv, map_location=self.device, weights_only=True))
                    model.to(self.device).eval()
                    self.models["ConvNeXt"] = model
                
                # CustomCNN Load
                path_cc = ROOT / "output" / "customcnn_hemorrhage.pth"
                if path_cc.exists():
                    model = CustomCNN()
                    model.load_state_dict(torch.load(path_cc, map_location=self.device, weights_only=True))
                    model.to(self.device).eval()
                    self.models["CustomCNN"] = model

            except Exception as e:
                self.load_error = str(e)

            if callback: callback()

        threading.Thread(target=_load, daemon=True).start()

    def predict(self, image_path: str) -> dict:
        if not self.models:
            return {"error": "No models loaded. Please run train.py first."}

        try:
            import torch
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            results = {}
            with torch.no_grad():
                for name, model in self.models.items():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                    results[name] = {
                        "label": pred.item(),
                        "confidence": conf.item() * 100,
                        "prob_normal": probs[0, 0].item() * 100,
                        "prob_hemorrhage": probs[0, 1].item() * 100
                    }
            return results
        except Exception as e:
            return {"error": str(e)}

    @property
    def loaded_names(self) -> list:
        return list(self.models.keys())

    @property
    def count(self) -> int:
        return len(self.models)

# ══════════════════════════════════════════════════════════
# Toast (THREAD-SAFE SÜRÜM)
# ══════════════════════════════════════════════════════════
class Toast:
    _current = None
    @classmethod
    def show(cls, parent, message: str, kind: str = "info", duration: int = 3500):
        if cls._current and cls._current.winfo_exists():
            cls._current.destroy()

        colors = {
            "error":   (COLORS["accent_red"],   "#7f1d1d"),
            "success": (COLORS["accent_green"],  "#14532d"),
            "info":    (COLORS["accent_blue"],   "#1e3a5f"),
        }
        fg, bg = colors.get(kind, colors["info"])

        frame = ctk.CTkFrame(parent, fg_color=bg, border_color=fg, border_width=1, corner_radius=10)
        frame.place(relx=0.5, rely=0.97, anchor="s")

        ctk.CTkLabel(frame, text=message, font=ctk.CTkFont("Inter", 13, "bold"), text_color=COLORS["text_primary"], padx=18, pady=10).pack()
        cls._current = frame

        # Thread-safe dismissal
        parent.after(duration, lambda: frame.destroy() if frame.winfo_exists() else None)

# ══════════════════════════════════════════════════════════
# Performans Paneli
# ══════════════════════════════════════════════════════════
class PerformancePanel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent", corner_radius=0)
        self._build_notebook()
        self._load_content()

    def _build_notebook(self):
        self.tab_view = ctk.CTkTabview(
            self,
            fg_color=COLORS["bg_panel"],
            segmented_button_fg_color=COLORS["bg_card"],
            segmented_button_selected_color=COLORS["accent_red"],
            segmented_button_selected_hover_color=COLORS["accent_red2"],
            segmented_button_unselected_color=COLORS["bg_card"],
            segmented_button_unselected_hover_color=COLORS["hover"],
            text_color=COLORS["text_primary"],
        )
        self.tab_view.pack(fill="both", expand=True, padx=2, pady=2)
        self.tab_metrics = self.tab_view.add("📈 Metrics")
        self.tab_plots   = self.tab_view.add("🖼 Graphics")

    def _load_content(self):
        self._load_metrics()
        self._load_plots()

    def _load_metrics(self):
        output_dir = ROOT / "output"
        metrics_files = list(output_dir.glob("*_metrics.txt"))
        
        scroll = ctk.CTkScrollableFrame(self.tab_metrics, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        if not metrics_files:
            ctk.CTkLabel(scroll, text="No metrics found yet.", font=ctk.CTkFont("Inter", 14), text_color=COLORS["text_sec"]).pack(expand=True, pady=40)
            return

        for col_idx, m_path in enumerate(metrics_files):
            model_name = m_path.stem.replace("_metrics", "")
            metrics = self._parse_metrics(m_path)
            
            card = ctk.CTkFrame(scroll, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
            card.grid(row=0, column=col_idx, padx=10, pady=10, sticky="nsew")
            scroll.grid_columnconfigure(col_idx, weight=1)

            ctk.CTkLabel(card, text=model_name, font=ctk.CTkFont("Inter", 15, "bold"), text_color=COLORS["accent_red"]).pack(pady=(14, 4))
            
            for label, value in metrics.items():
                row_f = ctk.CTkFrame(card, fg_color="transparent")
                row_f.pack(fill="x", padx=16, pady=3)
                ctk.CTkLabel(row_f, text=label+":", font=ctk.CTkFont("Inter", 12), text_color=COLORS["text_sec"], width=100, anchor="w").pack(side="left")
                ctk.CTkLabel(row_f, text=value, font=ctk.CTkFont("Inter", 12, "bold"), text_color=COLORS["text_primary"]).pack(side="right")

    def _parse_metrics(self, path: Path) -> dict:
        metrics = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        metrics[k.strip()] = v.strip()
        except Exception: pass
        return metrics

    def _load_plots(self):
        plots_dir = ROOT / "output"
        parent = self.tab_plots
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        label_map = {
            "ConvNext_training": "ConvNeXt — Training History",
            "CustomCNN_training": "CustomCNN — Training History",
            "ConvNext_cm": "ConvNeXt — Confusion Matrix",
            "CustomCNN_cm": "CustomCNN — Confusion Matrix",
            "roc_curves_comparison": "ROC Curves Comparison",
            "model_comparison": "Model Metrics Comparison"
        }

        # Öncelikli sıralama
        order = ["roc_curves_comparison", "model_comparison", "ConvNext_training", "CustomCNN_training", "ConvNext_cm", "CustomCNN_cm"]
        
        found_any = False
        for stem in order:
            for ext in [".png", ".jpg"]:
                p_path = plots_dir / (stem + ext)
                if p_path.exists():
                    found_any = True
                    frame = ctk.CTkFrame(scroll, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
                    frame.pack(fill="x", padx=10, pady=10)
                    
                    title = label_map.get(stem, stem.replace("_", " ").title())
                    ctk.CTkLabel(frame, text=title, font=ctk.CTkFont("Inter", 14, "bold"), text_color=COLORS["text_primary"]).pack(pady=(12, 6))

                    try:
                        img = Image.open(p_path)
                        max_w = 840
                        ratio = max_w / img.width
                        new_h = int(img.height * ratio)
                        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(max_w, new_h))
                        lbl = ctk.CTkLabel(frame, image=ctk_img, text="")
                        lbl.pack(padx=20, pady=(0, 20), fill="both", expand=True)
                        lbl._img_ref = ctk_img
                    except Exception: pass
                    break
        
        if not found_any:
            ctk.CTkLabel(scroll, text="No graphics found.", text_color=COLORS["text_sec"]).pack(pady=40)

# ══════════════════════════════════════════════════════════
# Upload Panel
# ══════════════════════════════════════════════════════════
class UploadPanel(ctk.CTkFrame):
    def __init__(self, parent, on_file_selected, on_analyze, on_open_metrics):
        super().__init__(parent, fg_color=COLORS["bg_panel"], corner_radius=16, border_width=1, border_color=COLORS["border"])
        self.on_file_selected = on_file_selected
        self.on_analyze = on_analyze
        self.on_open_metrics = on_open_metrics
        self.current_file = None
        self._build()

    def _build(self):
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=20, pady=(20, 0))
        ctk.CTkLabel(hdr, text="Upload CT Image", font=ctk.CTkFont("Inter", 16, "bold"), text_color=COLORS["text_primary"]).pack(anchor="w")
        ctk.CTkLabel(hdr, text="Select a brain CT image in PNG, JPG, or JPEG format", font=ctk.CTkFont("Inter", 11), text_color=COLORS["text_sec"]).pack(anchor="w", pady=(2, 0))

        self.drop_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_input"], corner_radius=12, border_width=2, border_color=COLORS["border"], height=160)
        self.drop_frame.pack(fill="x", padx=20, pady=16)
        self.drop_frame.pack_propagate(False)
        ctk.CTkLabel(self.drop_frame, text="🖼", font=ctk.CTkFont(size=40)).pack(pady=(24, 4))
        ctk.CTkLabel(self.drop_frame, text="Click to select file", font=ctk.CTkFont("Inter", 13), text_color=COLORS["text_sec"]).pack()
        ctk.CTkLabel(self.drop_frame, text="Max: 20 MB · PNG · JPG · JPEG", font=ctk.CTkFont("Inter", 10), text_color=COLORS["text_muted"]).pack(pady=(2, 0))

        self.drop_frame.bind("<Button-1>", lambda e: self._browse())
        for w in self.drop_frame.winfo_children():
            w.bind("<Button-1>", lambda e: self._browse())
            w.configure(cursor="hand2")
        self.drop_frame.configure(cursor="hand2")

        self.preview_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_input"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        preview_hdr = ctk.CTkFrame(self.preview_frame, fg_color="transparent")
        preview_hdr.pack(fill="x", padx=12, pady=(12, 0))
        ctk.CTkLabel(preview_hdr, text="Uploaded Image", font=ctk.CTkFont("Inter", 12, "bold"), text_color=COLORS["text_sec"]).pack(side="left")
        self.clear_btn = ctk.CTkButton(preview_hdr, text="✕", width=26, height=26, fg_color=COLORS["bg_card"], hover_color=COLORS["accent_red"], text_color=COLORS["text_sec"], corner_radius=6, command=self._clear)
        self.clear_btn.pack(side="right")
        self.img_label = ctk.CTkLabel(self.preview_frame, text="")
        self.img_label.pack(pady=10)
        self.meta_label = ctk.CTkLabel(self.preview_frame, text="", font=ctk.CTkFont("Inter", 10), text_color=COLORS["text_muted"])
        self.meta_label.pack(pady=(0, 10))

        self.analyze_btn = ctk.CTkButton(self, text="🔬  Analyze", font=ctk.CTkFont("Inter", 14, "bold"), fg_color=COLORS["accent_red"], hover_color=COLORS["accent_red2"], height=46, corner_radius=10, command=self.on_analyze, state="disabled")
        self.analyze_btn.pack(fill="x", padx=20, pady=(0, 10))
        self.browse_btn = ctk.CTkButton(self, text="📂  Select File", font=ctk.CTkFont("Inter", 13), fg_color=COLORS["bg_card"], hover_color=COLORS["hover"], text_color=COLORS["text_primary"], border_width=1, border_color=COLORS["border"], height=38, corner_radius=10, command=self._browse)
        self.browse_btn.pack(fill="x", padx=20, pady=(0, 10))
        ctk.CTkButton(self, text="📊  Model Graphics", font=ctk.CTkFont("Inter", 12), fg_color="transparent", hover_color=COLORS["hover"], text_color=COLORS["text_sec"], border_width=1, border_color=COLORS["border"], height=34, corner_radius=10, command=self.on_open_metrics).pack(fill="x", padx=20, pady=(0, 20))

    def _browse(self):
        path = filedialog.askopenfilename(title="Select CT Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if path: self._load_file(path)

    def _load_file(self, path: str):
        self.current_file = path
        sz = os.path.getsize(path) / 1024 / 1024
        self.meta_label.configure(text=f"{Path(path).name}  ·  {sz:.2f} MB")
        if PIL_OK:
            img = load_ctk_image(path, (260, 200))
            self.img_label.configure(image=img)
            self.img_label._ref = img
        self.drop_frame.pack_forget()
        self.preview_frame.pack(fill="x", padx=20, pady=(0, 12))
        self.analyze_btn.configure(state="normal")
        self.on_file_selected(path)

    def _clear(self):
        self.current_file = None
        self.preview_frame.pack_forget()
        self.drop_frame.pack(fill="x", padx=20, pady=16)
        self.analyze_btn.configure(state="disabled")
        self.on_file_selected(None)

    def set_analyzing(self, active: bool):
        if active: self.analyze_btn.configure(text="⏳  Analyzing...", state="disabled", fg_color=COLORS["bg_card"])
        else:      self.analyze_btn.configure(text="🔬  Analyze", state="normal", fg_color=COLORS["accent_red"])

# ══════════════════════════════════════════════════════════
# Results Panel
# ══════════════════════════════════════════════════════════
class ResultsPanel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color=COLORS["bg_panel"], corner_radius=16, border_width=1, border_color=COLORS["border"])
        self._build()

    def _build(self):
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=20, pady=(20, 0))
        ctk.CTkLabel(hdr, text="Analysis Results", font=ctk.CTkFont("Inter", 16, "bold"), text_color=COLORS["text_primary"]).pack(anchor="w")
        ctk.CTkLabel(hdr, text="Consensus results of AI models", font=ctk.CTkFont("Inter", 11), text_color=COLORS["text_sec"]).pack(anchor="w", pady=(2, 16))
        ctk.CTkFrame(self, height=1, fg_color=COLORS["border"]).pack(fill="x", padx=20, pady=(0, 16))
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._show_empty()

    def _show_empty(self):
        self._clear()
        f = ctk.CTkFrame(self.scroll, fg_color="transparent")
        f.pack(expand=True, fill="both", pady=40)
        ctk.CTkLabel(f, text="🧠", font=ctk.CTkFont(size=56)).pack(pady=(20, 8))
        ctk.CTkLabel(f, text="Waiting for image", font=ctk.CTkFont("Inter", 15, "bold"), text_color=COLORS["text_sec"]).pack()
        ctk.CTkLabel(f, text="Select an image from the left side\nand click the Analyze button.", font=ctk.CTkFont("Inter", 12), text_color=COLORS["text_muted"], justify="center").pack(pady=(4, 0))

    def show_loading(self, step="AI is analyzing..."):
        self._clear()
        f = ctk.CTkFrame(self.scroll, fg_color="transparent")
        f.pack(expand=True, fill="both", pady=40)
        ctk.CTkLabel(f, text="⚙", font=ctk.CTkFont(size=48)).pack(pady=(20, 8))
        self._step_lbl = ctk.CTkLabel(f, text=step, font=ctk.CTkFont("Inter", 15, "bold"), text_color=COLORS["text_primary"])
        self._step_lbl.pack()
        self._progress = ctk.CTkProgressBar(f, width=220, height=6, fg_color=COLORS["bg_card"], progress_color=COLORS["accent_red"], corner_radius=4)
        self._progress.pack(pady=10); self._progress.set(0)
        self._animate_progress()

    def _animate_progress(self):
        if not hasattr(self, "_progress") or not self._progress.winfo_exists(): return
        val = (self._progress.get() + 0.015) % 1.05
        self._progress.set(min(val, 1.0))
        self.after(40, self._animate_progress)

    def show_results(self, data: dict):
        self._clear()
        if "error" in data:
            ctk.CTkLabel(self.scroll, text=f"⚠ {data['error']}", text_color=COLORS["accent_red"]).pack(pady=40); return

        m_names = [k for k in data if isinstance(data[k], dict)]
        if not m_names: return

        # Get findings per model
        findings = {n: ("Hemorrhage" if data[n]["label"] == 1 else "Normal") for n in m_names}
        unique_findings = set(findings.values())
        agrees = len(unique_findings) == 1
        has_hemo = "Hemorrhage" in unique_findings

        # Determine Banner Style
        if agrees:
            kind = "hemo" if has_hemo else "normal"
            title = "⚠ Consensus: Hemorrhage" if has_hemo else "✓ Consensus: Normal"
            sub_text = "Both models agree on the diagnosis."
        else:
            kind = "warn"
            title = "⚠ Conflict: Models Disagree"
            sub_text = "Divergent analysis — cautious review suggested."

        banner_bg = COLORS[f"bg_{kind}"]
        banner_brd = COLORS[f"{kind}_bar"]

        banner = ctk.CTkFrame(self.scroll, fg_color=banner_bg, corner_radius=12, border_width=2, border_color=banner_brd)
        banner.pack(fill="x", padx=8, pady=(8, 4))
        inner = ctk.CTkFrame(banner, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=14)
        
        icon = "⚠" if kind in ["hemo", "warn"] else "✓"
        ctk.CTkLabel(inner, text=icon, font=ctk.CTkFont(size=28), text_color=banner_brd).pack(side="left", padx=(0, 12))
        
        info = ctk.CTkFrame(inner, fg_color="transparent")
        info.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(info, text=title, font=ctk.CTkFont("Inter", 15, "bold"), text_color=COLORS["text_primary"]).pack(anchor="w")
        ctk.CTkLabel(info, text=sub_text, font=ctk.CTkFont("Inter", 11), text_color=COLORS["text_sec"]).pack(anchor="w")
        
        # Display separate status labels inside banner
        status_box = ctk.CTkFrame(inner, fg_color="transparent")
        status_box.pack(side="right", padx=(10, 0))
        for name in m_names:
            f = findings[name]
            f_clr = COLORS["accent_red"] if f == "Hemorrhage" else COLORS["accent_green"]
            ctk.CTkLabel(status_box, text=f"{name}: {f}", font=ctk.CTkFont("Inter", 10, "bold"), text_color=f_clr).pack(anchor="e")

        for name in m_names:
            self._add_model_card(name, data[name])

    def _add_model_card(self, name, res):
        is_h = res["label"] == 1
        brd = COLORS["accent_red"] if is_h else COLORS["accent_green"]
        c = ctk.CTkFrame(self.scroll, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=brd)
        c.pack(fill="x", padx=8, pady=6)
        h = ctk.CTkFrame(c, fg_color="transparent")
        h.pack(fill="x", padx=14, pady=(12, 4))
        ctk.CTkLabel(h, text=name, font=ctk.CTkFont("Inter", 14, "bold")).pack(side="left")
        ctk.CTkLabel(h, text=f"{'Hemorrhage' if is_h else 'Normal'} {res['confidence']:.1f}%", text_color=brd, font=ctk.CTkFont("Inter", 12, "bold")).pack(side="right")
        
        for lbl, p, clr in [("Normal", res["prob_normal"], COLORS["normal_bar"]), ("Hemorrhage", res["prob_hemorrhage"], COLORS["hemo_bar"])]:
            row = ctk.CTkFrame(c, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=3)
            ctk.CTkLabel(row, text=lbl, font=ctk.CTkFont("Inter", 11), text_color=COLORS["text_sec"], width=70, anchor="w").pack(side="left")
            bg = ctk.CTkFrame(row, fg_color=COLORS["bg_input"], height=8, corner_radius=4); bg.pack(side="left", fill="x", expand=True, padx=8); bg.pack_propagate(False)
            fill = ctk.CTkFrame(bg, fg_color=clr, height=8, corner_radius=4, width=4); fill.pack(side="left")
            self.after(100, lambda f=fill, b=bg, p=p: f.configure(width=max(int(b.winfo_width()*p/100), 4)))
            ctk.CTkLabel(row, text=f"{p:.1f}%", font=ctk.CTkFont("JetBrains Mono", 11, "bold"), width=50).pack(side="right")

    def _clear(self):
        for w in self.scroll.winfo_children(): w.destroy()

# ══════════════════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════════════════
class Header(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color=COLORS["bg_panel"], corner_radius=0, height=60)
        self.pack_propagate(False)
        self._build()
    def _build(self):
        l = ctk.CTkFrame(self, fg_color="transparent"); l.pack(side="left", padx=20)
        ctk.CTkLabel(l, text="🧠", font=ctk.CTkFont(size=28)).pack(side="left")
        t = ctk.CTkFrame(l, fg_color="transparent"); t.pack(side="left", padx=10)
        ctk.CTkLabel(t, text="Brain CT Hemorrhage Analysis", font=ctk.CTkFont("Inter", 18, "bold")).pack(anchor="w")

class NeuroScanApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Brain CT Hemorrhage Analysis")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        self.configure(fg_color=COLORS["bg"])
        self.mgr = ModelManager()
        self._build_ui()
        self.mgr.load(lambda: self.after(0, lambda: Toast.show(self, "Models Loaded Successfully", "success")))

    def _build_ui(self):
        Header(self).pack(fill="x")
        ctk.CTkFrame(self, height=1, fg_color=COLORS["border"]).pack(fill="x")
        c = ctk.CTkFrame(self, fg_color="transparent"); c.pack(fill="both", expand=True, padx=16, pady=16)
        c.grid_columnconfigure(0, weight=0, minsize=320)
        c.grid_columnconfigure(1, weight=1); c.grid_rowconfigure(0, weight=1)
        self.upload = UploadPanel(c, lambda p: None, self._on_analyze, lambda: self.tabs.set("📊 Performans"))
        self.upload.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.tabs = ctk.CTkTabview(c, fg_color="transparent", segmented_button_selected_color=COLORS["accent_red"], corner_radius=16)
        self.tabs.grid(row=0, column=1, sticky="nsew")
        self.results = ResultsPanel(self.tabs.add("🔬 Analysis"))
        self.results.pack(fill="both", expand=True)
        self.perf = PerformancePanel(self.tabs.add("📊 Performance"))
        self.perf.pack(fill="both", expand=True)

    def _on_analyze(self):
        if not self.upload.current_file: return
        self.upload.set_analyzing(True)
        self.results.show_loading()
        def _run():
            res = self.mgr.predict(self.upload.current_file)
            self.after(500, lambda: self._show_results(res))
        threading.Thread(target=_run, daemon=True).start()

    def _show_results(self, res):
        self.upload.set_analyzing(False)
        self.results.show_results(res)

if __name__ == "__main__":
    NeuroScanApp().mainloop()
