#!/usr/bin/env python3
"""Terroir AI — QA System Dashboard v10

FIXES:
1. Buttons invisible on macOS — replaced ALL tk.Button with Canvas-drawn
   CanvasButton class. macOS cannot override Canvas drawing, so colors are
   always exactly what we specify.
2. Roboflow_API is now the PRIMARY classifier. local_classify() is the
   fallback used only if Roboflow fails or times out (>8s).
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading, time, queue, os, sys, random, base64
import cv2, numpy as np
from PIL import Image, ImageTk, ImageDraw
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diseases import DISEASE_DB, TOMATO_QUALITY_DB, get_db, get_class_map
from config import CROPS, ROBOFLOW_API_KEY as _CFG_API_KEY

# ─────────────────────────────────────────────────────────────────────────────
#  ROBOFLOW CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ROBOFLOW_API_KEY  = _CFG_API_KEY
# Active crop — changed by the UI crop selector
_ACTIVE_CROP = "potato"

ROBOFLOW_CLASS_MAP = {
    "Potato":                 "Healthy",
    "Bud":                    "Healthy",
    "Damaged potato":         "Damaged_Potato",
    "Defected potato":        "Defected_Potato",
    "Diseased-fungal potato": "Diseased_Fungal_Potato",
    "Diseased/fungal potato": "Diseased_Fungal_Potato",
    "Sprouted potato":        "Sprouted_Potato",
    # Tomato classes — all known variants from this model
    "Good":        "Good",
    "good":        "Good",
    "Fresh":       "Good",
    "fresh":       "Good",
    "Ripe":        "Good",
    "ripe":        "Good",
    "Healthy":     "Good",   # if model uses Healthy for tomato
    "Bad":         "Bad",
    "bad":         "Bad",
    "Defective":   "Bad",
    "defective":   "Bad",
    "Defect":      "Bad",
    "defect":      "Bad",
    "Rotten":      "Bad",
    "rotten":      "Bad",
    "Damaged":     "Bad",
    "damaged":     "Bad",
    "Diseased":    "Bad",
    "diseased":    "Bad",
    "Unripe":      "Unripe",
    "unripe":      "Unripe",
    "Under-ripe":  "Unripe",
    "UnderRipe":   "Unripe",
    # Potato classes that might appear when wrong crop selected
    "Potato":                 "_WRONG_CROP_",
    "Bud":                    "_WRONG_CROP_",
    "Damaged potato":         "_WRONG_CROP_",
    "Defected potato":        "_WRONG_CROP_",
    "Diseased-fungal potato": "_WRONG_CROP_",
    "Sprouted potato":        "_WRONG_CROP_",
}

# ─────────────────────────────────────────────────────────────────────────────
#  DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":        "#09090B",
    "surface":   "#18181B",
    "surface2":  "#27272A",
    "surface3":  "#3F3F46",
    "border":    "#27272A",
    "border2":   "#3F3F46",
    "green":     "#22C55E",
    "green_dk":  "#16A34A",
    "green_bg":  "#052E16",
    "yellow":    "#EAB308",
    "yellow_dk": "#CA8A04",
    "yellow_bg": "#422006",
    "red":       "#EF4444",
    "red_dk":    "#DC2626",
    "red_bg":    "#450A0A",
    "blue":      "#3B82F6",
    "blue_dk":   "#2563EB",
    "blue_bg":   "#0F1C3F",
    "slate":     "#475569",
    "slate_lt":  "#64748B",
    "text":      "#F4F4F5",
    "text2":     "#A1A1AA",
    "text3":     "#71717A",
    "text4":     "#52525B",
    "white":     "#FFFFFF",
    "black":     "#000000",
}

GRADE_MAP = {
    "Healthy":                ("A", C["green"],  C["green_bg"],  "PASS",      "KEEP"),
    "Sprouted_Potato":        ("C", C["yellow"], C["yellow_bg"], "SECONDARY", "SECONDARY"),
    "Damaged_Potato":         ("D", C["red"],    C["red_bg"],    "FAIL",      "REJECT"),
    "Defected_Potato":        ("D", C["red"],    C["red_bg"],    "FAIL",      "REJECT"),
    "Diseased_Fungal_Potato": ("F", C["red"],    C["red_bg"],    "FAIL",      "REJECT"),
    "Late_Blight":            ("F", C["red"],    C["red_bg"],    "FAIL",      "REJECT"),
    "Early_Blight":           ("D", C["red"],    C["red_bg"],    "FAIL",      "REJECT"),
    "Dry_Rot":                ("F", C["red"],    C["red_bg"],    "FAIL",      "REJECT"),
    "Common_Scab":            ("C", C["yellow"], C["yellow_bg"], "SECONDARY", "SECONDARY"),
    "Silver_Scurf":           ("B", C["yellow"], C["yellow_bg"], "WARN",      "KEEP"),
    "Black_Dot":              ("B", C["yellow"], C["yellow_bg"], "WARN",      "KEEP"),
    "Soft_Rot":               ("F", C["red"],    C["red_bg"],    "FAIL",      "REJECT"),
    "Wart_Disease":           ("F", C["red"],    C["red_bg"],    "FAIL",      "REJECT"),
    "Mechanical_Damage":      ("C", C["yellow"], C["yellow_bg"], "SECONDARY", "SECONDARY"),
    "Gangrene":               ("F", C["red"],    C["red_bg"],    "FAIL",      "REJECT"),
}

TOMATO_GRADE_MAP = {
    "Good":   ("A", C["green"],  C["green_bg"],  "PASS",      "KEEP"),
    "Bad":    ("D", C["red"],    C["red_bg"],    "FAIL",      "REJECT"),
    "Unripe": ("B", C["yellow"], C["yellow_bg"], "SECONDARY", "HOLD"),
}

ACTION_COLOR = {
    "KEEP":        C["green"],
    "REJECT":      C["red"],
    "SECONDARY":   C["yellow"],
    "REVIEW":      C["blue"],
    "CONDITIONAL": C["yellow"],
}
ACTION_BG = {
    "KEEP":        C["green_bg"],
    "REJECT":      C["red_bg"],
    "SECONDARY":   C["yellow_bg"],
    "REVIEW":      C["blue_bg"],
    "CONDITIONAL": C["yellow_bg"],
}

F = {
    "h2":      ("Helvetica", 15, "bold"),
    "h3":      ("Helvetica", 11, "bold"),
    "body_sm": ("Helvetica", 10),
    "caption": ("Helvetica",  9),
    "mono":    ("Courier",   10),
    "mono_sm": ("Courier",    9),
    "stat_lg": ("Helvetica", 26, "bold"),
    "badge":   ("Helvetica",  9, "bold"),
    "grade":   ("Helvetica", 72, "bold"),
    "btn":     ("Helvetica", 10, "bold"),
}


# ─────────────────────────────────────────────────────────────────────────────
#  CANVAS BUTTON — macOS-proof, always renders with correct colors
# ─────────────────────────────────────────────────────────────────────────────
class CanvasButton(tk.Frame):
    """
    A button drawn on a tk.Canvas inside a tk.Frame.
    Uses Frame as the outer container so pack/grid work normally.
    Canvas drawing is immune to macOS Aqua theme overrides.

    NOTE: width/height are NOT passed to tk.Frame.__init__ because Python 3.13
    on macOS misinterprets integer kwargs as window path strings. Instead we
    call self.configure() after super().__init__() completes.
    """
    def __init__(self, parent, text, bg, fg, hover_bg, command,
                 width=140, height=36, radius=5, font=None):
        # Resolve parent bg safely
        try:
            pbg = parent.cget("bg")
        except Exception:
            pbg = C["bg"]

        # Do NOT pass width/height here — configure them after init
        super().__init__(parent, bg=pbg, highlightthickness=0, bd=0)
        # Now safe to set dimensions
        self.configure(width=int(width), height=int(height))
        self.pack_propagate(False)

        self._text     = text
        self._bg       = bg
        self._fg       = fg
        self._hover_bg = hover_bg
        self._cmd      = command
        self._bw       = int(width)
        self._bh       = int(height)
        self._r        = radius
        self._font     = font or F["btn"]
        self._hovered  = False

        # Canvas inside the frame — also configure dimensions after init
        self._canvas = tk.Canvas(self, bg=pbg, highlightthickness=0, bd=0,
                                  cursor="hand2")
        self._canvas.configure(width=int(width), height=int(height))
        self._canvas.pack(fill="both", expand=True)

        # Defer first draw until widget is realized by the event loop
        self._canvas.after(1, self._draw)
        self._canvas.bind("<Enter>",    self._on_enter)
        self._canvas.bind("<Leave>",    self._on_leave)
        self._canvas.bind("<Button-1>", self._on_click)
        self.bind("<Button-1>", self._on_click)

    def _draw(self):
        self._canvas.delete("all")
        bg = self._hover_bg if self._hovered else self._bg
        r, w, h = self._r, self._bw, self._bh
        pts = [
            r, 0,  w-r, 0,
            w, 0,  w, r,
            w, h-r, w, h,
            w-r, h, r, h,
            0, h,  0, h-r,
            0, r,  0, 0,
            r, 0,
        ]
        self._canvas.create_polygon(pts, fill=bg, outline=bg, smooth=True)
        self._canvas.create_text(w//2, h//2, text=self._text,
                                  fill=self._fg, font=self._font, anchor="center")

    def _on_enter(self, _):
        self._hovered = True;  self._draw()

    def _on_leave(self, _):
        self._hovered = False; self._draw()

    def _on_click(self, _):
        if self._cmd: self._cmd()

    def set_text(self, text):
        self._text = text; self._draw()

    def set_colors(self, bg, fg, hover_bg):
        self._bg = bg; self._fg = fg; self._hover_bg = hover_bg; self._draw()


# ─────────────────────────────────────────────────────────────────────────────
#  CLASSIFIERS
# ─────────────────────────────────────────────────────────────────────────────
def roboflow_classify(rgb: np.ndarray, timeout_s: float = 8.0, crop: str = None) -> dict | None:
    """Call Roboflow API for the given crop. Returns dict on success, None on any failure."""
    global _ACTIVE_CROP
    crop = crop or _ACTIVE_CROP
    try:
        cfg  = CROPS[crop]
        model_path = f"{cfg['project']}/{cfg['version']}"
        t0   = time.time()
        bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        if h < 416 or w < 416:
            bgr = cv2.resize(bgr, (416, 416))
        _, buf  = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        encoded = base64.b64encode(buf.tobytes()).decode("utf-8")
        resp = requests.post(
            f"https://serverless.roboflow.com/{model_path}?api_key={ROBOFLOW_API_KEY}",
            data=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout_s,
        )
        data  = resp.json()
        preds = data.get("predictions", [])
        if not preds:
            print(f"[Roboflow] No predictions for {crop} — using local fallback")
            return None
        best    = max(preds, key=lambda x: x["confidence"])
        raw_cls = best["class"]
        class_map = get_class_map(crop)
        disease = class_map.get(raw_cls, raw_cls.replace(" ", "_"))
        conf    = best["confidence"]
        print(f"[Roboflow] {crop} → class='{raw_cls}' disease='{disease}' conf={conf:.1%} ({len(preds)} predictions)")

        # ── Wrong-crop detection ─────────────────────────────────────────────────
        # Detect tomato-like classes when in potato mode
        TOMATO_SIGNALS = {"Fresh", "fresh", "Good", "good", "Bad", "bad",
                          "Ripe", "ripe", "Unripe", "unripe", "Defective", "defective"}
        POTATO_SIGNALS = {"Potato", "Bud", "Damaged potato", "Defected potato",
                          "Diseased-fungal potato", "Sprouted potato"}
        wrong_crop_hint = None
        if crop == "potato" and raw_cls in TOMATO_SIGNALS:
            wrong_crop_hint = "tomato"
        elif crop == "tomato" and raw_cls in POTATO_SIGNALS:
            wrong_crop_hint = "potato"

        if disease == "_WRONG_CROP_":
            return {
                "disease":        "_WRONG_CROP_",
                "confidence":     conf,
                "action":         "REVIEW",
                "classifier":     "Roboflow_API",
                "notifiable":     False,
                "spread_risk":    "",
                "treatment":      "",
                "market_value":   "",
                "description":    f"Detected as {raw_cls} — switch crop mode.",
                "category":       "WRONG_CROP",
                "surface_pct":    0.0,
                "inference_ms":   (time.time() - t0) * 1000,
                "crop":           crop,
                "wrong_crop_hint": wrong_crop_hint or ("tomato" if crop == "potato" else "potato"),
                "raw_class":      raw_cls,
            }

        db      = get_db(crop)
        info    = db.get(disease, {})
        result  = {
            "disease":      disease,
            "confidence":   conf,
            "action":       info.get("action", "KEEP"),
            "classifier":   "Roboflow_API",
            "notifiable":   info.get("notifiable", False),
            "spread_risk":  info.get("spread_risk", ""),
            "treatment":    info.get("treatment", ""),
            "market_value": info.get("market_value", ""),
            "description":  info.get("description", ""),
            "category":     info.get("category", ""),
            "surface_pct":  0.0,
            "inference_ms": (time.time() - t0) * 1000,
            "crop":         crop,
        }
        if wrong_crop_hint:
            result["wrong_crop_hint"] = wrong_crop_hint
        return result
    except Exception as e:
        print(f"[Roboflow] Error ({crop}): {e}")
        return None


def local_classify(rgb: np.ndarray) -> dict:
    """Pure-OpenCV fallback. No network. Always works."""
    t0   = time.time()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lesion_ratio = mask.sum() / (255 * h * w)

    kern  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_lesions  = len(cnts)
    areas      = [cv2.contourArea(c) for c in cnts]
    max_lesion = max(areas) / (h * w) if areas else 0.0

    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_var = np.sqrt(sx**2 + sy**2).var() / (255**2)

    r_m = rgb[:, :, 0].mean() / 255
    g_m = rgb[:, :, 1].mean() / 255
    b_m = rgb[:, :, 2].mean() / 255
    brown      = r_m - g_m * 0.8 - b_m * 0.5
    dark_ratio = (gray < 70).sum() / (h * w)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].mean() / 255
    val = hsv[:, :, 2].mean() / 255

    sc = {
        "Late_Blight":  dark_ratio * 4.0 + lesion_ratio * 2.5 + brown * 2.0 + (1 - val) * 1.5,
        "Early_Blight": lesion_ratio * 2.5 + brown * 2.0 + edge_var * 2.5 + n_lesions * 0.12,
        "Soft_Rot":     dark_ratio * 3.5 + (1 - sat) * 2.5 + lesion_ratio * 1.5,
        "Common_Scab":  edge_var * 3.0 + lesion_ratio * 1.8 + n_lesions * 0.10,
        "Silver_Scurf": (1 - sat) * 2.5 + (1 - dark_ratio) * 1.5 + lesion_ratio * 1.0,
        "Healthy":      val * 3.5 + (1 - lesion_ratio) * 3.0 + (1 - dark_ratio) * 2.5 + sat * 0.5,
    }
    total = sum(max(v, 0.01) for v in sc.values())
    pr    = {k: max(v, 0.01) / total for k, v in sc.items()}
    best  = max(pr, key=pr.get)
    vals  = sorted(pr.values(), reverse=True)
    conf  = min(0.88, 0.50 + (vals[0] - vals[1]) * 1.8)

    info   = DISEASE_DB.get(best, {})
    action = info.get("action", "KEEP")
    if best == "Common_Scab":
        action = "REJECT" if lesion_ratio > info.get("scab_threshold", 0.10) else "SECONDARY"

    return {
        "disease":      best,
        "confidence":   conf,
        "action":       action,
        "classifier":   "CNN_Visual (fallback)",
        "notifiable":   info.get("notifiable", False),
        "spread_risk":  info.get("spread_risk", ""),
        "treatment":    info.get("treatment", ""),
        "market_value": info.get("market_value", ""),
        "description":  info.get("description", ""),
        "category":     info.get("category", ""),
        "surface_pct":  lesion_ratio,
        "n_lesions":    n_lesions,
        "max_lesion":   max_lesion,
        "inference_ms": (time.time() - t0) * 1000,
    }


# Minimum confidence to trust a potato model result.
# The potato Roboflow model legitimately returns 45–65% on real potatoes.
# Only flag as wrong crop if confidence is VERY low (below 0.35).
_POTATO_CONF_THRESHOLD = 0.35


def classify_image(rgb: np.ndarray, crop: str = None) -> dict:
    """Roboflow primary → local fallback. Crop-aware with wrong-crop detection."""
    global _ACTIVE_CROP
    crop   = crop or _ACTIVE_CROP
    result = roboflow_classify(rgb, timeout_s=8.0, crop=crop)
    if result is None:
        # Local fallback only available for potato
        if crop == "potato":
            result = local_classify(rgb)
        else:
            # Safe default for tomato when Roboflow is unreachable
            result = {
                "disease":      "Good",
                "confidence":   0.50,
                "action":       "REVIEW",
                "classifier":   "Fallback (no network)",
                "notifiable":   False,
                "spread_risk":  "",
                "treatment":    "",
                "market_value": "",
                "description":  "Network unavailable — manual inspection required.",
                "category":     "",
                "surface_pct":  0.0,
                "inference_ms": 0.0,
                "crop":         crop,
            }

    if "crop" not in result:
        result["crop"] = crop

    # ── Confidence-based wrong-crop detection (potato mode only) ──────────────────
    # The potato model has never seen a tomato, so it guesses with low confidence.
    # If confidence is below threshold AND it's not already flagged as wrong crop,
    # override the result to show the "Not a Potato" warning.
    if (crop == "potato"
            and result.get("disease") != "_WRONG_CROP_"
            and result.get("classifier") not in ("Simulation", "Fallback (no network)")
            and result.get("confidence", 1.0) < _POTATO_CONF_THRESHOLD):
        raw_cls = result.get("disease", "Unknown").replace("_", " ")
        result = {
            "disease":         "_WRONG_CROP_",
            "confidence":      result.get("confidence", 0.0),
            "action":          "REVIEW",
            "classifier":      result.get("classifier", "Roboflow_API"),
            "notifiable":      False,
            "spread_risk":     "",
            "treatment":       "",
            "market_value":    "",
            "description":     f"Low confidence — not a potato. Switch to Tomato mode.",
            "category":        "WRONG_CROP",
            "surface_pct":     0.0,
            "inference_ms":    result.get("inference_ms", 0.0),
            "crop":            "potato",
            "wrong_crop_hint": "tomato",
            "raw_class":       raw_cls,
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  INLINE LOGGER
# ─────────────────────────────────────────────────────────────────────────────
class _Logger:
    def __init__(self):
        import csv
        from datetime import datetime
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_path   = os.path.join(log_dir, f"session_{self.session_id}.csv")
        self._lock      = threading.Lock()
        self._n         = 0
        self.start_time = time.time()
        self.total = self.kept = self.rejected = self.secondary = 0
        self.value_saved = 0.0
        self._fields = ["timestamp","tuber_id","session_id","disease","classifier",
                        "confidence","action","surface_pct","inference_ms","notifiable"]
        with open(self.log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fields).writeheader()

    def log(self, r: dict) -> int:
        import csv
        from datetime import datetime
        with self._lock:
            self._n += 1
            tid = self._n
            row = {
                "timestamp":    datetime.now().isoformat(),
                "tuber_id":     tid,
                "session_id":   self.session_id,
                "disease":      r.get("disease", "Unknown"),
                "classifier":   r.get("classifier", "—"),
                "confidence":   round(r.get("confidence", 0), 4),
                "action":       r.get("action", "KEEP"),
                "surface_pct":  round(r.get("surface_pct", 0), 4),
                "inference_ms": round(r.get("inference_ms", 0), 2),
                "notifiable":   r.get("notifiable", False),
            }
            with open(self.log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self._fields).writerow(row)
            self.total += 1
            a = row["action"]
            if a == "KEEP":      self.kept      += 1
            elif a == "REJECT":  self.rejected  += 1; self.value_saved += 0.32
            else:                self.secondary += 1; self.value_saved += 0.08
        return tid

    def stats(self) -> dict:
        with self._lock:
            elapsed = max(1, time.time() - self.start_time)
            rr = (self.rejected / max(1, self.total)) * 100
            return dict(
                total=self.total, kept=self.kept,
                rejected=self.rejected, secondary=self.secondary,
                reject_rate=round(rr, 1),
                value_saved=round(self.value_saved, 2),
                rate_per_min=round(self.total / elapsed * 60, 1),
                log_path=self.log_path,
            )


# ─────────────────────────────────────────────────────────────────────────────
#  SCAN OVERLAY  — stipple glow, no 8-digit hex
# ─────────────────────────────────────────────────────────────────────────────
class ScanOverlay:
    def __init__(self, canvas):
        self.canvas   = canvas
        self._y       = 0
        self._dir     = 1
        self._running = False

    def start(self):
        if self._running: return
        self._running = True
        self._animate()

    def stop(self):
        self._running = False
        try: self.canvas.delete("scanline")
        except Exception: pass

    def _animate(self):
        if not self._running: return
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        if W < 10:
            self.canvas.after(50, self._animate); return
        try: self.canvas.delete("scanline")
        except Exception: pass

        for i in range(5, 0, -1):
            sp = "gray12" if i >= 4 else ("gray25" if i >= 2 else "gray50")
            if self._y - i > 0:
                self.canvas.create_line(0, self._y-i, W, self._y-i,
                                        fill=C["green"], width=1, stipple=sp, tags="scanline")
            if self._y + i < H:
                self.canvas.create_line(0, self._y+i, W, self._y+i,
                                        fill=C["green"], width=1, stipple=sp, tags="scanline")
        self.canvas.create_line(0, self._y, W, self._y,
                                fill=C["green"], width=2, tags="scanline")
        self._y += self._dir * 4
        if self._y >= H: self._dir = -1
        if self._y <= 0: self._dir =  1
        self.canvas.after(18, self._animate)


# ─────────────────────────────────────────────────────────────────────────────
#  CONVEYOR CANVAS
# ─────────────────────────────────────────────────────────────────────────────
class ConveyorCanvas(tk.Canvas):
    def __init__(self, parent, on_scan_cb, **kw):
        kw.pop("bg", None)
        super().__init__(parent, bg=C["surface"], highlightthickness=0, **kw)
        self.on_scan_cb  = on_scan_cb
        self._potatoes   = []
        self._next_spawn = 50
        self._frame      = 0
        self._running    = False
        self._scan_x_pct = 0.55
        self._processed  = set()
        self._results    = {}

    def start(self):
        self._running = True; self._animate()

    def stop(self):
        self._running = False

    def mark_result(self, pid, action):
        self._results[pid] = ACTION_COLOR.get(action, C["text3"])

    def _animate(self):
        if not self._running: return
        self._frame += 1
        self.delete("all")
        W = self.winfo_width(); H = self.winfo_height()
        if W < 10:
            self.after(33, self._animate); return

        BY1, BY2 = int(H * 0.10), int(H * 0.90)
        scan_x   = int(W * self._scan_x_pct)

        self.create_rectangle(0, BY1, W, BY2, fill="#1A1A1E", outline="")
        off = (self._frame * 3) % 40
        for i in range(-1, W // 40 + 2):
            x = i * 40 + off
            self.create_rectangle(x, BY1+4, x+20, BY2-4, fill="#1F1F24", outline="")
        self.create_rectangle(0, BY1, W, BY1+4, fill=C["border2"], outline="")
        self.create_rectangle(0, BY2-4, W, BY2,  fill=C["border2"], outline="")

        for i in range(5, 0, -1):
            sp = "gray12" if i > 4 else ("gray25" if i > 2 else "")
            kw = dict(fill=C["green"], width=i)
            if sp: kw["stipple"] = sp
            self.create_line(scan_x, BY1, scan_x, BY2, **kw)

        self.create_rectangle(scan_x+3, 3, scan_x+46, 18,
                               fill=C["green_bg"], outline=C["green"], width=1)
        self.create_text(scan_x+25, 11, text="SCAN", fill=C["green"],
                         font=("Helvetica", 7, "bold"))

        self._next_spawn -= 1
        if self._next_spawn <= 0:
            self._next_spawn = max(12, int(random.gauss(44, 8)))
            self._spawn(BY1, BY2)

        bm = (BY1 + BY2) // 2
        dead = []
        for p in self._potatoes:
            p["x"] += p["spd"]
            if p["x"] > W + 80:
                dead.append(p); continue
            col = self._results.get(p["id"])
            self._draw_potato(p, bm, col)
            if p["x"] >= scan_x and p["id"] not in self._processed:
                self._processed.add(p["id"])
                self.on_scan_cb(p)
        for p in dead:
            self._potatoes.remove(p)
        if len(self._processed) > 400:
            self._processed = set(list(self._processed)[-150:])
        self.after(33, self._animate)

    def _spawn(self, BY1, BY2):
        mid = (BY1 + BY2) // 2
        sz  = random.randint(26, 48)
        sh  = random.randint(100, 175)
        pid = self._frame * 100 + len(self._potatoes)
        self._potatoes.append({
            "id": pid, "x": -sz,
            "y":  mid + random.randint(-12, 12),
            "sz": sz,  "spd": random.uniform(2.0, 3.6),
            "col": (int(sh*.46), int(sh*.66), sh),
        })

    def _draw_potato(self, p, mid, rc=None):
        x, y, s = int(p["x"]), int(p["y"]), p["sz"]
        r, g, b  = p["col"]
        fill = f"#{r:02x}{g:02x}{b:02x}"
        hi   = f"#{min(255,r+40):02x}{min(255,g+40):02x}{min(255,b+40):02x}"
        self.create_oval(x-s+3, y-int(s*.72)+4, x+s+3, y+int(s*.72)+4,
                         fill="#040406", outline="")
        oc = rc if rc else C["border"]; ow = 2 if rc else 1
        self.create_oval(x-s, y-int(s*.74), x+s, y+int(s*.74),
                         fill=fill, outline=oc, width=ow)
        self.create_oval(x-s//2-2, y-s//2, x+s//8, y-s//8, fill=hi, outline="")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class TerriorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Terroir AI — QA System")
        self.root.configure(bg=C["bg"])
        self.root.geometry("1540x920")
        self.root.minsize(1200, 760)

        self.logger       = _Logger()
        self._running     = False
        self._result_q    = queue.Queue(maxsize=100)
        self._cam         = None
        self._cam_running = False
        self._img_ref     = None
        self._last_result = None
        self._hist_rows   = []
        self._sim_crop    = "potato"   # tracks active crop for simulation & load-image

        self._build_ui()
        self._tick_clock()
        self._poll_results()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── BUILD UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()
        tk.Frame(self.root, bg=C["border2"], height=1).pack(fill="x")
        self._build_body()
        self._build_footer()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=C["bg"], height=60)
        hdr.pack(fill="x"); hdr.pack_propagate(False)

        ll = tk.Frame(hdr, bg=C["bg"])
        ll.pack(side="left", padx=20, pady=14)
        tk.Label(ll, text="TERROIR AI", font=("Helvetica", 17, "bold"),
                 bg=C["bg"], fg=C["green"]).pack(side="left")
        tk.Label(ll, text="  QA SYSTEM", font=("Helvetica", 11),
                 bg=C["bg"], fg=C["text3"]).pack(side="left", pady=2)

        rr = tk.Frame(hdr, bg=C["bg"])
        rr.pack(side="right", padx=20, pady=10)

        self.clock_lbl = tk.Label(rr, text="", font=F["mono"],
                                   bg=C["surface"], fg=C["text2"], padx=12, pady=5)
        self.clock_lbl.pack(side="right", padx=(8, 0))

        sf = tk.Frame(rr, bg=C["green_bg"], padx=12, pady=5)
        sf.pack(side="right", padx=(8, 0))
        self._pulse_dot = tk.Label(sf, text="●", font=("Helvetica", 11),
                                    bg=C["green_bg"], fg=C["green"])
        self._pulse_dot.pack(side="left")
        tk.Label(sf, text="  SYSTEM ONLINE", font=F["badge"],
                 bg=C["green_bg"], fg=C["green"]).pack(side="left")

        self.inf_badge = tk.Label(rr, text="— ms", font=F["badge"],
                                   bg=C["blue_bg"], fg=C["blue"], padx=10, pady=5)
        self.inf_badge.pack(side="right", padx=(8, 0))

        self.clf_badge = tk.Label(rr, text="Roboflow_API", font=F["badge"],
                                   bg=C["surface2"], fg=C["text2"], padx=10, pady=5)
        self.clf_badge.pack(side="right", padx=(8, 0))

        # ── Crop Selector ─────────────────────────────────────────────────────
        crop_frame = tk.Frame(rr, bg=C["surface2"], padx=2, pady=2)
        crop_frame.pack(side="right", padx=(8, 0))
        tk.Label(crop_frame, text="CROP", font=F["badge"],
                 bg=C["surface2"], fg=C["text4"], padx=6).pack(side="left")
        self._crop_var = tk.StringVar(value="potato")
        crop_opts = list(CROPS.keys())
        crop_menu = tk.OptionMenu(crop_frame, self._crop_var, *crop_opts,
                                   command=self._on_crop_change)
        crop_menu.config(
            bg=C["surface3"], fg=C["text"], activebackground=C["green_bg"],
            activeforeground=C["green"], relief="flat", bd=0,
            font=F["badge"], padx=8, pady=4, highlightthickness=0,
            indicatoron=True,
        )
        crop_menu["menu"].config(
            bg=C["surface2"], fg=C["text"], activebackground=C["green_bg"],
            activeforeground=C["green"], font=F["badge"],
        )
        crop_menu.pack(side="left")

    def _on_crop_change(self, crop: str):
        """Called when user picks a different crop from the header dropdown."""
        global _ACTIVE_CROP
        _ACTIVE_CROP = crop
        crop_label = CROPS.get(crop, {}).get("label", crop.title())
        # Update camera panel title
        self.cam_panel_title.config(text=f"LIVE CAMERA FEED  ·  {crop_label.upper()}")
        # Update footer
        self.footer_lbl.config(text=f"Terroir AI QA System  ·  Crop switched to {crop_label}")
        # Update simulation to use correct disease pool
        self._sim_crop = crop
        # Reset grade card
        self.grade_lbl.config(text="—", fg=C["text3"])
        self.disease_lbl.config(text="Awaiting scan...", fg=C["text"])
        self.grade_desc.config(text=f"Switch to {crop_label} mode. Load an image or start scan.", fg=C["text3"])
        self.action_badge.config(text="STANDBY", bg=C["surface2"], fg=C["text3"])
        self.scan_status.config(text="● STANDBY", bg=C["surface2"], fg=C["text3"])
        # Clear history
        for row in self._hist_rows:
            row.destroy()
        self._hist_rows.clear()
        self._draw_placeholder()
        print(f"[Crop] Switched to {crop} — model: {CROPS[crop]['project']}/{CROPS[crop]['version']}")

    def _build_body(self):
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=14, pady=10)

        left = tk.Frame(body, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        right = tk.Frame(body, bg=C["bg"], width=430)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._build_camera_panel(left)
        self._build_conveyor_panel(left)
        self._build_right_panel(right)

    def _build_camera_panel(self, parent):
        panel = self._card(parent)
        panel.pack(fill="both", expand=True, pady=(0, 8))

        ph = tk.Frame(panel, bg=C["surface"])
        ph.pack(fill="x", padx=16, pady=(12, 8))
        self.cam_panel_title = tk.Label(ph, text="LIVE CAMERA FEED  ·  POTATO", font=F["h3"],
                 bg=C["surface"], fg=C["text2"])
        self.cam_panel_title.pack(side="left")
        self.ts_lbl = tk.Label(ph, text="", font=F["mono_sm"],
                                bg=C["surface"], fg=C["text4"])
        self.ts_lbl.pack(side="right", padx=(12, 0))
        self.scan_status = tk.Label(ph, text="● STANDBY", font=F["badge"],
                                     bg=C["surface2"], fg=C["text3"], padx=10, pady=4)
        self.scan_status.pack(side="right")

        cam_wrap = tk.Frame(panel, bg=C["border2"], padx=1, pady=1)
        cam_wrap.pack(fill="both", expand=True, padx=16)
        self.cam_canvas = tk.Canvas(cam_wrap, bg="#0A0A0D", highlightthickness=0)
        self.cam_canvas.pack(fill="both", expand=True)
        self._scan_overlay = ScanOverlay(self.cam_canvas)
        self.cam_canvas.after(200, self._draw_placeholder)

        # ── CANVAS BUTTONS — macOS-proof ──────────────────────────────────────
        btn_row = tk.Frame(panel, bg=C["surface"])
        btn_row.pack(fill="x", padx=16, pady=(10, 14))

        self._btn_start = CanvasButton(
            btn_row, text="▶  START SCAN",
            bg=C["green"], fg=C["black"], hover_bg=C["green_dk"],
            command=self._toggle_sim, width=148, height=38)
        self._btn_start.pack(side="left", padx=(0, 8))

        CanvasButton(
            btn_row, text="⬆  LOAD IMAGE",
            bg=C["slate"], fg=C["white"], hover_bg=C["slate_lt"],
            command=self._load_image, width=140, height=38
        ).pack(side="left", padx=(0, 8))

        CanvasButton(
            btn_row, text="📷  LIVE CAMERA",
            bg=C["slate"], fg=C["white"], hover_bg=C["slate_lt"],
            command=self._toggle_camera, width=148, height=38
        ).pack(side="left", padx=(0, 8))

        CanvasButton(
            btn_row, text="⬇  EXPORT LOG",
            bg=C["slate"], fg=C["white"], hover_bg=C["slate_lt"],
            command=self._export_log, width=140, height=38
        ).pack(side="right")

    def _build_conveyor_panel(self, parent):
        panel = self._card(parent)
        panel.pack(fill="x")

        ph = tk.Frame(panel, bg=C["surface"])
        ph.pack(fill="x", padx=16, pady=(10, 4))
        tk.Label(ph, text="CONVEYOR SIMULATION", font=F["h3"],
                 bg=C["surface"], fg=C["text2"]).pack(side="left")
        tk.Label(ph, text="Real-time belt tracking", font=F["caption"],
                 bg=C["surface"], fg=C["text4"]).pack(side="left", padx=8)

        wrap = tk.Frame(panel, bg=C["border2"], padx=1, pady=1)
        wrap.pack(fill="x", padx=16, pady=(0, 12))
        self.conveyor = ConveyorCanvas(wrap, on_scan_cb=self._on_conveyor_scan, height=90)
        self.conveyor.pack(fill="x")

    def _build_right_panel(self, parent):
        self._build_grade_card(parent)
        self._build_stats_row(parent)
        self._build_details_card(parent)
        self._build_history_card(parent)

    def _build_grade_card(self, parent):
        card = self._card(parent)
        card.pack(fill="x", pady=(0, 8))

        header = tk.Frame(card, bg=C["surface"])
        header.pack(fill="x", padx=16, pady=(12, 0))
        tk.Label(header, text="CURRENT ITEM STATUS", font=F["h3"],
                 bg=C["surface"], fg=C["text2"]).pack(side="left")
        self.action_badge = tk.Label(header, text="STANDBY", font=F["badge"],
                                      bg=C["surface2"], fg=C["text3"], padx=10, pady=4)
        self.action_badge.pack(side="right")

        gf = tk.Frame(card, bg=C["surface"])
        gf.pack(fill="x", padx=16, pady=(6, 4))
        self.grade_lbl = tk.Label(gf, text="—", font=F["grade"],
                                   bg=C["surface"], fg=C["text4"], width=2, anchor="w")
        self.grade_lbl.pack(side="left")

        ic = tk.Frame(gf, bg=C["surface"])
        ic.pack(side="left", padx=(8, 0), fill="y", expand=True)
        self.disease_lbl = tk.Label(ic, text="Awaiting scan...",
                                     font=F["h2"], bg=C["surface"], fg=C["text"], anchor="w")
        self.disease_lbl.pack(anchor="w")
        self.grade_desc = tk.Label(ic, text="No item detected",
                                    font=F["body_sm"], bg=C["surface"], fg=C["text3"],
                                    anchor="w", wraplength=230, justify="left")
        self.grade_desc.pack(anchor="w", pady=(4, 0))

        cf = tk.Frame(card, bg=C["surface"])
        cf.pack(fill="x", padx=16, pady=(0, 6))
        ch = tk.Frame(cf, bg=C["surface"])
        ch.pack(fill="x")
        tk.Label(ch, text="CONFIDENCE", font=F["badge"],
                 bg=C["surface"], fg=C["text4"]).pack(side="left")
        self.conf_pct = tk.Label(ch, text="—", font=F["badge"],
                                  bg=C["surface"], fg=C["text3"])
        self.conf_pct.pack(side="right")
        track = tk.Frame(cf, bg=C["surface3"], height=7)
        track.pack(fill="x", pady=(4, 0))
        self.conf_bar = tk.Frame(track, bg=C["text4"], height=7)
        self.conf_bar.place(relwidth=0.0, relheight=1.0)
        tk.Frame(card, bg=C["surface"], height=10).pack()

    def _build_stats_row(self, parent):
        row = tk.Frame(parent, bg=C["bg"])
        row.pack(fill="x", pady=(0, 8))
        self._stats = {}
        for i, (key, label, color) in enumerate([
            ("total",  "TOTAL SCANNED", C["text"]),
            ("defect", "DEFECT RATE",   C["red"]),
            ("rate",   "ITEMS / MIN",   C["blue"]),
        ]):
            card = self._card(row)
            card.pack(side="left", fill="both", expand=True,
                      padx=(0, 6) if i < 2 else (0, 0))
            tk.Label(card, text=label, font=F["badge"],
                     bg=C["surface"], fg=C["text3"], pady=8).pack()
            val = tk.Label(card, text="0", font=F["stat_lg"],
                           bg=C["surface"], fg=color)
            val.pack(pady=(0, 10))
            self._stats[key] = val

    def _build_details_card(self, parent):
        card = self._card(parent)
        card.pack(fill="x", pady=(0, 8))
        tk.Label(card, text="DISEASE DETAILS", font=F["h3"],
                 bg=C["surface"], fg=C["text2"],
                 padx=16, pady=10).pack(anchor="w")
        tk.Frame(card, bg=C["border"], height=1).pack(fill="x", padx=16)
        self._detail_fields = {}
        for label, key in [
            ("PATHOGEN",    "path_lbl"),
            ("DESCRIPTION", "desc_lbl"),
            ("TREATMENT",   "tx_lbl"),
            ("MARKET",      "mkt_lbl"),
            ("SPREAD RISK", "risk_lbl"),
        ]:
            r = tk.Frame(card, bg=C["surface"])
            r.pack(fill="x", padx=16, pady=3)
            tk.Label(r, text=label, font=F["badge"],
                     bg=C["surface"], fg=C["text4"],
                     width=11, anchor="w").pack(side="left")
            lbl = tk.Label(r, text="—", font=F["caption"],
                           bg=C["surface"], fg=C["text3"],
                           anchor="w", wraplength=240, justify="left")
            lbl.pack(side="left", fill="x", expand=True)
            self._detail_fields[key] = lbl
        tk.Frame(card, bg=C["surface"], height=10).pack()

    def _build_history_card(self, parent):
        card = self._card(parent)
        card.pack(fill="both", expand=True)

        hdr = tk.Frame(card, bg=C["surface"])
        hdr.pack(fill="x", padx=16, pady=(12, 6))
        tk.Label(hdr, text="RECENT SCANS", font=F["h3"],
                 bg=C["surface"], fg=C["text2"]).pack(side="left")
        self.hist_count = tk.Label(hdr, text="Last 8 items", font=F["caption"],
                                    bg=C["surface"], fg=C["text4"])
        self.hist_count.pack(side="right")

        col_hdr = tk.Frame(card, bg=C["surface2"])
        col_hdr.pack(fill="x", padx=16)
        for txt, w, anchor in [
            ("#", 4, "center"), ("TIME", 7, "w"),
            ("GR", 3, "center"), ("DISEASE", 0, "w"), ("CONF", 5, "e"),
        ]:
            tk.Label(col_hdr, text=txt, font=F["badge"],
                     bg=C["surface2"], fg=C["text4"],
                     width=w, anchor=anchor, pady=5).pack(
                side="left",
                padx=(8 if txt == "#" else 0, 0),
                fill="x" if txt == "DISEASE" else None,
                expand=(txt == "DISEASE"))

        self.hist_frame = tk.Frame(card, bg=C["surface"])
        self.hist_frame.pack(fill="both", expand=True, padx=16, pady=(4, 12))

    def _build_footer(self):
        tk.Frame(self.root, bg=C["border2"], height=1).pack(fill="x")
        foot = tk.Frame(self.root, bg=C["surface"], height=28)
        foot.pack(fill="x"); foot.pack_propagate(False)
        self.footer_lbl = tk.Label(foot, text="Terroir AI QA System  ·  Ready",
                                    font=F["mono_sm"], bg=C["surface"], fg=C["text4"])
        self.footer_lbl.pack(side="left", padx=16, pady=5)
        tk.Label(foot, text="Powered by Roboflow_API  ·  Terroir AI  ·  McKinney TX",
                 font=F["caption"], bg=C["surface"], fg=C["text4"]).pack(side="right", padx=16)

    # ── HELPERS ───────────────────────────────────────────────────────────────
    def _card(self, parent):
        return tk.Frame(parent, bg=C["surface"],
                        highlightbackground=C["border"],
                        highlightthickness=1)

    def _tick_clock(self):
        self.clock_lbl.config(text=time.strftime("  %a %b %d  %H:%M:%S  "))
        now = int(time.time() * 2) % 2
        self._pulse_dot.config(fg=C["green"] if now else C["green_dk"])
        self.root.after(500, self._tick_clock)

    # ── SIMULATION ────────────────────────────────────────────────────────────
    def _toggle_sim(self):
        if not self._running:
            self._running = True
            self._btn_start.set_text("⏹  STOP SCAN")
            self._btn_start.set_colors(C["red"], C["white"], C["red_dk"])
            self.scan_status.config(text="● SCANNING", bg=C["green_bg"], fg=C["green"])
            self.conveyor.start()
            self._scan_overlay.start()
        else:
            self._running = False
            self._btn_start.set_text("▶  START SCAN")
            self._btn_start.set_colors(C["green"], C["black"], C["green_dk"])
            self.scan_status.config(text="● STANDBY", bg=C["surface2"], fg=C["text3"])
            self.conveyor.stop()
            self._scan_overlay.stop()
            self._draw_placeholder()

    def _on_conveyor_scan(self, potato):
        crop = getattr(self, "_sim_crop", "potato")
        roll = random.random()
        if crop == "tomato":
            if   roll < 0.65: disease, conf = "Good",   random.uniform(0.80, 0.97)
            elif roll < 0.85: disease, conf = "Unripe", random.uniform(0.70, 0.88)
            else:             disease, conf = "Bad",    random.uniform(0.72, 0.90)
            db = TOMATO_QUALITY_DB
        else:
            if   roll < 0.62: disease, conf = "Healthy",                random.uniform(0.80, 0.97)
            elif roll < 0.75: disease, conf = "Sprouted_Potato",        random.uniform(0.70, 0.88)
            elif roll < 0.84: disease, conf = "Damaged_Potato",         random.uniform(0.72, 0.90)
            elif roll < 0.92: disease, conf = "Diseased_Fungal_Potato", random.uniform(0.74, 0.92)
            else:             disease, conf = "Defected_Potato",        random.uniform(0.68, 0.86)
            db = DISEASE_DB

        info = db.get(disease, {})
        result = {
            "disease":      disease,
            "confidence":   conf,
            "action":       info.get("action", "KEEP"),
            "classifier":   "Simulation",
            "notifiable":   info.get("notifiable", False),
            "spread_risk":  info.get("spread_risk", ""),
            "treatment":    info.get("treatment", ""),
            "market_value": info.get("market_value", ""),
            "description":  info.get("description", ""),
            "category":     info.get("category", ""),
            "surface_pct":  0.0,
            "inference_ms": random.uniform(18, 55),
            "crop":         crop,
        }
        try: self._result_q.put_nowait(("sim", result, None, potato["id"]))
        except queue.Full: pass

    # ── IMAGE LOAD ────────────────────────────────────────────────────────────
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select Potato Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All", "*.*")]
        )
        if not path: return

        bgr = cv2.imread(path)
        if bgr is None:
            messagebox.showerror("Error", f"Cannot read image:\n{path}")
            return

        if self._running:
            self._running = False
            self.conveyor.stop()

        crop = getattr(self, "_sim_crop", "potato")
        self.footer_lbl.config(text=f"Analysing via Roboflow [{crop}]: {os.path.basename(path)}...")
        self.scan_status.config(text="● ANALYSING", bg=C["blue_bg"], fg=C["blue"])
        self._scan_overlay.start()
        self._show_frame(bgr, result=None)

        rgb_copy = cv2.cvtColor(bgr.copy(), cv2.COLOR_BGR2RGB)
        bgr_copy = bgr.copy()

        def _run():
            result = classify_image(rgb_copy, crop=getattr(self, "_sim_crop", "potato"))   # Roboflow primary
            try:
                self._result_q.put_nowait(("image", result, bgr_copy, None))
            except queue.Full:
                try: self._result_q.get_nowait()
                except Exception: pass
                try: self._result_q.put_nowait(("image", result, bgr_copy, None))
                except Exception: pass

        threading.Thread(target=_run, daemon=True).start()

    # ── CAMERA ────────────────────────────────────────────────────────────────
    def _toggle_camera(self):
        if self._cam_running:
            self._cam_running = False
            if self._cam: self._cam.release()
            self._scan_overlay.stop()
            self.scan_status.config(text="● STANDBY", bg=C["surface2"], fg=C["text3"])
        else:
            self._cam = cv2.VideoCapture(0)
            if not self._cam.isOpened():
                messagebox.showerror("Camera Error", "Cannot open camera.")
                return
            self._cam_running = True
            self._latest_frame = None
            self._last_clf = 0.0
            self._scan_overlay.start()
            self.scan_status.config(text="● LIVE", bg=C["green_bg"], fg=C["green"])
            threading.Thread(target=self._cam_reader, daemon=True).start()
            self._cam_display_loop()

    def _cam_reader(self):
        while self._cam_running:
            ret, frame = self._cam.read()
            if ret: self._latest_frame = frame
            time.sleep(0.033)

    def _cam_display_loop(self):
        if not self._cam_running: return
        f = getattr(self, "_latest_frame", None)
        if f is not None:
            self._show_frame(f, self._last_result)
            if time.time() - self._last_clf > 2.0:
                self._last_clf = time.time()
                rgb = cv2.cvtColor(f.copy(), cv2.COLOR_BGR2RGB)
                bgr_c = f.copy()
                def _run(r=rgb, b=bgr_c):
                    res = classify_image(r)
                    try: self._result_q.put_nowait(("cam", res, b, None))
                    except queue.Full: pass
                threading.Thread(target=_run, daemon=True).start()
        self.root.after(40, self._cam_display_loop)

    # ── SHOW FRAME ────────────────────────────────────────────────────────────
    def _show_frame(self, bgr, result=None):
        self.cam_canvas.update_idletasks()
        W = self.cam_canvas.winfo_width()  or 600
        H = self.cam_canvas.winfo_height() or 300
        if W < 10 or H < 10: return

        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img  = Image.fromarray(rgb).resize((W, H), Image.LANCZOS)
        draw = ImageDraw.Draw(img)

        if result:
            action  = result.get("action", "KEEP")
            disease = result.get("disease", "")
            col_hex = ACTION_COLOR.get(action, "#666666")
            cr = int(col_hex[1:3], 16)
            cg = int(col_hex[3:5], 16)
            cb = int(col_hex[5:7], 16)
            for t in range(3):
                draw.rectangle([t, t, W-1-t, H-1-t], outline=(cr, cg, cb))
            entry = GRADE_MAP.get(disease, ("?", "", "", "?", "?"))
            grade, _, _, status, _ = entry
            tag = f"  {status} — Grade {grade}  "
            tw  = len(tag) * 7 + 4
            draw.rectangle([6, 6, 6+tw, 26], fill=(cr//6, cg//6, cb//6))
            draw.text((8, 8), tag, fill=(cr, cg, cb))

        tk_img = ImageTk.PhotoImage(img)
        self.cam_canvas.delete("all")
        self.cam_canvas.create_image(0, 0, anchor="nw", image=tk_img)
        self._img_ref = tk_img
        self.ts_lbl.config(text=time.strftime("%H:%M:%S"))

    # ── RESULT CARD ON CANVAS ─────────────────────────────────────────────────
    def _show_result_card(self, result):
        self.cam_canvas.update_idletasks()
        W = self.cam_canvas.winfo_width()  or 600
        H = self.cam_canvas.winfo_height() or 300

        disease = result.get("disease", "?")
        conf    = result.get("confidence", 0.0)
        action  = result.get("action", "KEEP")
        crop    = result.get("crop", "potato")
        hint    = result.get("wrong_crop_hint")

        # ── Wrong-crop special card ─────────────────────────────────────────────────
        if disease == "_WRONG_CROP_":
            raw = result.get("raw_class", "unknown")
            suggest = hint.title() if hint else "the other crop"
            pad = 16; cw = W - pad * 2; ch = 100
            cx  = pad; cy = H - ch - pad
            self.cam_canvas.create_rectangle(cx, cy, cx+cw, cy+ch,
                                              fill="#18181B", outline=C["yellow"], width=2,
                                              tags="result_card")
            self.cam_canvas.create_rectangle(cx, cy, cx+cw, cy+4,
                                              fill=C["yellow"], outline="", tags="result_card")
            self.cam_canvas.create_text(cx+52, cy+52,
                                         text="!", fill=C["yellow"],
                                         font=("Helvetica", 46, "bold"),
                                         tags="result_card")
            self.cam_canvas.create_text(cx+110, cy+28,
                                         text=f"Wrong Crop Detected",
                                         fill=C["yellow"], font=("Helvetica", 14, "bold"),
                                         anchor="w", tags="result_card")
            self.cam_canvas.create_text(cx+110, cy+50,
                                         text=f'Model saw "{raw}" — switch to {suggest.upper()} mode',
                                         fill="#A1A1AA", font=("Helvetica", 10),
                                         anchor="w", tags="result_card")
            self.cam_canvas.create_text(cx+110, cy+70,
                                         text=f"Use the CROP dropdown in the header ↑",
                                         fill="#71717A", font=("Helvetica", 9),
                                         anchor="w", tags="result_card")
            return

        gmap    = TOMATO_GRADE_MAP if crop == "tomato" else GRADE_MAP
        entry   = gmap.get(disease, ("?", C["text3"], C["surface2"], "?", "REVIEW"))
        grade, col, bg, status, _ = entry

        pad = 16; cw = W - pad * 2; ch = 120
        cx  = pad; cy = H - ch - pad

        self.cam_canvas.create_rectangle(cx, cy, cx+cw, cy+ch,
                                          fill="#18181B", outline=col, width=2,
                                          tags="result_card")
        self.cam_canvas.create_rectangle(cx, cy, cx+cw, cy+4,
                                          fill=col, outline="", tags="result_card")
        self.cam_canvas.create_text(cx+52, cy+58,
                                     text=grade, fill=col,
                                     font=("Helvetica", 52, "bold"),
                                     tags="result_card")
        self.cam_canvas.create_text(cx+110, cy+26,
                                     text=disease.replace("_", " "),
                                     fill=col, font=("Helvetica", 14, "bold"),
                                     anchor="w", tags="result_card")
        self.cam_canvas.create_text(cx+110, cy+48,
                                     text=f"{status}  ·  {action}",
                                     fill=col, font=("Helvetica", 10),
                                     anchor="w", tags="result_card")

        bx1, by1 = cx+110, cy+66
        bx2, by2 = cx+cw-16, cy+74
        self.cam_canvas.create_rectangle(bx1, by1, bx2, by2,
                                          fill="#3F3F46", outline="", tags="result_card")
        bar_w = int((bx2-bx1) * min(conf, 1.0))
        self.cam_canvas.create_rectangle(bx1, by1, bx1+bar_w, by2,
                                          fill=col, outline="", tags="result_card")
        self.cam_canvas.create_text(cx+110, cy+86,
                                     text=f"Confidence: {conf*100:.1f}%  ·  {result.get('classifier','—')}",
                                     fill="#A1A1AA", font=("Helvetica", 9),
                                     anchor="w", tags="result_card")
        crop = result.get("crop", "potato")
        info = get_db(crop).get(disease, {})
        desc = info.get("description", "")[:72]
        self.cam_canvas.create_text(cx+cw-16, cy+104,
                                     text=desc, fill="#71717A",
                                     font=("Helvetica", 8),
                                     anchor="e", tags="result_card")

    # ── PLACEHOLDER ───────────────────────────────────────────────────────────
    def _draw_placeholder(self):
        self.cam_canvas.delete("all")
        W = self.cam_canvas.winfo_width()  or 600
        H = self.cam_canvas.winfo_height() or 300
        for i in range(0, W, 50):
            self.cam_canvas.create_line(i, 0, i, H, fill="#111115", width=1)
        for i in range(0, H, 50):
            self.cam_canvas.create_line(0, i, W, i, fill="#111115", width=1)
        cx, cy = W//2, H//2
        for r in [55, 85, 115]:
            self.cam_canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                         outline=C["border2"], width=1)
        self.cam_canvas.create_line(cx-90, cy, cx+90, cy, fill=C["border2"], width=1)
        self.cam_canvas.create_line(cx, cy-90, cx, cy+90, fill=C["border2"], width=1)
        self.cam_canvas.create_oval(cx-3, cy-3, cx+3, cy+3, fill=C["text4"], outline="")
        self.cam_canvas.create_text(cx, cy+130,
                                     text="Press  ▶ START SCAN  to begin  ·  or load an image",
                                     fill=C["text4"], font=F["caption"])
        sz = 22
        for bx, by in [(20, 20), (W-20, 20), (20, H-20), (W-20, H-20)]:
            dx = 1 if bx < W//2 else -1
            dy = 1 if by < H//2 else -1
            self.cam_canvas.create_line(bx, by, bx+dx*sz, by, fill=C["text3"], width=2)
            self.cam_canvas.create_line(bx, by, bx, by+dy*sz, fill=C["text3"], width=2)

    # ── POLL RESULTS ──────────────────────────────────────────────────────────
    def _poll_results(self):
        try:
            while not self._result_q.empty():
                item   = self._result_q.get_nowait()
                kind   = item[0]
                result = item[1]
                bgr    = item[2]
                pid    = item[3] if len(item) > 3 else None

                tid = self.logger.log(result)
                s   = self.logger.stats()
                self._last_result = result

                self._update_grade_card(result)
                self._update_details(result)
                self._update_stats(s)
                self._add_history(result, tid)

                if kind == "image":
                    self._scan_overlay.stop()
                    if bgr is not None:
                        self._show_frame(bgr, result)
                    self._show_result_card(result)
                    self.scan_status.config(text="● COMPLETE",
                                             bg=C["green_bg"], fg=C["green"])
                    self.root.after(5000, lambda: self.scan_status.config(
                        text="● STANDBY", bg=C["surface2"], fg=C["text3"]))

                elif kind == "cam":
                    pass

                else:
                    dummy = self._make_dummy(result)
                    self._show_frame(dummy, result)
                    if pid is not None:
                        self.conveyor.mark_result(pid, result.get("action", "KEEP"))

                inf_ms  = result.get("inference_ms", 0)
                clf     = result.get("classifier", "—")
                disease = result.get("disease", "?").replace("_", " ")
                action  = result.get("action", "KEEP")
                self.inf_badge.config(text=f"{inf_ms:.0f} ms")
                self.clf_badge.config(text=clf)
                self.footer_lbl.config(
                    text=(f"Terroir AI  ·  #{tid}  {action}  ·  {disease}"
                          f"  {result.get('confidence', 0)*100:.0f}%  ·  {inf_ms:.0f} ms")
                )
        except Exception as e:
            print(f"[poll error] {e}")
        self.root.after(80, self._poll_results)

    # ── UPDATE HELPERS ────────────────────────────────────────────────────────
    def _update_grade_card(self, result):
        disease = result.get("disease", "?")
        conf    = result.get("confidence", 0.0)
        action  = result.get("action", "KEEP")
        crop    = result.get("crop", "potato")
        hint    = result.get("wrong_crop_hint")

        # ── Wrong-crop: show amber warning in the grade card ─────────────────────────
        if disease == "_WRONG_CROP_":
            raw     = result.get("raw_class", "?")
            suggest = hint.title() if hint else "other"
            self.grade_lbl.config(text="!", fg=C["yellow"])
            self.disease_lbl.config(text="Wrong Crop", fg=C["yellow"])
            self.grade_desc.config(
                text=f'Saw "{raw}" — switch to {suggest} mode',
                fg=C["yellow"])
            self.conf_bar.place(relwidth=min(conf, 1.0), relheight=1.0)
            self.conf_bar.config(bg=C["yellow"])
            self.conf_pct.config(text=f"{conf*100:.1f}%", fg=C["yellow"])
            self.action_badge.config(text="REVIEW",
                                      bg=C["surface2"], fg=C["yellow"])
            return

        gmap    = TOMATO_GRADE_MAP if crop == "tomato" else GRADE_MAP
        entry   = gmap.get(disease, ("?", C["text3"], C["surface2"], "?", "REVIEW"))
        grade, col, bg, status, _ = entry
        db   = get_db(crop)
        info = db.get(disease, {})

        self.grade_lbl.config(text=grade, fg=col)
        self.disease_lbl.config(text=disease.replace("_", " "), fg=col)
        self.grade_desc.config(text=info.get("description", "—")[:60], fg=C["text3"])
        self.conf_bar.place(relwidth=min(conf, 1.0), relheight=1.0)
        self.conf_bar.config(bg=col)
        self.conf_pct.config(text=f"{conf*100:.1f}%", fg=col)
        self.action_badge.config(
            text=action,
            bg=ACTION_BG.get(action, C["surface2"]),
            fg=col)

    def _update_details(self, result):
        disease = result.get("disease", "—")
        crop    = result.get("crop", "potato")
        info    = get_db(crop).get(disease, {})
        action  = result.get("action", "KEEP")
        col     = ACTION_COLOR.get(action, C["text3"])
        self._detail_fields["path_lbl"].config(text=disease.replace("_", " "), fg=col)
        self._detail_fields["desc_lbl"].config(text=info.get("description", "—"), fg=C["text3"])
        self._detail_fields["tx_lbl"].config(
            text=info.get("treatment", "—"),
            fg=C["yellow"] if action == "REJECT" else C["text3"])
        self._detail_fields["mkt_lbl"].config(text=info.get("market_value", "—"), fg=C["text3"])
        risk = info.get("spread_risk", "—")
        self._detail_fields["risk_lbl"].config(
            text=risk,
            fg=C["red"] if "CRITICAL" in risk else C["text3"])

    def _update_stats(self, s):
        total    = s.get("total", 0)
        rejected = s.get("rejected", 0)
        rate     = s.get("rate_per_min", 0.0)
        dr = f"{(rejected/total*100):.1f}%" if total > 0 else "0.0%"
        self._stats["total"].config(text=str(total))
        self._stats["defect"].config(text=dr)
        self._stats["rate"].config(text=f"{rate:.0f}")

    def _add_history(self, result, tid):
        disease = result.get("disease", "?")
        conf    = result.get("confidence", 0.0)
        crop    = result.get("crop", "potato")
        gmap    = TOMATO_GRADE_MAP if crop == "tomato" else GRADE_MAP
        entry   = gmap.get(disease, ("?", C["text3"], C["surface2"], "?", "REVIEW"))
        grade, col, bg, status, _ = entry
        ts = time.strftime("%H:%M:%S")

        row = tk.Frame(self.hist_frame, bg=C["surface2"], pady=4)
        row.pack(fill="x", pady=1)
        tk.Frame(row, bg=col, width=3).pack(side="left", fill="y")
        tk.Label(row, text=f"{tid:03d}", font=F["mono_sm"],
                 bg=C["surface2"], fg=C["text4"], width=4).pack(side="left", padx=4)
        tk.Label(row, text=ts, font=F["mono_sm"],
                 bg=C["surface2"], fg=C["text4"], width=8).pack(side="left")
        tk.Label(row, text=f" {grade} ", font=F["badge"],
                 bg=bg, fg=col, width=3).pack(side="left", padx=6)
        tk.Label(row, text=disease.replace("_", " ")[:22], font=F["caption"],
                 bg=C["surface2"], fg=C["text"], anchor="w").pack(
            side="left", fill="x", expand=True)
        tk.Label(row, text=f"{conf*100:.0f}%", font=F["mono_sm"],
                 bg=C["surface2"], fg=col, width=4).pack(side="right", padx=6)

        self._hist_rows.append(row)
        if len(self._hist_rows) > 8:
            self._hist_rows.pop(0).destroy()
        self.hist_count.config(text=f"{self.logger.stats().get('total', 0)} total scanned")

    def _make_dummy(self, result):
        W, H = 600, 320
        img  = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = (9, 9, 12)
        cv2.rectangle(img, (0, H//4), (W, 3*H//4), (18, 20, 26), -1)
        cx, cy = W//2, H//2
        cv2.ellipse(img, (cx, cy), (105, 82), 0, 0, 360, (70, 108, 148), -1)
        cv2.ellipse(img, (cx-28, cy-22), (35, 27), 20, 0, 360, (100, 138, 180), -1)
        if result.get("action") == "REJECT":
            for _ in range(random.randint(2, 5)):
                lx = cx + random.randint(-60, 60)
                ly = cy + random.randint(-40, 40)
                cv2.ellipse(img, (lx, ly),
                            (random.randint(8, 28), random.randint(6, 18)),
                            random.randint(0, 180), 0, 360, (8, 12, 5), -1)
        return img

    # ── LOG EXPORT ────────────────────────────────────────────────────────────
    def _export_log(self):
        s    = self.logger.stats()
        path = s.get("log_path", "")
        if path and os.path.exists(path):
            messagebox.showinfo("Session Log",
                                f"Log saved:\n{path}\n\n"
                                f"Scanned: {s['total']}  ·  Rejected: {s['rejected']}\n"
                                f"Value Protected: ${s.get('value_saved', 0):.2f}")
        else:
            messagebox.showinfo("Log", "No session data yet.")

    # ── CLOSE ─────────────────────────────────────────────────────────────────
    def _on_close(self):
        self._running     = False
        self._cam_running = False
        self.conveyor.stop()
        self._scan_overlay.stop()
        if self._cam: self._cam.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = TerriorApp()
    app.run()
